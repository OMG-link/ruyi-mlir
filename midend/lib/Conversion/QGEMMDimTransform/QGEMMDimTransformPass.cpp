//===- QGEMMDimTransformPass.cpp - QGEMM shape simplification -*- C++ -*-===//
//
// Licensed under the Apache License v2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file implements the scoped dimension-transform simplification pass used
// before QGEMM recognition. The pass deliberately does not normalize arbitrary
// tensor programs: only the coarse scopes reconstructed from candidate roots
// are made available to the standard reshape folding and forwarding patterns.
//
//===----------------------------------------------------------------------===//

#include "../QGEMMRecognize/Boundary.h"
#include "../QGEMMRecognize/CandidateRoots.h"
#include "../QGEMMRecognize/OperatorWalk.h"

#include "Conversion/QGEMMDimTransform/QGEMMDimTransformPass.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

using ScopedOperations = llvm::SetVector<Operation *>;

/// Return true only for the final-reduction shape accepted by recognition.
static bool hasRankTwoResult(const buddy::QGEMMCandidateRoot &candidate) {
  if (candidate.reduce->getNumResults() != 1)
    return false;
  auto resultType =
      dyn_cast<RankedTensorType>(candidate.reduce->getResult(0).getType());
  return resultType && resultType.getRank() == 2;
}

/// Return true when an operation belongs to the function being simplified.
static bool isInFunction(func::FuncOp funcOp, Operation *operation) {
  if (!operation)
    return false;
  func::FuncOp parent = operation->getParentOfType<func::FuncOp>();
  return parent && parent.getOperation() == funcOp.getOperation();
}

/// Add an operation to the deterministic coarse-scope list when it belongs to
/// the function being simplified.
static void addScopedOperation(func::FuncOp funcOp, Operation *operation,
                               ScopedOperations &scopedOperations,
                               llvm::DenseSet<Operation *> &scopedSet) {
  if (!isInFunction(funcOp, operation))
    return;
  if (scopedSet.insert(operation).second)
    scopedOperations.insert(operation);
}

/// Follow the memory-side source path hidden behind a tensor boundary.
///
/// OperatorWalk intentionally stops at bufferization.to_tensor and memref
/// results. The layout expression for an input is nevertheless part of the
/// simplifier's coarse scope. In the captured IR the source path commonly has
/// the form
///
///   memref.reinterpret_cast -> memref.copy -> memref.alloc
///       -> bufferization.to_tensor
///
/// so the walk below follows the defining operands and, for an allocation,
/// the copy that fills it. It only crosses the source-producing memref ops
/// listed here; arbitrary users and unrelated memory computations remain out
/// of scope.
static void
collectMemrefSourceChain(func::FuncOp funcOp, Value value, Operation *producer,
                         ScopedOperations &scopedOperations,
                         llvm::DenseSet<Operation *> &scopedSet,
                         llvm::DenseSet<Value> &visitedValues,
                         llvm::DenseSet<Operation *> &visitedOperations) {
  if (!value || !visitedValues.insert(value).second)
    return;

  if (!producer)
    producer = value.getDefiningOp();
  if (!isInFunction(funcOp, producer))
    return;
  if (!visitedOperations.insert(producer).second)
    return;

  // A boundary which is already admitted by the shared walk is not a source
  // chain to recover. This keeps the source tracing tied to Boundary's
  // classification rather than introducing a second admission predicate.
  if (!buddy::isQGEMMOperatorBoundary(producer))
    return;

  if (isa<bufferization::ToTensorOp>(producer)) {
    addScopedOperation(funcOp, producer, scopedOperations, scopedSet);
    collectMemrefSourceChain(funcOp, producer->getOperand(0), nullptr,
                             scopedOperations, scopedSet, visitedValues,
                             visitedOperations);
    return;
  }

  if (isa<memref::AllocOp>(producer)) {
    addScopedOperation(funcOp, producer, scopedOperations, scopedSet);
    Value allocation = producer->getResult(0);
    for (Operation *user : allocation.getUsers()) {
      auto copy = dyn_cast<memref::CopyOp>(user);
      if (!copy || copy.getTarget() != allocation)
        continue;
      addScopedOperation(funcOp, copy.getOperation(), scopedOperations,
                         scopedSet);
      collectMemrefSourceChain(funcOp, copy.getSource(), nullptr,
                               scopedOperations, scopedSet, visitedValues,
                               visitedOperations);
    }
    return;
  }

  // Reinterpret-cast is the required memref-side dimension transform. The
  // other source-only view transforms are included so a static chain can be
  // traced through them without admitting unrelated memref computations.
  if (isa<memref::ReinterpretCastOp, memref::CastOp, memref::SubViewOp,
          memref::ExpandShapeOp, memref::CollapseShapeOp>(producer)) {
    addScopedOperation(funcOp, producer, scopedOperations, scopedSet);
    if (producer->getNumOperands() != 0)
      collectMemrefSourceChain(funcOp, producer->getOperand(0), nullptr,
                               scopedOperations, scopedSet, visitedValues,
                               visitedOperations);
  }
}

/// Build the operation list that the scoped greedy driver is allowed to see.
static ScopedOperations collectCoarseScope(func::FuncOp funcOp) {
  ScopedOperations scopedOperations;
  llvm::DenseSet<Operation *> scopedSet;
  llvm::DenseSet<Value> visitedBoundaryValues;
  llvm::DenseSet<Operation *> visitedSourceOperations;

  for (const buddy::QGEMMCandidateRoot &candidate :
       buddy::findQGEMMCandidateRoots(funcOp)) {
    if (!hasRankTwoResult(candidate))
      continue;

    buddy::QGEMMOperatorWalk walk = buddy::walkQGEMMOperator(candidate);
    for (Operation *operation : walk.operations)
      addScopedOperation(funcOp, operation, scopedOperations, scopedSet);

    for (const buddy::QGEMMBoundary &boundary : walk.boundaries) {
      collectMemrefSourceChain(funcOp, boundary.value, boundary.producer,
                               scopedOperations, scopedSet,
                               visitedBoundaryValues, visitedSourceOperations);
    }
  }

  // Tensor empty values are ordinary walk boundaries, but their operations
  // need to be in the explicit list when they are used by a scoped operation.
  // This lets the standard greedy driver's dead-operation handling visit an
  // empty after a scoped reshape rewrite removes its last use, without
  // admitting unrelated empty tensors from the function.
  funcOp.walk([&](tensor::EmptyOp emptyOp) {
    Value result = emptyOp.getResult();
    if (llvm::any_of(result.getUsers(),
                     [&](Operation *user) { return scopedSet.contains(user); }))
      addScopedOperation(funcOp, emptyOp.getOperation(), scopedOperations,
                         scopedSet);
  });

  // OperatorWalk intentionally builds its set in backwards traversal order.
  // Sort the explicit operation list by stable function IR order so pattern
  // selection does not depend on traversal or allocation order.
  llvm::DenseMap<Operation *, unsigned> irOrder;
  unsigned ordinal = 0;
  funcOp.walk(
      [&](Operation *operation) { irOrder.try_emplace(operation, ordinal++); });
  llvm::SmallVector<Operation *> ordered(scopedOperations.begin(),
                                         scopedOperations.end());
  llvm::stable_sort(ordered, [&](Operation *lhs, Operation *rhs) {
    return irOrder.lookup(lhs) < irOrder.lookup(rhs);
  });
  scopedOperations.clear();
  scopedOperations.insert(ordered.begin(), ordered.end());

  return scopedOperations;
}

/// Return true when an indexing map contains a constant result. Triton-shared
/// uses such results for broadcast dimensions; normalizing those maps through
/// the upstream dropUnitDims utility produces a projected permutation while
/// preserving the tensor semantics.
static bool hasConstantIndexingMapResult(linalg::GenericOp genericOp) {
  return llvm::any_of(genericOp.getIndexingMapsArray(), [](AffineMap map) {
    return llvm::any_of(map.getResults(),
                        [](AffineExpr expr) { return isa<AffineConstantExpr>(expr); });
  });
}

/// Normalize scoped broadcast generics once before the greedy rewrite. Only
/// the generics selected by the initial coarse scope are eligible; the scope
/// is reconstructed by the caller after replacements create reshapes and
/// collapse operands.
static void normalizeScopedBroadcasts(
    func::FuncOp funcOp, const ScopedOperations &scopedOperations) {
  SmallVector<linalg::GenericOp> candidates;
  for (Operation *operation : scopedOperations) {
    auto genericOp = dyn_cast<linalg::GenericOp>(operation);
    if (genericOp && hasConstantIndexingMapResult(genericOp))
      candidates.push_back(genericOp);
  }

  IRRewriter rewriter(funcOp.getContext());
  linalg::ControlDropUnitDims options;
  for (linalg::GenericOp genericOp : candidates) {
    rewriter.setInsertionPoint(genericOp);
    FailureOr<linalg::DropUnitDimsResult> result =
        linalg::dropUnitDims(rewriter, genericOp, options);
    if (succeeded(result))
      rewriter.replaceOp(genericOp, result->replacements);
  }
}

/// Return true when every use of a value belongs to the same operation.
///
/// An operation that consumes a value both as an input and as its destination
/// creates two uses but a single consumer. The shared fusion patterns emit a
/// type-restoring reshape exactly when a consumer other than the moved reshape
/// remains, so counting consumers rather than uses is the condition that
/// decides whether such a reshape will appear.
static bool hasSingleConsumerOp(Value value) {
  Operation *consumer = nullptr;
  for (OpOperand &use : value.getUses()) {
    if (!consumer) {
      consumer = use.getOwner();
      continue;
    }
    if (use.getOwner() != consumer)
      return false;
  }
  return consumer != nullptr;
}

/// Admit the forward reshape-fusion directions through the shared callback.
/// Both the by-expansion and by-collapsing families are registered, but their
/// owners have different legality conditions. An expand_shape owner has no
/// additional condition: moving it toward its producer leaves the restoring
/// collapse_shape downstream in the prescribed order. A collapse_shape owner
/// is admitted only when its producer is not a transpose and its source has a
/// single consumer operation; these conditions avoid moving a collapse through
/// the permutation boundary or creating a restoring expand_shape that would
/// form an inversion for the expansion family. A failed match is an ordinary
/// pattern failure and emits no pass diagnostic.
static linalg::ControlFusionFn
makeScopeControlFn(const llvm::DenseSet<Operation *> &scopedSet) {
  return [&scopedSet](OpOperand *operand) {
    if (!operand)
      return false;
    Operation *owner = operand->getOwner();
    Operation *producer = operand->get().getDefiningOp();
    if (!owner || !producer || !scopedSet.contains(owner) ||
        !scopedSet.contains(producer))
      return false;
    // Moving an expand_shape toward its producer always leaves the restoring
    // collapse_shape downstream, which is the order the canonical form
    // prescribes. A collapse_shape may only move when the move keeps that
    // order intact: never past a transpose, because the canonical order places
    // the permutation first and the custom transpose rewrite moves a collapse
    // in the opposite direction; and only when the moved reshape is the sole
    // consumer, because otherwise a restoring expand_shape appears downstream
    // and forms an inversion the expansion family would act on again.
    if (isa<tensor::ExpandShapeOp>(owner))
      return true;
    if (isa<tensor::CollapseShapeOp>(owner))
      return !isa<linalg::TransposeOp>(producer) &&
             hasSingleConsumerOp(operand->get());
    return false;
  };
}

/// Give every scoped use after the first one its own empty allocation.
///
/// The use list is snapshotted as OpOperand pointers before any operand is
/// redirected. This is deliberately a pre-processing step rather than a
/// rewrite pattern: two uses in the same operation must be split as well, and
/// users outside the coarse scope must keep the original value. Cloning the
/// operation also copies dynamic-size operands and discardable attributes.
static void unshareScopedEmptyOps(func::FuncOp funcOp,
                                  ScopedOperations &scopedOperations,
                                  llvm::DenseSet<Operation *> &scopedSet) {
  SmallVector<tensor::EmptyOp> scopedEmpties;
  for (Operation *operation : scopedOperations)
    if (auto empty = dyn_cast<tensor::EmptyOp>(operation))
      scopedEmpties.push_back(empty);

  llvm::DenseMap<Operation *, unsigned> irOrder;
  unsigned ordinal = 0;
  funcOp.walk(
      [&](Operation *operation) { irOrder.try_emplace(operation, ordinal++); });

  for (tensor::EmptyOp empty : scopedEmpties) {
    SmallVector<OpOperand *> scopedUses;
    for (OpOperand &use : empty.getResult().getUses())
      if (scopedSet.contains(use.getOwner()))
        scopedUses.push_back(&use);

    if (scopedUses.size() <= 1)
      continue;

    // MLIR use-list order is not a stable IR-order contract. Sort the
    // snapshot explicitly so the retained use is independent of allocation
    // or hash-table order; operand number also distinguishes two uses by the
    // same operation, as required for an op that consumes the empty twice.
    llvm::stable_sort(scopedUses, [&](OpOperand *lhs, OpOperand *rhs) {
      unsigned lhsOrder = irOrder.lookup(lhs->getOwner());
      unsigned rhsOrder = irOrder.lookup(rhs->getOwner());
      if (lhsOrder != rhsOrder)
        return lhsOrder < rhsOrder;
      return lhs->getOperandNumber() < rhs->getOperandNumber();
    });

    // Keep the first use on the original operation. Insert each clone after
    // the preceding clone so the resulting order is deterministic.
    Operation *insertionPoint = empty.getOperation();
    for (unsigned index = 1; index < scopedUses.size(); ++index) {
      OpBuilder builder(empty.getOperation()->getContext());
      builder.setInsertionPointAfter(insertionPoint);
      Operation *clone = builder.clone(*empty.getOperation());
      scopedUses[index]->set(clone->getResult(0));
      scopedSet.insert(clone);
      scopedOperations.insert(clone);
      insertionPoint = clone;
    }
  }

  // The explicit greedy root list remains in stable function IR order after
  // the newly inserted clones have been added to the scope.
  ScopedOperations ordered;
  funcOp.walk([&](Operation *operation) {
    if (scopedSet.contains(operation))
      ordered.insert(operation);
  });
  scopedOperations.clear();
  scopedOperations.insert(ordered.begin(), ordered.end());
}

/// Remove erased operations from the explicit scope set. The greedy driver
/// may recycle an erased operation's address for a newly created operation;
/// retaining the stale pointer would make the scope callback depend on the
/// allocator state across processes.
class ScopedSetListener : public RewriterBase::Listener {
public:
  explicit ScopedSetListener(llvm::DenseSet<Operation *> &scopedSet)
      : scopedSet(scopedSet) {}

  void notifyOperationInserted(Operation *operation,
                               OpBuilder::InsertPoint) override {
    // ExistingAndNewOps allows a rewrite-created operation to become a new
    // greedy root. It is part of the same scoped rewrite, so keep it eligible
    // for subsequent forwarding in this fixed-point iteration; otherwise the
    // next invocation would discover a larger scope and make the pass
    // non-idempotent.
    scopedSet.insert(operation);
  }

  void notifyOperationReplaced(Operation *operation,
                               ValueRange) override {
    scopedSet.erase(operation);
  }

  void notifyOperationErased(Operation *operation) override {
    scopedSet.erase(operation);
  }

private:
  llvm::DenseSet<Operation *> &scopedSet;
};

/// Rewrite transpose(collapse(x)) as collapse(transpose(x)) when the
/// collapsed groups remain contiguous and in their original order under the
/// lifted transpose permutation.
class TransposeAfterCollapsePattern
    : public OpRewritePattern<linalg::TransposeOp> {
public:
  using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp transpose,
                                PatternRewriter &rewriter) const override {
    auto collapse = transpose.getInput().getDefiningOp<tensor::CollapseShapeOp>();
    if (!collapse)
      return failure();

    auto sourceType = dyn_cast<RankedTensorType>(collapse.getSrc().getType());
    if (!sourceType)
      return failure();

    SmallVector<ReassociationIndices> oldReassociation =
        collapse.getReassociationIndices();
    ArrayRef<int64_t> oldPermutation = transpose.getPermutation();
    if (oldReassociation.size() != oldPermutation.size())
      return failure();

    SmallVector<bool> seenGroups(oldReassociation.size(), false);
    SmallVector<int64_t> liftedPermutation;
    SmallVector<ReassociationIndices> newReassociation;
    liftedPermutation.reserve(sourceType.getRank());
    newReassociation.reserve(oldReassociation.size());

    // `oldPermutation` is output-dimension -> collapsed-input-dimension.
    // Replace each selected collapsed dimension with its complete original
    // group. The group order is retained, while the groups themselves appear
    // in the transposed order; this is the transpose that can legally precede
    // the collapse. The new reassociation records those consecutive blocks.
    for (int64_t groupIndex : oldPermutation) {
      if (groupIndex < 0 ||
          groupIndex >= static_cast<int64_t>(oldReassociation.size()) ||
          seenGroups[groupIndex] || oldReassociation[groupIndex].empty())
        return failure();
      seenGroups[groupIndex] = true;

      newReassociation.emplace_back();
      for (int64_t sourceDim : oldReassociation[groupIndex]) {
        if (sourceDim < 0 || sourceDim >= sourceType.getRank())
          return failure();
        liftedPermutation.push_back(sourceDim);
        newReassociation.back().push_back(liftedPermutation.size() - 1);
      }
    }

    if (liftedPermutation.size() !=
        static_cast<size_t>(sourceType.getRank()))
      return failure();

    // Check the legality condition explicitly rather than relying on the
    // verifier: every old group must be adjacent in the lifted permutation and
    // its members must retain their old relative order. For example, groups
    // [[0, 1], [2]] with a lifted order [0, 2, 1] split the first group and
    // therefore match-fail silently instead of sorting the source dimensions.
    for (const ReassociationIndices &group : oldReassociation) {
      SmallVector<unsigned> positions;
      positions.reserve(group.size());
      for (int64_t sourceDim : group) {
        auto position = llvm::find(liftedPermutation, sourceDim);
        if (position == liftedPermutation.end())
          return failure();
        positions.push_back(position - liftedPermutation.begin());
      }
      for (unsigned index = 1; index < positions.size(); ++index)
        if (positions[index] != positions[0] + index ||
            liftedPermutation[positions[index]] != group[index])
          return failure();
    }

    SmallVector<int64_t> transposedShape;
    SmallVector<Value> dynamicSizes;
    transposedShape.reserve(liftedPermutation.size());
    for (int64_t sourceDim : liftedPermutation) {
      int64_t size = sourceType.getDimSize(sourceDim);
      transposedShape.push_back(size);
      if (ShapedType::isDynamic(size))
        dynamicSizes.push_back(rewriter.create<tensor::DimOp>(
            transpose.getLoc(), collapse.getSrc(), sourceDim));
    }

    auto transposedType = RankedTensorType::get(
        transposedShape, sourceType.getElementType(), sourceType.getEncoding());
    auto expectedResultType = tensor::CollapseShapeOp::inferCollapsedType(
        transposedType, newReassociation);
    if (expectedResultType != transpose->getResult(0).getType())
      return failure();

    Value transposeInit = rewriter.create<tensor::EmptyOp>(
        transpose.getLoc(), transposedType, dynamicSizes);
    auto movedTranspose = rewriter.create<linalg::TransposeOp>(
        transpose.getLoc(), collapse.getSrc(), transposeInit, liftedPermutation);
    auto movedCollapse = rewriter.create<tensor::CollapseShapeOp>(
        transpose.getLoc(), transpose->getResult(0).getType(),
        movedTranspose.getResult().front(), newReassociation);
    rewriter.replaceOp(transpose, movedCollapse.getResult());
    return success();
  }
};

/// Replace a chain of adjacent reassociative reshapes with its minimal form.
///
/// expand_shape and collapse_shape both preserve the linear element order and
/// only change how dimensions are grouped, so a chain built from them is fully
/// determined by its two end types; the intermediate types carry no
/// information. The minimal realization is therefore a closed form rather than
/// a search: no operation when the ends coincide, one when
/// getReassociationIndicesForReshape succeeds, and otherwise two through the
/// fully collapsed one-dimensional shape, which is always reachable in both
/// directions. Because two is the global lower bound in that last case, merging
/// a sub-chain can never beat handling the whole chain, and no merge order has
/// to be considered.
///
/// The two-operation form is a collapse followed by an expand, which is the
/// reverse of the order the canonical form prescribes. It is inert here: both
/// reshape-forwarding families require the reshape's producer to be a linalg
/// operation, and in this pair the producer is a reshape, so neither family can
/// match it and the inversion cannot feed the divergence those families exhibit
/// when they coexist.
///
/// Termination: a rewrite is only applied when it strictly lowers the number of
/// reshapes in the chain, so the reshape count of the scope decreases and the
/// pattern cannot rematch its own result. No linalg iteration domain changes,
/// so the pattern does not participate in the inversion measure that governs
/// the forwarding families.
template <typename ReshapeOpTy>
class MinimizeReshapeChainPattern : public OpRewritePattern<ReshapeOpTy> {
public:
  using OpRewritePattern<ReshapeOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOpTy op,
                                PatternRewriter &rewriter) const override {
    // Walk upstream across adjacent reassociative reshapes. A member is only
    // absorbed when the next member is its sole user, so the rewrite always
    // removes operations rather than adding one next to a value that stays
    // live. linalg.transpose and tensor.reshape end the chain: the former is a
    // permutation this pattern does not reason about, the latter carries its
    // shape as an operand and has no static reassociation.
    Value head = op.getSrc();
    unsigned chainLength = 1;
    while (Operation *producer = head.getDefiningOp()) {
      if (!isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(producer))
        break;
      if (!producer->getResult(0).hasOneUse())
        break;
      head = producer->getOperand(0);
      ++chainLength;
    }

    auto headType = dyn_cast<RankedTensorType>(head.getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!headType || !resultType)
      return rewriter.notifyMatchFailure(op, "chain ends are not ranked");
    if (!headType.hasStaticShape() || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "chain ends are not static");

    // Zero operations: the chain is an identity on its ends.
    if (headType == resultType) {
      if (chainLength < 1)
        return failure();
      rewriter.replaceOp(op, head);
      return success();
    }

    // One operation, when the ends are rank-compatible.
    if (std::optional<SmallVector<ReassociationIndices>> reassociation =
            getReassociationIndicesForReshape(headType, resultType)) {
      if (chainLength <= 1)
        return rewriter.notifyMatchFailure(op, "chain is already minimal");
      if (headType.getRank() < resultType.getRank())
        rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
            op, resultType, head, *reassociation);
      else
        rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
            op, resultType, head, *reassociation);
      return success();
    }

    // Two operations through the flat shape. Reachable for any pair of static
    // shapes with equal element counts, since a single group spanning every
    // dimension is always a valid reassociation.
    if (chainLength <= 2)
      return rewriter.notifyMatchFailure(op, "chain is already minimal");

    auto flatType = RankedTensorType::get({headType.getNumElements()},
                                          headType.getElementType(),
                                          headType.getEncoding());
    std::optional<SmallVector<ReassociationIndices>> toFlat =
        getReassociationIndicesForReshape(headType, flatType);
    std::optional<SmallVector<ReassociationIndices>> fromFlat =
        getReassociationIndicesForReshape(flatType, resultType);
    if (!toFlat || !fromFlat)
      return rewriter.notifyMatchFailure(op, "flat shape is not reachable");

    Value flat = rewriter.create<tensor::CollapseShapeOp>(
        op.getLoc(), flatType, head, *toFlat);
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(op, resultType, flat,
                                                      *fromFlat);
    return success();
  }
};

struct QGEMMDimTransformPass
    : public PassWrapper<QGEMMDimTransformPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QGEMMDimTransformPass)

  StringRef getArgument() const override {
    return "qgemm-simplify-dim-transforms";
  }

  StringRef getDescription() const override {
    return "Simplify dimension-transform chains in canonical-form QGEMM "
           "scopes";
  }

  QGEMMDimTransformPass() = default;
  QGEMMDimTransformPass(const QGEMMDimTransformPass &other)
      : PassWrapper(other) {}

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    ScopedOperations scopedOperations = collectCoarseScope(funcOp);
    if (scopedOperations.empty())
      return;

    // Normalize only the originally selected broadcast generics with the
    // upstream dropUnitDims utility. Recompute the scope after replacements so
    // generated reshapes and changed tensor boundaries are eligible for the
    // one scoped greedy invocation below.
    normalizeScopedBroadcasts(funcOp, scopedOperations);
    scopedOperations = collectCoarseScope(funcOp);
    if (scopedOperations.empty())
      return;

    MLIRContext *context = funcOp.getContext();
    llvm::DenseSet<Operation *> scopedSet;
    scopedSet.insert(scopedOperations.begin(), scopedOperations.end());
    unshareScopedEmptyOps(funcOp, scopedOperations, scopedSet);

    // Dispatch the complete stable IR-ordered scope once. ExistingAndNewOps
    // and the unlimited greedy iteration provide the fixed point. The scope
    // listener only removes erased operations so recycled addresses cannot
    // turn a newly created operation into a stale scope member.
    RewritePatternSet patterns(context);
    // Reassociative reshape canonicalization composes adjacent same-kind
    // reshapes and cancels an inverse pair. Neither rewrite changes the
    // iteration-domain rank of a linalg operation, so these patterns do not
    // participate in the inversion mechanism that governs which reshape
    // forwarding families may coexist here. Without them the scoped rewrites
    // leave behind reshape round trips that are pure churn, such as a collapse
    // to a flat shape followed by an expand that reintroduces the original
    // dimensions.
    tensor::ExpandShapeOp::getCanonicalizationPatterns(patterns, context);
    tensor::CollapseShapeOp::getCanonicalizationPatterns(patterns, context);
    tensor::populateReassociativeReshapeFoldingPatterns(patterns);
    tensor::populateBubbleUpExpandShapePatterns(patterns);
    tensor::populateFoldTensorEmptyPatterns(patterns,
                                            /*foldSingleUseOnly=*/true);

    linalg::ControlFusionFn controlFn = makeScopeControlFn(scopedSet);
    linalg::populateFoldReshapeOpsByExpansionPatterns(patterns, controlFn);
    linalg::populateFoldReshapeOpsByCollapsingPatterns(patterns, controlFn);

    patterns.add<TransposeAfterCollapsePattern>(context, PatternBenefit(100));
    patterns.add<MinimizeReshapeChainPattern<tensor::ExpandShapeOp>,
                 MinimizeReshapeChainPattern<tensor::CollapseShapeOp>>(
        context, PatternBenefit(100));
    // Termination measure: every admitted rewrite strictly decreases the
    // number of collapse-before-expand inversions in the scope. A by-expansion
    // rewrite moves an expand_shape toward its producer and leaves any
    // type-restoring collapse_shape downstream, which is already the prescribed
    // order. A by-collapsing rewrite is admitted only when it needs no restoring
    // reshape and does not cross a transpose, so it cannot create an inversion
    // that the by-expansion family can match again. Reshape-chain minimization
    // monotonically decreases the reshape count, changes no linalg iteration
    // domain, and is therefore independent of the collapse-before-expand
    // inversion measure.
    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps)
        .setMaxIterations(GreedyRewriteConfig::kNoLimit)
        .enableFolding();
    ScopedSetListener scopeListener(scopedSet);
    config.setListener(&scopeListener);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    // The op-list greedy driver drains its worklist exactly once. Its region
    // counterpart re-seeds every operation and repeats until an iteration makes
    // no change, and that outer iteration is what this loop restores.
    //
    // The gap it covers is specific: erasing an operation re-enqueues the
    // defining operations of that operation's operands, and never the other
    // users of those operands. A reshape whose eligibility depends on how many
    // consumers its source value has therefore stays unvisited after a sibling
    // consumer dies, because neither the reshape nor its operand changed.
    //
    // The region driver is not usable here even though it owns that iteration:
    // its strict-mode set is the whole region, so it cannot express a scope
    // that is a subset of a function, and the reassociative-reshape, bubble-up
    // and empty-folding pattern sets in this pool take no control callback and
    // would then rewrite operations outside the recognized operators.
    //
    // The bound is an assertion, not a tuning knob. Every admitted rewrite
    // strictly decreases the number of collapse-before-expand inversions in a
    // bounded scope, so a well-behaved input reaches its fixed point in a
    // couple of iterations. Exhausting the bound means an unforeseen cycle
    // exists, and failing loudly is preferable to emitting IR that is not
    // idempotent.
    constexpr unsigned kMaxScopedIterations = 64;
    unsigned iteration = 0;
    for (bool changed = true; changed;) {
      if (++iteration > kMaxScopedIterations) {
        funcOp.emitError("QGEMM dimension-transform simplification did not "
                         "reach a fixed point");
        signalPassFailure();
        return;
      }

      SmallVector<Operation *> roots;
      funcOp.walk([&](Operation *operation) {
        if (scopedSet.contains(operation))
          roots.push_back(operation);
      });
      if (roots.empty())
        break;

      changed = false;
      if (failed(applyOpPatternsGreedily(roots, frozenPatterns, config,
                                         &changed))) {
        signalPassFailure();
        return;
      }
    }

    // Run MLIR's standard CSE once after the scoped rewrites. It also removes
    // dead old reshapes and coalesces the temporary empties created while the
    // scoped rewrites were made independent.
    //
    // This CSE is deliberately not scope-restricted. It can therefore merge or
    // delete operations of the same function that lie outside the coarse scope,
    // which is accepted: the alternative is a scope-restricted reimplementation
    // of CSE, and what it can touch is limited to what CSE does anywhere.
    // Functions other than the one being simplified are untouched, which is the
    // guarantee the scope test asserts.
    OpPassManager csePipeline(func::FuncOp::getOperationName());
    csePipeline.addPass(createCSEPass());
    if (failed(runPipeline(csePipeline, funcOp.getOperation()))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

namespace mlir {
namespace buddy {

void registerQGEMMDimTransformPass() {
  PassRegistration<QGEMMDimTransformPass>();
}

} // namespace buddy
} // namespace mlir
