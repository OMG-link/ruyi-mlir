//===- AxisTracker.cpp - Axis tracking for QGEMM coupling -------*- C++ -*-===//
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
// The producer-ward mapping functions below (getRank through
// getProducerMapping) are textual copies of the recognizer's functions in
// QGEMMRecognize/Recognition.cpp. Only the enclosing namespace and the include
// lines differ. Extracting them into one shared implementation is a follow-up.
//
//===----------------------------------------------------------------------===//

#include "AxisTracker.h"

#include "../QGEMMRecognize/Recognition.h"

#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

#define DEBUG_TYPE "qgemm-decompress-coupling"

using namespace mlir;

namespace mlir {
namespace buddy {
namespace qgemm_coupling {
namespace {

// ---- Begin copy of QGEMMRecognize/Recognition.cpp ----

static std::optional<unsigned> getRank(Value value) {
  auto shapedType = dyn_cast<ShapedType>(value.getType());
  if (!shapedType || !shapedType.hasRank())
    return std::nullopt;
  return shapedType.getRank();
}

static DimMapping makeIdentityMapping(unsigned rank) {
  DimMapping mapping;
  mapping.reserve(rank);
  for (unsigned dimension = 0; dimension < rank; ++dimension)
    mapping.push_back(dimension);
  return mapping;
}

static DimMapping makeNullMapping(unsigned rank) {
  return DimMapping(rank, std::nullopt);
}

static DimMapping resizeMapping(const DimMapping &mapping, unsigned rank) {
  DimMapping resized = makeNullMapping(rank);
  for (unsigned index = 0, end = std::min<unsigned>(mapping.size(), rank);
       index < end; ++index)
    resized[index] = mapping[index];
  return resized;
}

/// Map the result dimensions of an affine operand map through the consumer's
/// iteration-space mapping.  Only dimension expressions are meaningful to the
/// recognizer; symbols and compound expressions are intentionally unknown.
static DimMapping mapThroughAffineMap(AffineMap map,
                                      const DimMapping &iterationMapping) {
  DimMapping mapping;
  mapping.reserve(map.getNumResults());
  for (AffineExpr expression : map.getResults()) {
    if (auto dimension = dyn_cast<AffineDimExpr>(expression)) {
      unsigned position = dimension.getPosition();
      if (position < iterationMapping.size()) {
        mapping.push_back(iterationMapping[position]);
        continue;
      }
    }
    mapping.push_back(std::nullopt);
  }
  return mapping;
}

/// Return the iteration-space mapping of an operation from the mapping of its
/// result.  A reduction iterator is explicitly mapped to no final output
/// dimension.  This also handles a linalg.generic with reduction iterators,
/// which is the form produced when simplification rebuilds a non-root reduce.
static DimMapping getIterationMapping(Operation *operation,
                                      const DimMapping &resultMapping) {
  if (auto reduce = dyn_cast<linalg::ReduceOp>(operation)) {
    if (reduce.getInputs().empty())
      return resultMapping;

    auto inputRank = getRank(reduce.getInputs().front());
    if (!inputRank)
      return {};

    DenseSet<int64_t> reductionDimensions;
    for (int64_t dimension : reduce.getDimensions())
      reductionDimensions.insert(dimension);

    DimMapping mapping = makeNullMapping(*inputRank);
    unsigned resultDimension = 0;
    for (unsigned inputDimension = 0; inputDimension < *inputRank;
         ++inputDimension) {
      if (reductionDimensions.contains(inputDimension))
        continue;
      if (resultDimension < resultMapping.size())
        mapping[inputDimension] = resultMapping[resultDimension];
      ++resultDimension;
    }
    return mapping;
  }

  if (auto generic = dyn_cast<linalg::GenericOp>(operation)) {
    SmallVector<utils::IteratorType> iteratorTypes =
        generic.getIteratorTypesArray();
    DimMapping mapping = makeNullMapping(iteratorTypes.size());
    if (generic.getNumDpsInits() == 0)
      return mapping;

    unsigned outputOperandNumber = generic.getNumDpsInputs();
    if (outputOperandNumber >= operation->getNumOperands())
      return mapping;

    OpOperand &output = operation->getOpOperand(outputOperandNumber);
    AffineMap outputMap = generic.getMatchingIndexingMap(&output);
    for (auto [resultDimension, expression] :
         llvm::enumerate(outputMap.getResults())) {
      auto dimension = dyn_cast<AffineDimExpr>(expression);
      if (!dimension || resultDimension >= resultMapping.size())
        continue;
      unsigned position = dimension.getPosition();
      if (position < mapping.size())
        mapping[position] = resultMapping[resultDimension];
    }

    // Do not infer a reduction from an operation name.  The iterator type is
    // the authoritative source, including for generic reductions rebuilt by
    // the dimension-transform pass.
    if (isQGEMMReduction(operation)) {
      for (auto [dimension, iteratorType] : llvm::enumerate(iteratorTypes)) {
        if (linalg::isReductionIterator(iteratorType))
          mapping[dimension] = std::nullopt;
      }
    }
    return mapping;
  }

  auto rank = operation->getNumResults() ? getRank(operation->getResult(0))
                                         : std::optional<unsigned>();
  if (!rank)
    return resultMapping;
  return resizeMapping(resultMapping, *rank);
}

/// Compute the mapping of the producer result for one consumer operand.
static DimMapping getProducerMapping(Operation *consumer, OpOperand &operand,
                                     const DimMapping &iterationMapping) {
  if (auto generic = dyn_cast<linalg::GenericOp>(consumer))
    return mapThroughAffineMap(generic.getMatchingIndexingMap(&operand),
                               iterationMapping);

  if (auto reduce = dyn_cast<linalg::ReduceOp>(consumer)) {
    unsigned inputCount = reduce.getInputs().size();
    if (operand.getOperandNumber() < inputCount)
      return iterationMapping;

    DenseSet<int64_t> reductionDimensions;
    for (int64_t dimension : reduce.getDimensions())
      reductionDimensions.insert(dimension);

    DimMapping mapping;
    for (unsigned dimension = 0; dimension < iterationMapping.size();
         ++dimension) {
      if (!reductionDimensions.contains(dimension))
        mapping.push_back(iterationMapping[dimension]);
    }
    return mapping;
  }

  if (auto transpose = dyn_cast<linalg::TransposeOp>(consumer)) {
    if (operand.getOperandNumber() ==
        transpose.getInputMutable().getOperandNumber()) {
      ArrayRef<int64_t> permutation = transpose.getPermutation();
      DimMapping mapping = makeNullMapping(permutation.size());
      for (unsigned outputDimension = 0; outputDimension < permutation.size();
           ++outputDimension) {
        int64_t inputDimension = permutation[outputDimension];
        if (inputDimension >= 0 &&
            static_cast<unsigned>(inputDimension) < mapping.size() &&
            outputDimension < iterationMapping.size())
          mapping[inputDimension] = iterationMapping[outputDimension];
      }
      return mapping;
    }
    return iterationMapping;
  }

  if (auto expand = dyn_cast<tensor::ExpandShapeOp>(consumer)) {
    SmallVector<ReassociationIndices> reassociation =
        expand.getReassociationIndices();
    DimMapping mapping;
    mapping.reserve(reassociation.size());
    for (ReassociationIndices group : reassociation) {
      std::optional<unsigned> knownDimension;
      for (int64_t outputDimension : group) {
        if (outputDimension < 0 ||
            static_cast<unsigned>(outputDimension) >= iterationMapping.size())
          continue;
        if (iterationMapping[outputDimension]) {
          knownDimension = iterationMapping[outputDimension];
          break;
        }
      }
      mapping.push_back(knownDimension);
    }
    return mapping;
  }

  if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(consumer)) {
    SmallVector<ReassociationIndices> reassociation =
        collapse.getReassociationIndices();
    DimMapping mapping;
    for (unsigned outputDimension = 0; outputDimension < reassociation.size();
         ++outputDimension) {
      for (int64_t ignored : reassociation[outputDimension]) {
        (void)ignored;
        if (outputDimension < iterationMapping.size())
          mapping.push_back(iterationMapping[outputDimension]);
        else
          mapping.push_back(std::nullopt);
      }
    }
    return mapping;
  }

  auto rank = getRank(operand.get());
  if (!rank)
    return {};
  return resizeMapping(iterationMapping, *rank);
}

// ---- End copy of QGEMMRecognize/Recognition.cpp ----

/// Producer-ward mapping used by the tracker: the recognizer copy above has
/// no rule for linalg.broadcast, so a single-dim named broadcast is handled
/// here before delegating (its input drops the broadcast dims, its init has
/// the result's dims).
static DimMapping
getTrackerProducerMapping(Operation *consumer, OpOperand &operand,
                          const DimMapping &iterationMapping) {
  if (auto broadcast = dyn_cast<linalg::BroadcastOp>(consumer)) {
    if (operand.getOperandNumber() !=
        broadcast.getInputMutable().getOperandNumber())
      return iterationMapping;
    ArrayRef<int64_t> dimensions = broadcast.getDimensions();
    DimMapping mapping;
    for (unsigned dimension = 0; dimension < iterationMapping.size();
         ++dimension)
      if (!llvm::is_contained(dimensions, static_cast<int64_t>(dimension)))
        mapping.push_back(iterationMapping[dimension]);
    return mapping;
  }
  return getProducerMapping(consumer, operand, iterationMapping);
}

/// The shape operand of a tensor.reshape is a 1-D extent list, not data: its
/// dims are not dims of any value of Omega(r), so no axis is tracked on it.
/// The recognizer's fallback mapping would otherwise give it the result's
/// leading axis, which a rebuild that inserts that axis turns into a wrong
/// position on the second run.
static bool isShapeOperand(Operation *op, OpOperand &operand) {
  auto reshape = dyn_cast<tensor::ReshapeOp>(op);
  return reshape && &operand == &reshape.getShapeMutable();
}

} // namespace

AxisTracker::AxisTracker(Operation *finalReduce,
                         const SetVector<Operation *> &omega, unsigned mDim,
                         unsigned nDim)
    : finalReduce(finalReduce), omega(omega), mDim(mDim), nDim(nDim) {}

void AxisTracker::run() {
  assert(finalReduce && finalReduce->getNumResults() == 1 &&
         "qgemm.root op must have exactly one result");
  auto rank = getRank(finalReduce->getResult(0));
  assert(rank && "qgemm.root result must be a ranked shaped value");
  DimMapping resultMapping = makeIdentityMapping(*rank);
  recordValue(finalReduce->getResult(0), resultMapping);
  visitOperation(finalReduce, getIterationMapping(finalReduce, resultMapping));
  visitCopySources();
  propagateToUsers();
  markGrouped();
#ifndef NDEBUG
  verifyExtents();
#endif
}

/// Mirrors RecognitionAnalyzer::visitOperation without the broadcast-edge
/// classification: each op of Omega(r) is visited once, and every traceable
/// operand value records the producer-ward mapping computed on the way.
void AxisTracker::visitOperation(Operation *operation,
                                 const DimMapping &iterationMapping) {
  if (!omega.contains(operation))
    return;
  if (!visited.insert(operation).second)
    return;

  for (OpOperand &operand : operation->getOpOperands()) {
    if (!isa<TensorType, BaseMemRefType>(operand.get().getType()) ||
        isShapeOperand(operation, operand))
      continue;

    DimMapping producerResultMapping = disambiguate(
        operand.get(),
        getTrackerProducerMapping(operation, operand, iterationMapping));
    recordValue(operand.get(), producerResultMapping);

    Operation *producer = operand.get().getDefiningOp();
    if (!producer || !omega.contains(producer))
      continue;

    DimMapping producerIterationMapping =
        getIterationMapping(producer, producerResultMapping);
    visitOperation(producer, producerIterationMapping);
  }
}

/// memref.copy rule. The producer-ward walk follows operands only, and a
/// memref.copy is no operand of anything: it reaches its target buffer by a
/// write. Its source has the target's shape and dimension order, so the
/// target's mapping carries over to the source unchanged, and the walk
/// continues from the source's producer.
void AxisTracker::visitCopySources() {
  for (Operation *op : omega) {
    auto copy = dyn_cast<memref::CopyOp>(op);
    if (!copy)
      continue;
    const DimMapping *targetMapping = getMapping(copy.getTarget());
    std::optional<unsigned> rank = getRank(copy.getSource());
    if (!targetMapping || !rank || *rank != targetMapping->size())
      continue;
    DimMapping mapping = *targetMapping;
    recordValue(copy.getSource(), mapping);
    Operation *producer = copy.getSource().getDefiningOp();
    if (producer && omega.contains(producer))
      visitOperation(producer, getIterationMapping(producer, mapping));
  }
}
std::optional<int64_t> AxisTracker::getRootExtent(unsigned rootDim) const {
  auto type = cast<ShapedType>(finalReduce->getResult(0).getType());
  if (rootDim >= type.getRank() || type.isDynamicDim(rootDim))
    return std::nullopt;
  return type.getDimSize(rootDim);
}

/// The recognizer's producer-ward collapse_shape rule copies a result dim's
/// root entry onto every source dim of its group, so a value can come back
/// with the same root dim at several positions (e.g. through the iq2_xxs
/// weight chain's flatten/unflatten). Only one of them carries the axis: keep
/// the positions whose static extent equals the root dim's extent; if exactly
/// one matches, that is the axis. Otherwise keep only the first occurrence,
/// which is the grouped case where the axis is merged with others.
DimMapping AxisTracker::disambiguate(Value value,
                                     const DimMapping &mapping) const {
  auto type = dyn_cast<ShapedType>(value.getType());
  if (!type || !type.hasRank())
    return mapping;
  DimMapping result = mapping;
  SmallVector<unsigned> seen;
  for (std::optional<unsigned> entry : mapping) {
    if (!entry || llvm::is_contained(seen, *entry))
      continue;
    seen.push_back(*entry);
    SmallVector<unsigned> positions;
    for (auto [position, other] : llvm::enumerate(mapping))
      if (other == entry)
        positions.push_back(position);
    if (positions.size() < 2)
      continue;
    std::optional<int64_t> extent = getRootExtent(*entry);
    SmallVector<unsigned> matching;
    if (extent)
      for (unsigned position : positions)
        if (position < type.getRank() && !type.isDynamicDim(position) &&
            type.getDimSize(position) == *extent)
          matching.push_back(position);
    unsigned keep = matching.size() == 1 ? matching.front() : positions.front();
    for (unsigned position : positions)
      if (position != keep)
        result[position] = std::nullopt;
  }
  return result;
}

/// A tracked axis position must have the root dim's static extent, unless the
/// value is produced or consumed by a reshape that groups the axis, where the
/// axis is merged with other dims.
void AxisTracker::verifyExtents() const {
  for (Value value : valueOrder) {
    auto type = dyn_cast<ShapedType>(value.getType());
    if (!type || !type.hasRank())
      continue;
    for (Axis axis : {Axis::M, Axis::N}) {
      std::optional<unsigned> position = getAxisPosition(value, axis);
      std::optional<int64_t> extent = getRootExtent(getRootDim(axis));
      if (!position || !extent || type.isDynamicDim(*position))
        continue;
      bool groupedAdjacent = false;
      if (Operation *def = value.getDefiningOp())
        groupedAdjacent |= isGrouped(def, axis);
      for (Operation *user : value.getUsers())
        groupedAdjacent |= isGrouped(user, axis);
      if (groupedAdjacent)
        continue;
      if (type.getDimSize(*position) != *extent) {
        LLVM_DEBUG({
          llvm::dbgs() << "[qgemm-decompress-coupling] extent mismatch on ";
          value.printAsOperand(llvm::dbgs(), OpPrintingFlags());
          llvm::dbgs() << " axis " << static_cast<unsigned>(axis) << "\n";
        });
        assert(false && "tracked axis position disagrees with the root extent");
      }
    }
  }
}

/// The first path that reaches a value fixes its mapping; later paths only
/// fill entries the first one left unknown. A root dim already present in the
/// mapping is never assigned to a second position.
bool AxisTracker::recordValue(Value value, const DimMapping &rawMapping) {
  DimMapping mapping = disambiguate(value, rawMapping);
  auto [it, inserted] = valueMappings.try_emplace(value, mapping);
  if (inserted) {
    valueOrder.push_back(value);
    return true;
  }
  DimMapping &existing = it->second;
  bool changed = false;
  for (unsigned index = 0,
                end = std::min<unsigned>(existing.size(), mapping.size());
       index < end; ++index) {
    if (existing[index] || !mapping[index])
      continue;
    if (llvm::is_contained(existing, mapping[index]))
      continue;
    existing[index] = mapping[index];
    changed = true;
  }
  return changed;
}

// ---- Consumer-ward propagation (not part of the recognizer copy) ----

DimMapping
getResultMappingFromOperand(Operation *consumer, OpOperand &operand,
                            const DimMapping &operandMapping,
                            SmallVectorImpl<unsigned> *groupedRootDims) {
  if (consumer->getNumResults() == 0)
    return {};
  auto resultRank = getRank(consumer->getResult(0));
  if (!resultRank)
    return {};
  DimMapping result = makeNullMapping(*resultRank);

  if (auto generic = dyn_cast<linalg::GenericOp>(consumer)) {
    // Invert the operand map over its dim-expression results, then read the
    // iteration-space mapping through the output map.
    DimMapping iteration = makeNullMapping(generic.getNumLoops());
    AffineMap operandMap = generic.getMatchingIndexingMap(&operand);
    for (auto [index, expression] : llvm::enumerate(operandMap.getResults())) {
      auto dimension = dyn_cast<AffineDimExpr>(expression);
      if (!dimension || index >= operandMapping.size())
        continue;
      unsigned position = dimension.getPosition();
      if (position < iteration.size() && !iteration[position])
        iteration[position] = operandMapping[index];
    }
    if (generic.getNumDpsInits() == 0)
      return result;
    AffineMap outputMap =
        generic.getMatchingIndexingMap(generic.getDpsInitOperand(0));
    for (auto [index, expression] : llvm::enumerate(outputMap.getResults())) {
      auto dimension = dyn_cast<AffineDimExpr>(expression);
      if (!dimension || index >= result.size())
        continue;
      unsigned position = dimension.getPosition();
      if (position < iteration.size())
        result[index] = iteration[position];
    }
    return result;
  }
  if (auto reduce = dyn_cast<linalg::ReduceOp>(consumer)) {
    if (operand.getOperandNumber() >= reduce.getInputs().size())
      return resizeMapping(operandMapping, *resultRank);
    DenseSet<int64_t> reductionDimensions;
    for (int64_t dimension : reduce.getDimensions())
      reductionDimensions.insert(dimension);
    unsigned resultDimension = 0;
    for (unsigned dimension = 0; dimension < operandMapping.size();
         ++dimension) {
      if (reductionDimensions.contains(dimension))
        continue;
      if (resultDimension < result.size())
        result[resultDimension] = operandMapping[dimension];
      ++resultDimension;
    }
    return result;
  }
  if (auto transpose = dyn_cast<linalg::TransposeOp>(consumer)) {
    if (operand.getOperandNumber() !=
        transpose.getInputMutable().getOperandNumber())
      return resizeMapping(operandMapping, *resultRank);
    ArrayRef<int64_t> permutation = transpose.getPermutation();
    for (unsigned outputDimension = 0; outputDimension < permutation.size() &&
                                       outputDimension < result.size();
         ++outputDimension) {
      int64_t inputDimension = permutation[outputDimension];
      if (inputDimension >= 0 &&
          static_cast<unsigned>(inputDimension) < operandMapping.size())
        result[outputDimension] = operandMapping[inputDimension];
    }
    return result;
  }
  if (auto expand = dyn_cast<tensor::ExpandShapeOp>(consumer)) {
    if (operand.get() != expand.getSrc())
      return result;
    ArrayRef<int64_t> shape = expand.getResultType().getShape();
    for (auto [sourceDimension, group] :
         llvm::enumerate(expand.getReassociationIndices())) {
      if (sourceDimension >= operandMapping.size() ||
          !operandMapping[sourceDimension])
        continue;
      if (group.size() == 1) {
        result[group.front()] = operandMapping[sourceDimension];
        continue;
      }
      if (groupedRootDims)
        groupedRootDims->push_back(*operandMapping[sourceDimension]);
      // The axis lands on the group's only non-unit dimension, if unique.
      std::optional<int64_t> carrier;
      unsigned nonUnit = 0;
      for (int64_t outputDimension : group) {
        if (shape[outputDimension] != 1) {
          carrier = outputDimension;
          ++nonUnit;
        }
      }
      if (nonUnit == 1)
        result[*carrier] = operandMapping[sourceDimension];
    }
    return result;
  }
  if (auto broadcast = dyn_cast<linalg::BroadcastOp>(consumer)) {
    if (operand.getOperandNumber() !=
        broadcast.getInputMutable().getOperandNumber())
      return resizeMapping(operandMapping, *resultRank);
    ArrayRef<int64_t> dimensions = broadcast.getDimensions();
    unsigned inputDimension = 0;
    for (unsigned dimension = 0; dimension < result.size(); ++dimension) {
      if (llvm::is_contained(dimensions, static_cast<int64_t>(dimension)))
        continue;
      if (inputDimension < operandMapping.size())
        result[dimension] = operandMapping[inputDimension];
      ++inputDimension;
    }
    return result;
  }
  if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(consumer)) {
    if (operand.get() != collapse.getSrc())
      return result;
    for (auto [outputDimension, group] :
         llvm::enumerate(collapse.getReassociationIndices())) {
      for (int64_t sourceDimension : group) {
        if (static_cast<unsigned>(sourceDimension) >= operandMapping.size() ||
            !operandMapping[sourceDimension])
          continue;
        if (group.size() > 1 && groupedRootDims)
          groupedRootDims->push_back(*operandMapping[sourceDimension]);
        if (!result[outputDimension])
          result[outputDimension] = operandMapping[sourceDimension];
      }
    }
    return result;
  }
  // Rank-preserving ops (slices, fill, element-wise tensor ops): positions
  // carry over only when the ranks agree.
  if (operandMapping.size() == *resultRank)
    return operandMapping;
  return result;
}

void AxisTracker::propagateToUsers() {
  // Monotone fixed point: entries are only filled, never changed, so the
  // worklist drains.
  SmallVector<Value> worklist(valueOrder.begin(), valueOrder.end());
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    DimMapping mapping = valueMappings.lookup(value);
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (!omega.contains(user) || user->getNumResults() == 0 ||
          isShapeOperand(user, use))
        continue;
      DimMapping resultMapping =
          getResultMappingFromOperand(user, use, mapping, nullptr);
      if (resultMapping.empty())
        continue;
      Value result = user->getResult(0);
      if (!recordValue(result, resultMapping))
        continue;
      worklist.push_back(result);
      // Newly known result entries also reach the user's other operands.
      DimMapping iteration =
          getIterationMapping(user, valueMappings.lookup(result));
      for (OpOperand &other : user->getOpOperands()) {
        if (&other == &use ||
            !isa<TensorType, BaseMemRefType>(other.get().getType()) ||
            isShapeOperand(user, other))
          continue;
        Operation *producer = other.get().getDefiningOp();
        if (!producer || !omega.contains(producer))
          continue;
        if (recordValue(other.get(),
                        getTrackerProducerMapping(user, other, iteration)))
          worklist.push_back(other.get());
      }
    }
  }
}

void AxisTracker::markGrouped() {
  for (Operation *op : omega) {
    if (!isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(op))
      continue;
    OpOperand &source = op->getOpOperand(0);
    const DimMapping *sourceMapping = getMapping(source.get());
    if (!sourceMapping)
      continue;
    SmallVector<unsigned> groupedRootDims;
    (void)getResultMappingFromOperand(op, source, *sourceMapping,
                                      &groupedRootDims);
    for (unsigned rootDim : groupedRootDims) {
      if (rootDim == mDim)
        grouped.insert({op, static_cast<unsigned>(Axis::M)});
      if (rootDim == nDim)
        grouped.insert({op, static_cast<unsigned>(Axis::N)});
    }
  }
}

std::optional<unsigned> AxisTracker::getExpectedPosition(OpOperand &use,
                                                         Axis axis) const {
  return getAxisPosition(use.get(), axis);
}

const DimMapping *AxisTracker::getMapping(Value value) const {
  auto it = valueMappings.find(value);
  return it == valueMappings.end() ? nullptr : &it->second;
}

std::optional<unsigned> AxisTracker::getAxisPosition(Value value,
                                                     Axis axis) const {
  const DimMapping *mapping = getMapping(value);
  if (!mapping)
    return std::nullopt;
  unsigned rootDim = getRootDim(axis);
  for (auto [position, entry] : llvm::enumerate(*mapping)) {
    if (entry && *entry == rootDim)
      return position;
  }
  return std::nullopt;
}

void AxisTracker::print(llvm::raw_ostream &os) const {
  for (Value value : valueOrder) {
    os << "  ";
    value.printAsOperand(os, OpPrintingFlags());
    os << ": m=";
    if (auto m = getAxisPosition(value, Axis::M))
      os << *m;
    else
      os << "-";
    os << " n=";
    if (auto n = getAxisPosition(value, Axis::N))
      os << *n;
    else
      os << "-";
    if (Operation *op = value.getDefiningOp()) {
      if (isGrouped(op, Axis::M))
        os << " grouped(m)";
      if (isGrouped(op, Axis::N))
        os << " grouped(n)";
    }
    os << "\n";
  }
}

} // namespace qgemm_coupling
} // namespace buddy
} // namespace mlir
