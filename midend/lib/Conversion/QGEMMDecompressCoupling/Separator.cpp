//===- Separator.cpp - Separation plan for QGEMM coupling -----------------===//
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
// An introduction of axis d happens on an edge (value v, consumer c) where c's
// result has a d axis and v does not. On a linalg.generic it is detected with
// the recognizer's getBroadcastEdge criterion: a parallel iteration dim mapped
// to d that the operand indexing map does not read.
//
//===----------------------------------------------------------------------===//

#include "Separator.h"
#include "Utils/MemrefView.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace buddy {
namespace qgemm_coupling {

static constexpr Axis kAxes[] = {Axis::M, Axis::N};

StringRef getAxisName(Axis axis) { return axis == Axis::M ? "m" : "n"; }

StringRef getUnsupportedFormName(UnsupportedForm form) {
  switch (form) {
  case UnsupportedForm::NonProjectedPermutationMap:
    return "non-projected-permutation map";
  case UnsupportedForm::NonZeroConstantMapResult:
    return "non-zero constant map result";
  case UnsupportedForm::NonGenericIntroducer:
    return "non-generic introducer";
  case UnsupportedForm::InconsistentBroadcastPositions:
    return "inconsistent broadcast positions";
  case UnsupportedForm::UnsupportedForm:
    return "unsupported form";
  }
  llvm_unreachable("unknown UnsupportedForm");
}

/// Iteration dimension of `generic` that carries `axis`, read through the
/// output map at the axis position of the first result.
static std::optional<unsigned>
getAxisLoop(linalg::GenericOp generic, const AxisTracker &tracker, Axis axis) {
  if (generic->getNumResults() == 0 || generic.getNumDpsInits() == 0)
    return std::nullopt;
  std::optional<unsigned> position =
      tracker.getAxisPosition(generic->getResult(0), axis);
  if (!position)
    return std::nullopt;
  AffineMap outputMap =
      generic.getMatchingIndexingMap(generic.getDpsInitOperand(0));
  if (*position >= outputMap.getNumResults())
    return std::nullopt;
  auto dimension = dyn_cast<AffineDimExpr>(outputMap.getResult(*position));
  if (!dimension)
    return std::nullopt;
  return dimension.getPosition();
}

/// Generic form: one input, one init, all-parallel iterators, identity output
/// map, input map equal to the identity minus exactly one dimension that is a
/// tracked m or n, and a body that only yields the input block argument.
static std::optional<Axis>
getGenericPureBroadcastAxis(linalg::GenericOp generic,
                            const AxisTracker &tracker) {
  if (generic.getNumDpsInputs() != 1 || generic.getNumDpsInits() != 1 ||
      generic->getNumResults() != 1)
    return std::nullopt;
  if (generic.getNumParallelLoops() != generic.getNumLoops())
    return std::nullopt;
  AffineMap outputMap =
      generic.getMatchingIndexingMap(generic.getDpsInitOperand(0));
  if (!outputMap.isIdentity())
    return std::nullopt;
  AffineMap inputMap =
      generic.getMatchingIndexingMap(generic.getDpsInputOperand(0));
  unsigned loops = generic.getNumLoops();
  if (inputMap.getNumResults() + 1 != loops)
    return std::nullopt;
  // The input map must be the identity with exactly one dimension omitted.
  std::optional<unsigned> omitted;
  unsigned next = 0;
  for (unsigned loop = 0; loop < loops; ++loop) {
    AffineExpr expression;
    if (next < inputMap.getNumResults())
      expression = inputMap.getResult(next);
    auto dimension = dyn_cast_if_present<AffineDimExpr>(expression);
    if (dimension && dimension.getPosition() == loop) {
      ++next;
      continue;
    }
    if (omitted)
      return std::nullopt;
    omitted = loop;
  }
  if (!omitted || next != inputMap.getNumResults())
    return std::nullopt;

  Block &body = generic.getRegion().front();
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || body.getOperations().size() != 1 ||
      yield.getNumOperands() != 1 || yield.getOperand(0) != body.getArgument(0))
    return std::nullopt;

  for (Axis axis : kAxes) {
    if (tracker.getAxisPosition(generic->getResult(0), axis) == omitted)
      return axis;
  }
  return std::nullopt;
}

std::optional<Axis> getPureBroadcastAxis(Operation *op,
                                         const AxisTracker &tracker) {
  if (auto generic = dyn_cast<linalg::GenericOp>(op))
    return getGenericPureBroadcastAxis(generic, tracker);
  if (auto broadcast = dyn_cast<linalg::BroadcastOp>(op)) {
    ArrayRef<int64_t> dimensions = broadcast.getDimensions();
    if (dimensions.size() != 1 || broadcast->getNumResults() != 1)
      return std::nullopt;
    for (Axis axis : kAxes) {
      std::optional<unsigned> position =
          tracker.getAxisPosition(broadcast->getResult(0), axis);
      if (position && static_cast<int64_t>(*position) == dimensions.front())
        return axis;
    }
  }
  return std::nullopt;
}

namespace {

class SeparationPlanner {
public:
  SeparationPlanner(const SetVector<Operation *> &omega,
                    const AxisTracker &tracker)
      : omega(omega), tracker(tracker) {}

  SeparationPlan run() {
    for (Operation *op : omega) {
      if (op->getNumResults() == 0)
        continue;
      if (std::optional<Axis> axis = getPureBroadcastAxis(op, tracker)) {
        plan.existingPBs.push_back(op);
        // The rebuild emits only the generic form (Req 3.3).
        if (isa<linalg::BroadcastOp>(op))
          ++plan.rewrites;
        continue;
      }
      if (auto generic = dyn_cast<linalg::GenericOp>(op))
        visitGeneric(generic);
      else
        visitNonGeneric(op);
    }
    return std::move(plan);
  }

private:
  void addR1(Operation *op, Axis axis, unsigned operandNumber,
             UnsupportedForm form) {
    plan.r1Diagnostics.push_back({op, axis, operandNumber, form});
  }

  void visitGeneric(linalg::GenericOp generic) {
    std::optional<unsigned> axisLoop[2] = {
        getAxisLoop(generic, tracker, Axis::M),
        getAxisLoop(generic, tracker, Axis::N)};
    SmallVector<utils::IteratorType> iterators =
        generic.getIteratorTypesArray();

    for (OpOperand *operand : generic.getDpsInputOperands()) {
      auto operandType = dyn_cast<RankedTensorType>(operand->get().getType());
      if (!operandType)
        continue;
      AffineMap map = generic.getMatchingIndexingMap(operand);
      SmallVector<bool> used(generic.getNumLoops(), false);
      for (AffineExpr expression : map.getResults())
        if (auto dimension = dyn_cast<AffineDimExpr>(expression))
          used[dimension.getPosition()] = true;

      // getBroadcastEdge criterion restricted to the tracked m / n loops.
      SmallVector<Axis, 2> axes;
      for (Axis axis : kAxes) {
        std::optional<unsigned> loop = axisLoop[static_cast<unsigned>(axis)];
        if (loop && !linalg::isReductionIterator(iterators[*loop]) &&
            !used[*loop])
          axes.push_back(axis);
      }
      if (axes.empty())
        continue;

      unsigned operandNumber = operand->getOperandNumber();
      Introduction introduction{generic, operandNumber, axes, {}, {},
                                {},      false};
      SmallVector<AffineExpr> kept;
      SmallVector<bool> seen(generic.getNumLoops(), false);
      std::optional<UnsupportedForm> failure;
      for (auto [index, expression] : llvm::enumerate(map.getResults())) {
        if (auto dimension = dyn_cast<AffineDimExpr>(expression)) {
          if (seen[dimension.getPosition()]) {
            failure = UnsupportedForm::NonProjectedPermutationMap;
            break;
          }
          seen[dimension.getPosition()] = true;
          kept.push_back(expression);
          continue;
        }
        if (auto constant = dyn_cast<AffineConstantExpr>(expression)) {
          if (constant.getValue() != 0) {
            failure = UnsupportedForm::NonZeroConstantMapResult;
            break;
          }
          // A constant-0 result must read a unit axis (Req 2.4).
          if (operandType.getDimSize(index) != 1) {
            failure = UnsupportedForm::UnsupportedForm;
            break;
          }
          introduction.unitDrops.push_back(index);
          continue;
        }
        failure = UnsupportedForm::NonProjectedPermutationMap;
        break;
      }
      if (failure) {
        for (Axis axis : axes)
          addR1(generic, axis, operandNumber, *failure);
        continue;
      }

      // Broadcast_Position (Req 2.6): the split-out PB's result has the
      // consumer's iteration-domain order restricted to the operand's dims
      // plus the inserted axis, so the axis goes after every operand dim
      // read through a lower loop. Nothing here depends on m/n order.
      MLIRContext *context = generic.getContext();
      SmallVector<unsigned, 2> loops;
      for (Axis axis : axes)
        loops.push_back(*axisLoop[static_cast<unsigned>(axis)]);
      llvm::sort(loops);
      for (unsigned loop : loops) {
        unsigned at = llvm::count_if(kept, [&](AffineExpr expression) {
          return cast<AffineDimExpr>(expression).getPosition() < loop;
        });
        kept.insert(kept.begin() + at, getAffineDimExpr(loop, context));
      }
      introduction.newOperandMap =
          AffineMap::get(generic.getNumLoops(), 0, kept, context);
      for (Axis axis : axes) {
        AffineExpr loop =
            getAffineDimExpr(*axisLoop[static_cast<unsigned>(axis)], context);
        introduction.positions.push_back(llvm::find(kept, loop) - kept.begin());
      }
      introduction.identityOperandMap = introduction.newOperandMap.isIdentity();
      plan.introductions.push_back(std::move(introduction));
      ++plan.rewrites;
    }
  }

  /// Any other op whose result has an axis none of its data operands has is
  /// an introduction through a non-generic op (Req 2.5).
  void visitNonGeneric(Operation *op) {
    // A load chain's memref view op reads an unranked or differently shaped
    // memref and states the physical layout of that load. Its axes come from
    // memory rather than from a broadcast of a value of Omega(r), so it is
    // not an introduction.
    if (isMemrefView(op))
      return;
    SmallVector<OpOperand *> dataOperands;
    if (auto dps = dyn_cast<DestinationStyleOpInterface>(op)) {
      for (OpOperand *operand : dps.getDpsInputOperands())
        if (isa<ShapedType>(operand->get().getType()))
          dataOperands.push_back(operand);
    } else {
      for (OpOperand &operand : op->getOpOperands())
        if (isa<ShapedType>(operand.get().getType()))
          dataOperands.push_back(&operand);
    }
    if (dataOperands.empty())
      return;
    for (Axis axis : kAxes) {
      if (!tracker.getAxisPosition(op->getResult(0), axis))
        continue;
      bool carried = llvm::any_of(dataOperands, [&](OpOperand *operand) {
        return tracker.getAxisPosition(operand->get(), axis).has_value();
      });
      if (!carried)
        addR1(op, axis, dataOperands.front()->getOperandNumber(),
              UnsupportedForm::NonGenericIntroducer);
    }
  }

  const SetVector<Operation *> &omega;
  const AxisTracker &tracker;
  SeparationPlan plan;
};

} // namespace

SeparationPlan computeSeparationPlan(const SetVector<Operation *> &omega,
                                     const AxisTracker &tracker) {
  return SeparationPlanner(omega, tracker).run();
}

SmallVector<AffineMap> getSeparatedMaps(Operation *op,
                                        const SeparationPlan &plan) {
  auto generic = dyn_cast<linalg::GenericOp>(op);
  if (!generic)
    return {};
  SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
  for (const Introduction &introduction : plan.introductions)
    if (introduction.consumer == op)
      maps[introduction.operandNumber] = introduction.newOperandMap;
  return maps;
}

void emitR1Remarks(const SeparationPlan &plan, unsigned rootId) {
  for (const R1Diagnostic &diagnostic : plan.r1Diagnostics) {
    diagnostic.op->emitRemark()
        << "qgemm-decompress-coupling: unsupported introduction of parallel "
           "dimension '"
        << getAxisName(diagnostic.axis) << "' on operand #"
        << diagnostic.operandNumber << " ("
        << getUnsupportedFormName(diagnostic.form) << "); root " << rootId
        << " left unchanged";
  }
}

void SeparationPlan::print(llvm::raw_ostream &os) const {
  os << "  introductions: " << introductions.size()
     << ", existing PBs: " << existingPBs.size() << ", rewrites: " << rewrites
     << ", R1: " << r1Diagnostics.size() << "\n";
  for (const Introduction &introduction : introductions) {
    os << "    intro " << introduction.consumer->getName() << " operand #"
       << introduction.operandNumber << " axes=";
    for (Axis axis : introduction.axes)
      os << getAxisName(axis);
    os << " producer=";
    Value operand =
        introduction.consumer->getOperand(introduction.operandNumber);
    if (Operation *producer = operand.getDefiningOp())
      os << producer->getName();
    else
      os << "block-arg";
    os << " positions=";
    llvm::interleave(introduction.positions, os, ",");
    os << " unitDrops=" << introduction.unitDrops.size()
       << " newMap=" << introduction.newOperandMap
       << (introduction.identityOperandMap ? " (identity)" : "") << "\n";
  }
  for (Operation *pb : existingPBs)
    os << "    existing PB " << pb->getName() << "\n";
}

} // namespace qgemm_coupling
} // namespace buddy
} // namespace mlir
