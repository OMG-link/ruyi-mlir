//===- RuleTable.cpp - Per-op axis rules for QGEMM coupling ---------------===//
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
// Per-op-kind strip / insert rules and the rebuild that applies them (Req 7).
//
// Supported kinds, one TypeSwitch case each:
//   - linalg.generic (element-wise, cast, tensor.extract / linalg.index
//     lookups, reduction iterators with d parallel)
//   - linalg.reduce (d a kept dimension)
//   - linalg.transpose
//   - tensor.expand_shape / tensor.collapse_shape (d its own group)
//   - tensor.reshape with static shapes, treated like expand/collapse: d must
//     sit on a boundary where the prefix extents of source and result agree,
//     which is the reshape analogue of a singleton reassociation group
//   - tensor.extract_slice / tensor.insert_slice, rank-reducing ones too, with
//     the axis on a full-extent unit-stride entry
//   - tensor.empty, linalg.fill
// Every other kind answers false, so the caller raises R2 / R3 (Req 7.6).
//
// Rebuild works on dimension labels: the new dims of a value are listed as
// either the index of an old dim or a marker for an axis inserted at its
// target position. Axes present before and after are removed and reinserted
// as markers, so a position change is handled by the same path.
//
//===----------------------------------------------------------------------===//

#include "RuleTable.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace buddy {
namespace qgemm_coupling {

static constexpr Axis kAxes[] = {Axis::M, Axis::N};

AxisLayout AxisLayout::of(Value value, const AxisTracker &tracker) {
  AxisLayout layout;
  for (Axis axis : kAxes)
    layout.set(axis, tracker.getAxisPosition(value, axis));
  return layout;
}

AxisLayout AxisLayout::kept(Value value, bool keepM, bool keepN,
                            const AxisTracker &tracker) {
  AxisLayout layout = AxisLayout::of(value, tracker);
  bool keep[] = {keepM, keepN};
  for (Axis axis : kAxes)
    if (!keep[static_cast<unsigned>(axis)] && layout.get(axis))
      layout = layout.without(axis);
  return layout;
}

AxisLayout AxisLayout::withInserted(Axis axis, unsigned position) const {
  assert(!get(axis) && "axis inserted twice");
  AxisLayout layout = *this;
  for (Axis other : kAxes)
    if (std::optional<unsigned> pos = get(other); pos && *pos >= position)
      layout.set(other, *pos + 1);
  layout.set(axis, position);
  return layout;
}

AxisLayout AxisLayout::without(Axis axis) const {
  std::optional<unsigned> removed = get(axis);
  assert(removed && "removing an axis the layout lacks");
  AxisLayout layout = *this;
  layout.set(axis, std::nullopt);
  for (Axis other : kAxes)
    if (std::optional<unsigned> pos = layout.get(other); pos && *pos > *removed)
      layout.set(other, *pos - 1);
  return layout;
}

std::optional<int64_t> RuleTable::getAxisSize(Axis axis) const {
  auto type = dyn_cast<RankedTensorType>(
      tracker.getFinalReduce()->getResult(0).getType());
  if (!type)
    return std::nullopt;
  unsigned dim = tracker.getRootDim(axis);
  if (dim >= type.getRank() || type.isDynamicDim(dim))
    return std::nullopt;
  return type.getDimSize(dim);
}

//===----------------------------------------------------------------------===//
// Label helpers
//===----------------------------------------------------------------------===//

namespace {

/// Marker label of an inserted axis.
int64_t axisMarker(Axis axis) { return -1 - static_cast<int64_t>(axis); }
Axis markerAxis(int64_t label) {
  assert(label < 0 && "not an axis marker");
  return static_cast<Axis>(-1 - label);
}

/// New dims of a value of rank `rank` whose axes move from `from` to `to`:
/// each entry is an old dim index or an axis marker.
SmallVector<int64_t> relabel(unsigned rank, AxisLayout from, AxisLayout to) {
  SmallVector<int64_t> labels;
  for (unsigned i = 0; i < rank; ++i)
    labels.push_back(i);
  SmallVector<unsigned> removed;
  for (Axis axis : kAxes)
    if (std::optional<unsigned> pos = from.get(axis)) {
      assert(*pos < rank && "axis position out of range");
      removed.push_back(*pos);
    }
  llvm::sort(removed, std::greater<unsigned>());
  for (unsigned pos : removed)
    labels.erase(labels.begin() + pos);
  SmallVector<std::pair<unsigned, int64_t>> inserted;
  for (Axis axis : kAxes)
    if (std::optional<unsigned> pos = to.get(axis))
      inserted.push_back({*pos, axisMarker(axis)});
  llvm::sort(inserted);
  for (auto [pos, marker] : inserted) {
    if (pos > labels.size())
      llvm_unreachable("target axis position beyond the rebuilt rank");
    labels.insert(labels.begin() + pos, marker);
  }
  return labels;
}

/// Index of `label` in `labels`, which must contain it.
unsigned indexOf(ArrayRef<int64_t> labels, int64_t label) {
  const auto *it = llvm::find(labels, label);
  if (it == labels.end())
    llvm_unreachable("dimension lost while rebuilding");
  return it - labels.begin();
}

bool isStatic(Type type) {
  auto shaped = dyn_cast<RankedTensorType>(type);
  return shaped && shaped.hasStaticShape();
}

bool hasStaticResults(Operation *op) {
  return op->getNumResults() == 1 && isStatic(op->getResult(0).getType());
}

int64_t product(ArrayRef<int64_t> shape) {
  int64_t result = 1;
  for (int64_t size : shape)
    result *= size;
  return result;
}

/// `value` with an SSA operand replaced by its rebuilt counterpart; a static
/// attribute passes through.
OpFoldResult remapFoldResult(OpFoldResult value, const IRMapping &mapping) {
  if (auto ssa = dyn_cast<Value>(value))
    return mapping.lookupOrDefault(ssa);
  return value;
}

bool isConstantIndex(OpFoldResult value, int64_t expected) {
  std::optional<int64_t> constant = getConstantIntValue(value);
  return constant && *constant == expected;
}

/// Group of `groups` containing dim `dim`.
std::optional<unsigned> findGroup(ArrayRef<ReassociationIndices> groups,
                                  unsigned dim) {
  for (auto [index, group] : llvm::enumerate(groups))
    if (llvm::is_contained(group, static_cast<int64_t>(dim)))
      return index;
  return std::nullopt;
}

} // namespace

//===----------------------------------------------------------------------===//
// Rule implementation
//===----------------------------------------------------------------------===//

namespace {

class Rules {
public:
  Rules(const RuleTable &table, const AxisTracker &tracker)
      : table(table), tracker(tracker) {}

  RankedTensorType relabelType(Type oldType, ArrayRef<int64_t> labels) const {
    auto type = cast<RankedTensorType>(oldType);
    SmallVector<int64_t> shape;
    for (int64_t label : labels) {
      if (label >= 0) {
        shape.push_back(type.getDimSize(label));
        continue;
      }
      std::optional<int64_t> size = table.getAxisSize(markerAxis(label));
      if (!size)
        llvm_unreachable("inserted axis has no static extent");
      shape.push_back(*size);
    }
    return RankedTensorType::get(shape, type.getElementType(),
                                 type.getEncoding());
  }

  std::optional<unsigned> resultPos(Operation *op, Axis axis) const {
    return tracker.getAxisPosition(op->getResult(0), axis);
  }

  //===--- linalg.generic -------------------------------------------------===//

  /// Iteration dim carrying the result's `axis`, read through the output map.
  std::optional<unsigned> axisLoop(linalg::GenericOp generic,
                                   std::optional<unsigned> pos) const {
    if (!pos)
      return std::nullopt;
    AffineMap map =
        generic.getMatchingIndexingMap(generic.getDpsInitOperand(0));
    if (*pos >= map.getNumResults())
      return std::nullopt;
    auto dim = dyn_cast<AffineDimExpr>(map.getResult(*pos));
    if (!dim)
      return std::nullopt;
    return dim.getPosition();
  }

  bool isSingleResultGeneric(linalg::GenericOp generic) const {
    return generic->getNumResults() == 1 && generic.getNumDpsInits() == 1 &&
           hasStaticResults(generic);
  }

  bool canStrip(linalg::GenericOp generic, Axis axis) const {
    if (!isSingleResultGeneric(generic))
      return false;
    std::optional<unsigned> loop = axisLoop(generic, resultPos(generic, axis));
    if (!loop)
      return false;
    assert(generic.getIteratorTypesArray()[*loop] ==
               utils::IteratorType::parallel &&
           "the axis of a QGEMM root is always a parallel iterator");
    for (AffineMap map : generic.getIndexingMapsArray())
      for (AffineExpr expr : map.getResults())
        if (expr.isFunctionOfDim(*loop) && !isa<AffineDimExpr>(expr))
          return false;
    // Reading the coordinate makes the op vary along the axis (Req 4.2).
    WalkResult walk = generic.getRegion().walk([&](linalg::IndexOp index) {
      return index.getDim() == *loop ? WalkResult::interrupt()
                                     : WalkResult::advance();
    });
    return !walk.wasInterrupted();
  }

  bool canInsert(linalg::GenericOp generic, Axis axis) const {
    return isSingleResultGeneric(generic) && table.getAxisSize(axis);
  }

  /// Iteration domain of a generic rebuilt with result layout `target`.
  struct GenericLoops {
    /// Old loop index, or an axis marker for an inserted axis.
    SmallVector<int64_t> loops;
    std::optional<unsigned> oldLoop[2];
    std::optional<unsigned> newLoop[2];
  };

  /// Loops of `generic` rebuilt with result layout `target`: stripped axis
  /// loops removed, kept ones in place, and an inserted axis placed right
  /// after the last loop the output map reads ahead of the axis's result
  /// position, so an identity output map stays the identity.
  GenericLoops computeLoops(linalg::GenericOp generic,
                            AxisLayout target) const {
    GenericLoops result;
    AxisLayout oldLayout = AxisLayout::of(generic->getResult(0), tracker);
    for (Axis axis : kAxes) {
      result.oldLoop[static_cast<unsigned>(axis)] =
          axisLoop(generic, oldLayout.get(axis));
      if (oldLayout.get(axis) && !result.oldLoop[static_cast<unsigned>(axis)])
        llvm_unreachable("result axis not read through a loop dim");
    }
    SmallVector<int64_t> &loops = result.loops;
    for (unsigned i = 0, e = generic.getNumLoops(); i < e; ++i)
      loops.push_back(i);
    for (Axis axis : kAxes) {
      unsigned a = static_cast<unsigned>(axis);
      if (result.oldLoop[a] && !target.get(axis))
        loops.erase(
            llvm::find(loops, static_cast<int64_t>(*result.oldLoop[a])));
    }
    AffineMap outputMap =
        generic.getMatchingIndexingMap(generic.getDpsInitOperand(0));
    SmallVector<int64_t> outputLabels =
        relabel(outputMap.getNumResults(), oldLayout, target);
    // Loop label read by output label `label`, or none.
    auto loopOf = [&](int64_t label) -> std::optional<int64_t> {
      if (label >= 0) {
        auto dim = dyn_cast<AffineDimExpr>(outputMap.getResult(label));
        if (!dim)
          return std::nullopt;
        return dim.getPosition();
      }
      unsigned a = static_cast<unsigned>(markerAxis(label));
      if (result.oldLoop[a])
        return *result.oldLoop[a];
      return label;
    };
    SmallVector<std::pair<unsigned, Axis>> inserted;
    for (Axis axis : kAxes)
      if (target.get(axis) && !result.oldLoop[static_cast<unsigned>(axis)])
        inserted.push_back({*target.get(axis), axis});
    llvm::sort(inserted, [](auto x, auto y) { return x.first < y.first; });
    for (auto [position, axis] : inserted) {
      unsigned at = 0;
      for (int64_t label : ArrayRef<int64_t>(outputLabels).take_front(position))
        if (std::optional<int64_t> loop = loopOf(label)) {
          const auto *it = llvm::find(loops, *loop);
          if (it != loops.end())
            at = std::max<unsigned>(at, it - loops.begin() + 1);
        }
      loops.insert(loops.begin() + at, axisMarker(axis));
    }
    for (Axis axis : kAxes) {
      unsigned a = static_cast<unsigned>(axis);
      if (!target.get(axis))
        continue;
      result.newLoop[a] = indexOf(
          loops, result.oldLoop[a] ? static_cast<int64_t>(*result.oldLoop[a])
                                   : axisMarker(axis));
    }
    return result;
  }

  /// Required layout of a generic input read through `map` (Req 6.8): an
  /// axis the map reads stays where the map reads it, a stripped one is
  /// dropped, and an inserted one takes the position its loop has among the
  /// loops the map reads.
  AxisLayout operandLayout(linalg::GenericOp generic, AffineMap map,
                           AxisLayout target) const {
    GenericLoops loops = computeLoops(generic, target);
    // Each kept result: the new loop index it is ordered by and the axis it
    // carries, if any.
    struct Entry {
      int64_t key;
      std::optional<Axis> axis;
    };
    SmallVector<Entry> entries;
    for (AffineExpr expr : map.getResults()) {
      std::optional<Axis> carried;
      bool stripped = false;
      for (Axis axis : kAxes) {
        std::optional<unsigned> loop =
            loops.oldLoop[static_cast<unsigned>(axis)];
        if (!loop || expr != getAffineDimExpr(*loop, generic.getContext()))
          continue;
        if (target.get(axis))
          carried = axis;
        else
          stripped = true;
      }
      if (stripped)
        continue;
      int64_t key = -1;
      for (auto [index, label] : llvm::enumerate(loops.loops))
        if (label >= 0 && expr.isFunctionOfDim(label))
          key = std::max<int64_t>(key, index);
      entries.push_back({key, carried});
    }
    SmallVector<std::pair<unsigned, Axis>> inserted;
    for (Axis axis : kAxes) {
      unsigned a = static_cast<unsigned>(axis);
      if (!target.get(axis) ||
          llvm::any_of(entries, [&](const Entry &e) { return e.axis == axis; }))
        continue;
      inserted.push_back({*loops.newLoop[a], axis});
    }
    llvm::sort(inserted, [](auto x, auto y) { return x.first < y.first; });
    for (auto [loop, axis] : inserted) {
      unsigned at = llvm::count_if(entries, [&](const Entry &e) {
        return e.key < static_cast<int64_t>(loop);
      });
      entries.insert(entries.begin() + at, {static_cast<int64_t>(loop), axis});
    }
    AxisLayout layout;
    for (auto [index, entry] : llvm::enumerate(entries))
      if (entry.axis)
        layout.set(*entry.axis, index);
    return layout;
  }

  Operation *rebuild(OpBuilder &builder, linalg::GenericOp generic,
                     ArrayRef<Value> newOperands,
                     ArrayRef<AxisLayout> operandLayouts,
                     AxisLayout target) const {
    if (!isSingleResultGeneric(generic))
      llvm_unreachable("rebuild of an unsupported linalg.generic");
    AxisLayout oldLayout = AxisLayout::of(generic->getResult(0), tracker);
    unsigned numLoops = generic.getNumLoops();
    GenericLoops computed = computeLoops(generic, target);
    SmallVector<int64_t> &loops = computed.loops;
    std::optional<unsigned> *oldLoop = computed.oldLoop;
    std::optional<unsigned> *newLoop = computed.newLoop;

    MLIRContext *context = builder.getContext();
    SmallVector<AffineExpr> dimReplacements;
    for (unsigned i = 0; i < numLoops; ++i) {
      const auto *it = llvm::find(loops, static_cast<int64_t>(i));
      dimReplacements.push_back(
          it == loops.end() ? getAffineConstantExpr(0, context)
                            : getAffineDimExpr(it - loops.begin(), context));
    }
    SmallVector<utils::IteratorType> oldIterators =
        generic.getIteratorTypesArray();
    SmallVector<utils::IteratorType> iterators;
    for (int64_t label : loops)
      iterators.push_back(label >= 0 ? oldIterators[label]
                                     : utils::IteratorType::parallel);

    auto rebuildMap = [&](AffineMap oldMap, AxisLayout to) {
      AxisLayout from;
      for (Axis axis : kAxes) {
        unsigned a = static_cast<unsigned>(axis);
        if (!oldLoop[a])
          continue;
        for (auto [index, expr] : llvm::enumerate(oldMap.getResults()))
          if (expr == getAffineDimExpr(*oldLoop[a], context))
            from.set(axis, index);
      }
      SmallVector<int64_t> labels = relabel(oldMap.getNumResults(), from, to);
      SmallVector<AffineExpr> results;
      for (int64_t label : labels) {
        if (label >= 0) {
          AffineExpr expr = oldMap.getResult(label);
          for (Axis axis : kAxes) {
            unsigned a = static_cast<unsigned>(axis);
            if (oldLoop[a] && !newLoop[a] && expr.isFunctionOfDim(*oldLoop[a]))
              llvm_unreachable("stripped axis still read by an operand");
          }
          results.push_back(expr.replaceDimsAndSymbols(dimReplacements, {}));
          continue;
        }
        std::optional<unsigned> loop =
            newLoop[static_cast<unsigned>(markerAxis(label))];
        if (!loop)
          llvm_unreachable("operand carries an axis the rebuilt op lacks");
        results.push_back(getAffineDimExpr(*loop, context));
      }
      return AffineMap::get(loops.size(), 0, results, context);
    };

    SmallVector<AffineMap> oldMaps = generic.getIndexingMapsArray();
    SmallVector<AffineMap> maps;
    SmallVector<Value> inputs, outputs;
    AxisLayout initFrom = oldLayout;
    RankedTensorType resultType = relabelType(
        generic->getResult(0).getType(),
        relabel(
            cast<RankedTensorType>(generic->getResult(0).getType()).getRank(),
            initFrom, target));
    for (OpOperand &operand : generic->getOpOperands()) {
      unsigned number = operand.getOperandNumber();
      bool isInit = generic.isDpsInit(&operand);
      maps.push_back(rebuildMap(oldMaps[number],
                                isInit ? target : operandLayouts[number]));
      if (!isInit) {
        inputs.push_back(newOperands[number]);
        continue;
      }
      // Req 7.2: an init the body does not read becomes a fresh empty tensor.
      if (!generic.payloadUsesValueFromOperand(&operand)) {
        outputs.push_back(builder.create<tensor::EmptyOp>(
            generic.getLoc(), resultType.getShape(),
            resultType.getElementType()));
        continue;
      }
      if (newOperands[number].getType() != resultType)
        llvm_unreachable("read init does not match the rebuilt result");
      outputs.push_back(newOperands[number]);
    }

    auto rebuilt = builder.create<linalg::GenericOp>(
        generic.getLoc(), TypeRange{resultType}, inputs, outputs, maps,
        iterators);
    IRMapping mapping;
    generic.getRegion().cloneInto(&rebuilt.getRegion(), mapping);
    // Req 7.3: renumber linalg.index; the tensor.extract table is captured
    // from above and stays untouched.
    rebuilt.getRegion().walk([&](linalg::IndexOp index) {
      const auto *it = llvm::find(loops, static_cast<int64_t>(index.getDim()));
      if (it == loops.end())
        llvm_unreachable("linalg.index reads a stripped axis");
      index.setDim(it - loops.begin());
    });
    rebuilt->setDiscardableAttrs(generic->getDiscardableAttrDictionary());
    return rebuilt;
  }

  //===--- linalg.reduce --------------------------------------------------===//

  /// Input dim of a reduce that becomes result dim `resultDim`.
  static std::optional<unsigned> keptInputDim(linalg::ReduceOp reduce,
                                              unsigned resultDim) {
    auto inputType = cast<ShapedType>(reduce.getInputs()[0].getType());
    unsigned kept = 0;
    for (unsigned i = 0; i < inputType.getRank(); ++i) {
      if (llvm::is_contained(reduce.getDimensions(), static_cast<int64_t>(i)))
        continue;
      if (kept++ == resultDim)
        return i;
    }
    return std::nullopt;
  }

  bool isSupportedReduce(linalg::ReduceOp reduce) const {
    return reduce->getNumResults() >= 1 &&
           llvm::all_of(reduce->getOperandTypes(), isStatic);
  }

  bool canStrip(linalg::ReduceOp reduce, Axis axis) const {
    if (!isSupportedReduce(reduce))
      return false;
    std::optional<unsigned> pos = resultPos(reduce, axis);
    if (!pos)
      return false;
    std::optional<unsigned> inputDim = keptInputDim(reduce, *pos);
    assert(inputDim && "the axis of a QGEMM root is never a reduced dim");
    (void)inputDim;
    return true;
  }

  bool canInsert(linalg::ReduceOp reduce, Axis axis) const {
    return isSupportedReduce(reduce) && table.getAxisSize(axis).has_value();
  }

  Operation *rebuild(OpBuilder &builder, linalg::ReduceOp reduce,
                     ArrayRef<Value> newOperands,
                     ArrayRef<AxisLayout> operandLayouts,
                     AxisLayout target) const {
    AxisLayout oldLayout = AxisLayout::of(reduce->getResult(0), tracker);
    AxisLayout oldInput;
    for (Axis axis : kAxes)
      if (std::optional<unsigned> pos = oldLayout.get(axis))
        oldInput.set(axis, keptInputDim(reduce, *pos));
    auto inputType = cast<RankedTensorType>(reduce.getInputs()[0].getType());
    auto resultType = cast<RankedTensorType>(reduce->getResult(0).getType());
    SmallVector<int64_t> inputLabels =
        relabel(inputType.getRank(), oldInput, operandLayouts[0]);
    SmallVector<int64_t> resultLabels =
        relabel(resultType.getRank(), oldLayout, target);
    SmallVector<int64_t> dimensions;
    SmallVector<int64_t> keptLabels;
    for (auto [index, label] : llvm::enumerate(inputLabels)) {
      if (label >= 0 && llvm::is_contained(reduce.getDimensions(), label)) {
        dimensions.push_back(index);
        continue;
      }
      // Express the kept input dim as the result label it becomes.
      if (label < 0) {
        keptLabels.push_back(label);
        continue;
      }
      unsigned resultDim = 0;
      for (int64_t i = 0; i < label; ++i)
        if (!llvm::is_contained(reduce.getDimensions(), i))
          ++resultDim;
      keptLabels.push_back(resultDim);
    }
    if (keptLabels != resultLabels)
      llvm_unreachable("reduce operand and target layouts disagree");

    IRMapping mapping;
    mapping.map(reduce->getOperands(), newOperands);
    Operation *rebuilt = builder.clone(*reduce, mapping);
    cast<linalg::ReduceOp>(rebuilt).setDimensions(dimensions);
    for (auto [result, init] : llvm::zip(
             rebuilt->getResults(), cast<linalg::ReduceOp>(rebuilt).getInits()))
      result.setType(init.getType());
    if (rebuilt->getResult(0).getType() !=
        relabelType(resultType, resultLabels))
      llvm_unreachable("reduce init does not match the rebuilt result");
    return rebuilt;
  }

  //===--- linalg.transpose -----------------------------------------------===//

  bool canStrip(linalg::TransposeOp transpose, Axis axis) const {
    return hasStaticResults(transpose) && resultPos(transpose, axis);
  }

  bool canInsert(linalg::TransposeOp transpose, Axis axis) const {
    return hasStaticResults(transpose) && table.getAxisSize(axis).has_value();
  }

  Operation *rebuild(OpBuilder &builder, linalg::TransposeOp transpose,
                     ArrayRef<Value> newOperands,
                     ArrayRef<AxisLayout> operandLayouts,
                     AxisLayout target) const {
    ArrayRef<int64_t> permutation = transpose.getPermutation();
    AxisLayout oldLayout = AxisLayout::of(transpose->getResult(0), tracker);
    AxisLayout oldInput;
    for (Axis axis : kAxes)
      if (std::optional<unsigned> pos = oldLayout.get(axis))
        oldInput.set(axis, permutation[*pos]);
    unsigned rank = permutation.size();
    SmallVector<int64_t> inputLabels =
        relabel(rank, oldInput, operandLayouts[0]);
    SmallVector<int64_t> resultLabels = relabel(rank, oldLayout, target);
    if (inputLabels.size() != resultLabels.size())
      llvm_unreachable("transpose operand and target ranks disagree");
    SmallVector<int64_t> newPermutation;
    for (int64_t label : resultLabels)
      newPermutation.push_back(
          indexOf(inputLabels, label >= 0 ? permutation[label] : label));

    IRMapping mapping;
    mapping.map(transpose->getOperands(), newOperands);
    Operation *rebuilt = builder.clone(*transpose, mapping);
    auto rebuiltTranspose = cast<linalg::TransposeOp>(rebuilt);
    rebuiltTranspose.setPermutation(newPermutation);
    rebuilt->getResult(0).setType(rebuiltTranspose.getInit().getType());
    if (rebuilt->getResult(0).getType() !=
        relabelType(transpose->getResult(0).getType(), resultLabels))
      llvm_unreachable("transpose init does not match the rebuilt result");
    return rebuilt;
  }

  //===--- tensor.expand_shape / tensor.collapse_shape --------------------===//

  /// `expanded` is the higher-rank side; groups index it by the collapsed
  /// side's dims.
  ///
  /// `axis` may share its group with other dims only when every other member
  /// has static extent 1: the group then keeps those unit members and the
  /// collapsed dim stays equal to the axis extent. A non-unit partner would
  /// make the collapsed dim's linear index depend on the axis, so that case
  /// has no rule (Req 7.6).
  bool canStripReshape(Operation *op, Axis axis,
                       ArrayRef<ReassociationIndices> groups,
                       bool resultIsExpanded) const {
    return hasOnlyUnitPartners(op, axis, groups, resultIsExpanded);
  }

  /// Whether every other member of `axis`'s reassociation group has static
  /// extent 1; trivially true for a singleton group.
  bool hasOnlyUnitPartners(Operation *op, Axis axis,
                           ArrayRef<ReassociationIndices> groups,
                           bool resultIsExpanded) const {
    if (!hasStaticResults(op) || !isStatic(op->getOperand(0).getType()))
      return false;
    std::optional<unsigned> pos = resultPos(op, axis);
    if (!pos)
      return false;
    Value expandedValue =
        resultIsExpanded ? op->getResult(0) : op->getOperand(0);
    std::optional<unsigned> expandedPos =
        resultIsExpanded ? pos : tracker.getAxisPosition(expandedValue, axis);
    if (!expandedPos)
      return false;
    std::optional<unsigned> group = findGroup(groups, *expandedPos);
    if (!group || (!resultIsExpanded && *group != *pos))
      return false;
    ArrayRef<int64_t> shape =
        cast<RankedTensorType>(expandedValue.getType()).getShape();
    for (int64_t member : groups[*group])
      if (member != static_cast<int64_t>(*expandedPos) && shape[member] != 1)
        return false;
    return true;
  }

  /// The inserted axis becomes a new singleton group at the position its
  /// consumer requires, so every existing group, including one that combines
  /// another axis with other dims, stays intact (Req 7.4). An expand_shape
  /// additionally needs that position on a group boundary (canInsertAt).
  bool canInsertReshape(Operation *op, Axis axis) const {
    return hasStaticResults(op) && table.getAxisSize(axis).has_value();
  }

  /// Rebuilt reassociation: walk the collapsed side's labels; an old group
  /// keeps its expanded dims relabelled, an inserted axis is a singleton.
  SmallVector<ReassociationIndices>
  rebuildGroups(ArrayRef<ReassociationIndices> groups,
                ArrayRef<int64_t> collapsedLabels,
                ArrayRef<int64_t> expandedLabels) const {
    SmallVector<ReassociationIndices> result;
    for (int64_t label : collapsedLabels) {
      ReassociationIndices group;
      if (label < 0) {
        group.push_back(indexOf(expandedLabels, label));
      } else {
        for (int64_t dim : groups[label])
          group.push_back(indexOf(expandedLabels, dim));
      }
      result.push_back(group);
    }
    // A group whose axis was stripped leaves its unit members behind with no
    // collapsed dim; they join the preceding group, or the following one at
    // the front.
    int64_t next = 0;
    SmallVector<ReassociationIndices> merged;
    ReassociationIndices leading;
    for (ReassociationIndices &group : result) {
      ReassociationIndices orphans;
      while (!group.empty() && next < group.front())
        orphans.push_back(next++);
      if (merged.empty())
        llvm::append_range(leading, orphans);
      else
        llvm::append_range(merged.back(), orphans);
      for (int64_t dim : group)
        if (dim != next++)
          llvm_unreachable("rebuilt reassociation is not contiguous");
      merged.push_back(group);
    }
    ReassociationIndices trailing;
    while (next < static_cast<int64_t>(expandedLabels.size()))
      trailing.push_back(next++);
    if (!merged.empty()) {
      merged.front().insert(merged.front().begin(), leading.begin(),
                            leading.end());
      llvm::append_range(merged.back(), trailing);
    }
    return merged;
  }

  /// Old collapsed-side layout derived from the expanded-side one.
  static AxisLayout collapsedLayout(ArrayRef<ReassociationIndices> groups,
                                    AxisLayout expanded) {
    AxisLayout layout;
    for (Axis axis : kAxes)
      if (std::optional<unsigned> pos = expanded.get(axis))
        layout.set(axis, findGroup(groups, *pos));
    return layout;
  }

  /// Old expanded-side layout derived from the collapsed-side one.
  /// A stripped axis in a group with unit members (canStripReshape) sits on
  /// the group's one non-unit member, or on its only member.
  static AxisLayout expandedLayout(ArrayRef<ReassociationIndices> groups,
                                   AxisLayout collapsed,
                                   ArrayRef<int64_t> expandedShape) {
    AxisLayout layout;
    for (Axis axis : kAxes)
      if (std::optional<unsigned> pos = collapsed.get(axis)) {
        const ReassociationIndices &group = groups[*pos];
        std::optional<int64_t> carrier;
        if (group.size() == 1)
          carrier = group.front();
        for (int64_t member : group)
          if (group.size() != 1 && expandedShape[member] != 1) {
            if (carrier)
              llvm_unreachable("rebuild of a non-unit grouped reshape axis");
            carrier = member;
          }
        if (!carrier)
          llvm_unreachable("grouped reshape axis has no carrier dim");
        layout.set(axis, static_cast<unsigned>(*carrier));
      }
    return layout;
  }

  /// A kept axis that a reshape groups with another dim (Req 6.8) does not
  /// move: it is treated as an ordinary dim of its group, so only the other
  /// axis is relabelled. A grouped axis stripped from both sides stays in the
  /// layouts (canStripReshape); inserting a grouped axis has no rule.
  void dropGroupedKeptAxes(Operation *op, AxisLayout &oldResult,
                           AxisLayout &source, AxisLayout &target) const {
    for (Axis axis : kAxes) {
      if (!tracker.isGrouped(op, axis) || !oldResult.get(axis))
        continue;
      if (!target.get(axis) && !source.get(axis))
        continue;
      if (!target.get(axis) || !source.get(axis))
        llvm_unreachable("rebuild moves a grouped reshape axis");
      oldResult.set(axis, std::nullopt);
      source.set(axis, std::nullopt);
      target.set(axis, std::nullopt);
    }
  }

  Operation *rebuild(OpBuilder &builder, tensor::ExpandShapeOp expand,
                     ArrayRef<Value> newOperands,
                     ArrayRef<AxisLayout> operandLayouts,
                     AxisLayout target) const {
    SmallVector<ReassociationIndices> groups = expand.getReassociationIndices();
    AxisLayout oldResult = AxisLayout::of(expand.getResult(), tracker);
    AxisLayout sourceLayout = operandLayouts[0];
    dropGroupedKeptAxes(expand, oldResult, sourceLayout, target);
    AxisLayout oldSource = collapsedLayout(groups, oldResult);
    auto resultType = expand.getResultType();
    SmallVector<int64_t> sourceLabels =
        relabel(expand.getSrcType().getRank(), oldSource, sourceLayout);
    SmallVector<int64_t> resultLabels =
        relabel(resultType.getRank(), oldResult, target);
    SmallVector<ReassociationIndices> newGroups =
        rebuildGroups(groups, sourceLabels, resultLabels);
    auto rebuilt = builder.create<tensor::ExpandShapeOp>(
        expand.getLoc(), relabelType(resultType, resultLabels), newOperands[0],
        newGroups);
    rebuilt->setDiscardableAttrs(expand->getDiscardableAttrDictionary());
    return rebuilt;
  }

  Operation *rebuild(OpBuilder &builder, tensor::CollapseShapeOp collapse,
                     ArrayRef<Value> newOperands,
                     ArrayRef<AxisLayout> operandLayouts,
                     AxisLayout target) const {
    SmallVector<ReassociationIndices> groups =
        collapse.getReassociationIndices();
    AxisLayout oldResult = AxisLayout::of(collapse.getResult(), tracker);
    AxisLayout sourceLayout = operandLayouts[0];
    dropGroupedKeptAxes(collapse, oldResult, sourceLayout, target);
    AxisLayout oldSource =
        expandedLayout(groups, oldResult, collapse.getSrcType().getShape());
    auto resultType = collapse.getResultType();
    SmallVector<int64_t> sourceLabels =
        relabel(collapse.getSrcType().getRank(), oldSource, sourceLayout);
    SmallVector<int64_t> resultLabels =
        relabel(resultType.getRank(), oldResult, target);
    SmallVector<ReassociationIndices> newGroups =
        rebuildGroups(groups, resultLabels, sourceLabels);
    auto rebuilt = builder.create<tensor::CollapseShapeOp>(
        collapse.getLoc(), relabelType(resultType, resultLabels),
        newOperands[0], newGroups);
    rebuilt->setDiscardableAttrs(collapse->getDiscardableAttrDictionary());
    return rebuilt;
  }

  //===--- tensor.reshape (static shapes) ---------------------------------===//

  bool isStaticReshape(tensor::ReshapeOp reshape) const {
    return hasStaticResults(reshape) && isStatic(reshape.getSource().getType());
  }

  /// The reshape analogue of a singleton group: `axis` sits at `sourceAt` /
  /// `resultAt` with equal prefix extents on both sides.
  static bool isBoundary(ArrayRef<int64_t> source, unsigned sourceAt,
                         ArrayRef<int64_t> result, unsigned resultAt) {
    return sourceAt <= source.size() && resultAt <= result.size() &&
           product(source.take_front(sourceAt)) ==
               product(result.take_front(resultAt));
  }

  bool canStrip(tensor::ReshapeOp reshape, Axis axis) const {
    if (!isStaticReshape(reshape))
      return false;
    std::optional<unsigned> resultAt = resultPos(reshape, axis);
    std::optional<unsigned> sourceAt =
        tracker.getAxisPosition(reshape.getSource(), axis);
    if (!resultAt || !sourceAt)
      return false;
    ArrayRef<int64_t> source = reshape.getSource().getType().getShape();
    ArrayRef<int64_t> result = reshape.getResult().getType().getShape();
    return source[*sourceAt] == result[*resultAt] &&
           isBoundary(source, *sourceAt, result, *resultAt);
  }

  bool canInsert(tensor::ReshapeOp reshape, Axis axis) const {
    // The shared boundary at the derived position is checked by
    // canInsertAt.
    return isStaticReshape(reshape) && table.getAxisSize(axis).has_value();
  }

  Operation *rebuild(OpBuilder &builder, tensor::ReshapeOp reshape,
                     ArrayRef<Value> newOperands,
                     ArrayRef<AxisLayout> operandLayouts,
                     AxisLayout target) const {
    AxisLayout oldSource = AxisLayout::of(reshape.getSource(), tracker);
    AxisLayout oldResult = AxisLayout::of(reshape.getResult(), tracker);
    RankedTensorType sourceType =
        relabelType(reshape.getSource().getType(),
                    relabel(reshape.getSource().getType().getRank(), oldSource,
                            operandLayouts[0]));
    RankedTensorType resultType = relabelType(
        reshape.getResult().getType(),
        relabel(reshape.getResult().getType().getRank(), oldResult, target));
    if (newOperands[0].getType() != sourceType ||
        product(sourceType.getShape()) != product(resultType.getShape()))
      llvm_unreachable("reshape operand does not match the rebuilt source");
    for (Axis axis : kAxes) {
      std::optional<unsigned> from = operandLayouts[0].get(axis);
      std::optional<unsigned> to = target.get(axis);
      if (from.has_value() != to.has_value() ||
          (from && !isBoundary(sourceType.getShape(), *from,
                               resultType.getShape(), *to)))
        llvm_unreachable("reshape axis does not sit on a shared boundary");
    }
    auto shapeType = cast<RankedTensorType>(reshape.getShape().getType());
    auto newShapeType = RankedTensorType::get({resultType.getRank()},
                                              shapeType.getElementType());
    SmallVector<APInt> extents;
    unsigned width = shapeType.getElementType().getIntOrFloatBitWidth();
    for (int64_t size : resultType.getShape())
      extents.push_back(APInt(width, size, /*isSigned=*/true));
    Value shape = builder.create<arith::ConstantOp>(
        reshape.getLoc(), DenseIntElementsAttr::get(newShapeType, extents));
    auto rebuilt = builder.create<tensor::ReshapeOp>(
        reshape.getLoc(), resultType, newOperands[0], shape);
    rebuilt->setDiscardableAttrs(reshape->getDiscardableAttrDictionary());
    return rebuilt;
  }

  //===--- tensor.extract_slice / tensor.insert_slice ---------------------===//
  //
  // A slice has a full side (extract source, insert dest), whose dims index
  // offsets/sizes/strides, and a possibly rank-reduced side (extract result,
  // insert source). The axis must keep a full-extent unit-stride entry on a
  // non-dropped full dim.

  /// Full-side dim -> reduced-side dim, or -1 for a dropped unit dim.
  static SmallVector<int64_t> fullToReduced(const llvm::SmallBitVector &dropped,
                                            unsigned fullRank) {
    SmallVector<int64_t> map;
    int64_t next = 0;
    for (unsigned i = 0; i < fullRank; ++i)
      map.push_back(dropped.test(i) ? -1 : next++);
    return map;
  }

  /// Old full-side layout derived from the reduced-side one.
  static AxisLayout fullLayout(ArrayRef<int64_t> toReduced,
                               AxisLayout reduced) {
    AxisLayout layout;
    for (Axis axis : kAxes)
      if (std::optional<unsigned> pos = reduced.get(axis))
        layout.set(axis, indexOf(toReduced, *pos));
    return layout;
  }

  /// Old reduced-side layout derived from the full-side one.
  static AxisLayout reducedLayout(ArrayRef<int64_t> toReduced,
                                  AxisLayout full) {
    AxisLayout layout;
    for (Axis axis : kAxes)
      if (std::optional<unsigned> pos = full.get(axis)) {
        if (toReduced[*pos] < 0)
          llvm_unreachable("tracked axis on a dropped slice dim");
        layout.set(axis, static_cast<unsigned>(toReduced[*pos]));
      }
    return layout;
  }

  static bool isFullEntry(OffsetSizeAndStrideOpInterface slice, unsigned dim,
                          int64_t extent) {
    return isConstantIndex(slice.getMixedOffsets()[dim], 0) &&
           isConstantIndex(slice.getMixedSizes()[dim], extent) &&
           isConstantIndex(slice.getMixedStrides()[dim], 1);
  }

  /// Strip test shared by both slice kinds; `fullPos` is the axis's full-side
  /// dim.
  static bool canStripSlice(OffsetSizeAndStrideOpInterface slice,
                            RankedTensorType fullType,
                            std::optional<unsigned> fullPos) {
    return fullPos &&
           isFullEntry(slice, *fullPos, fullType.getDimSize(*fullPos));
  }

  bool isStaticExtract(tensor::ExtractSliceOp slice) const {
    return hasStaticResults(slice) && isStatic(slice.getSourceType());
  }

  bool canStrip(tensor::ExtractSliceOp slice, Axis axis) const {
    if (!isStaticExtract(slice))
      return false;
    std::optional<unsigned> pos = resultPos(slice, axis);
    if (!pos)
      return false;
    SmallVector<int64_t> toReduced =
        fullToReduced(slice.getDroppedDims(), slice.getSourceType().getRank());
    return canStripSlice(slice, slice.getSourceType(),
                         indexOf(toReduced, *pos));
  }

  bool canInsert(tensor::ExtractSliceOp slice, Axis axis) const {
    return isStaticExtract(slice) && table.getAxisSize(axis);
  }

  bool isStaticInsert(tensor::InsertSliceOp slice) const {
    return hasStaticResults(slice) && isStatic(slice.getSourceType());
  }

  bool canStrip(tensor::InsertSliceOp slice, Axis axis) const {
    if (!isStaticInsert(slice))
      return false;
    std::optional<unsigned> pos = resultPos(slice, axis);
    SmallVector<int64_t> toReduced =
        fullToReduced(slice.getDroppedDims(), slice.getDestType().getRank());
    return pos && toReduced[*pos] >= 0 &&
           canStripSlice(slice, slice.getDestType(), *pos);
  }

  bool canInsert(tensor::InsertSliceOp slice, Axis axis) const {
    return isStaticInsert(slice) && table.getAxisSize(axis);
  }

  /// Offsets, sizes and strides relabelled over the full side; inserted axes
  /// get (0, extent, 1). Checks that the reduced side's labels follow.
  void rebuildEntries(OpBuilder &builder, OffsetSizeAndStrideOpInterface slice,
                      ArrayRef<int64_t> fullLabels,
                      ArrayRef<int64_t> reducedLabels,
                      ArrayRef<int64_t> toReduced, const IRMapping &mapping,
                      SmallVectorImpl<OpFoldResult> &offsets,
                      SmallVectorImpl<OpFoldResult> &sizes,
                      SmallVectorImpl<OpFoldResult> &strides) const {
    auto remap = [&](OpFoldResult value) -> OpFoldResult {
      if (auto v = dyn_cast<Value>(value))
        return mapping.lookupOrDefault(v);
      return value;
    };
    SmallVector<OpFoldResult> oldOffsets = slice.getMixedOffsets();
    SmallVector<OpFoldResult> oldSizes = slice.getMixedSizes();
    SmallVector<OpFoldResult> oldStrides = slice.getMixedStrides();
    SmallVector<int64_t> expectedReduced;
    for (int64_t label : fullLabels) {
      if (label >= 0) {
        offsets.push_back(remap(oldOffsets[label]));
        sizes.push_back(remap(oldSizes[label]));
        strides.push_back(remap(oldStrides[label]));
        if (toReduced[label] >= 0)
          expectedReduced.push_back(toReduced[label]);
        continue;
      }
      std::optional<int64_t> extent = table.getAxisSize(markerAxis(label));
      if (!extent)
        llvm_unreachable("inserted axis has no static extent");
      offsets.push_back(builder.getIndexAttr(0));
      sizes.push_back(builder.getIndexAttr(*extent));
      strides.push_back(builder.getIndexAttr(1));
      expectedReduced.push_back(label);
    }
    if (ArrayRef<int64_t>(expectedReduced) != reducedLabels)
      llvm_unreachable("slice full and reduced layouts disagree");
  }

  Operation *rebuild(OpBuilder &builder, tensor::ExtractSliceOp slice,
                     ArrayRef<Value> newOperands,
                     ArrayRef<AxisLayout> operandLayouts,
                     AxisLayout target) const {
    SmallVector<int64_t> toReduced =
        fullToReduced(slice.getDroppedDims(), slice.getSourceType().getRank());
    AxisLayout oldResult = AxisLayout::of(slice.getResult(), tracker);
    AxisLayout oldSource = fullLayout(toReduced, oldResult);
    SmallVector<int64_t> fullLabels =
        relabel(slice.getSourceType().getRank(), oldSource, operandLayouts[0]);
    SmallVector<int64_t> reducedLabels =
        relabel(slice.getResultType().getRank(), oldResult, target);
    IRMapping mapping;
    mapping.map(slice->getOperands(), newOperands);
    SmallVector<OpFoldResult> offsets, sizes, strides;
    rebuildEntries(builder, slice, fullLabels, reducedLabels, toReduced,
                   mapping, offsets, sizes, strides);
    auto rebuilt = builder.create<tensor::ExtractSliceOp>(
        slice.getLoc(), relabelType(slice.getResultType(), reducedLabels),
        newOperands[0], offsets, sizes, strides);
    rebuilt->setDiscardableAttrs(slice->getDiscardableAttrDictionary());
    return rebuilt;
  }

  Operation *rebuild(OpBuilder &builder, tensor::InsertSliceOp slice,
                     ArrayRef<Value> newOperands,
                     ArrayRef<AxisLayout> operandLayouts,
                     AxisLayout target) const {
    if (!(operandLayouts[1] == target))
      llvm_unreachable("insert_slice dest and target layouts disagree");
    SmallVector<int64_t> toReduced =
        fullToReduced(slice.getDroppedDims(), slice.getDestType().getRank());
    AxisLayout oldResult = AxisLayout::of(slice.getResult(), tracker);
    AxisLayout oldSource = reducedLayout(toReduced, oldResult);
    SmallVector<int64_t> fullLabels =
        relabel(slice.getDestType().getRank(), oldResult, target);
    SmallVector<int64_t> reducedLabels =
        relabel(slice.getSourceType().getRank(), oldSource, operandLayouts[0]);
    IRMapping mapping;
    mapping.map(slice->getOperands(), newOperands);
    SmallVector<OpFoldResult> offsets, sizes, strides;
    rebuildEntries(builder, slice, fullLabels, reducedLabels, toReduced,
                   mapping, offsets, sizes, strides);
    auto rebuilt = builder.create<tensor::InsertSliceOp>(
        slice.getLoc(), newOperands[0], newOperands[1], offsets, sizes,
        strides);
    rebuilt->setDiscardableAttrs(slice->getDiscardableAttrDictionary());
    return rebuilt;
  }

  //===--- memref.reinterpret_cast ----------------------------------------===//
  //
  // The axis is folded into the view as a dimension of stride 0 and the
  // axis's extent, at the position Req 6.8 gives, with the offset unchanged
  // (Req 7.9). Stride 0 makes every index along the axis read the same
  // memory, so the lift costs no extra traffic. Offsets, sizes and strides
  // are handled as OpFoldResults, so a static attribute and a dynamic SSA
  // value are treated alike; the weight chain's offset is dynamic.

  bool canStrip(memref::ReinterpretCastOp view, Axis axis) const {
    std::optional<unsigned> pos = resultPos(view, axis);
    std::optional<int64_t> extent = table.getAxisSize(axis);
    if (!pos || !extent)
      return false;
    // Only a dimension this pass could have inserted is removable: one that
    // reads no memory of its own.
    return isConstantIndex(view.getMixedStrides()[*pos], 0) &&
           isConstantIndex(view.getMixedSizes()[*pos], *extent);
  }

  bool canInsert(memref::ReinterpretCastOp view, Axis axis) const {
    return table.getAxisSize(axis).has_value();
  }

  Operation *rebuild(OpBuilder &builder, memref::ReinterpretCastOp view,
                     ArrayRef<Value> newOperands, AxisLayout target) const {
    AxisLayout oldLayout = AxisLayout::of(view.getResult(), tracker);
    SmallVector<int64_t> labels =
        relabel(view.getType().getRank(), oldLayout, target);
    IRMapping mapping;
    mapping.map(view->getOperands(), newOperands);
    SmallVector<OpFoldResult> oldSizes = view.getMixedSizes();
    SmallVector<OpFoldResult> oldStrides = view.getMixedStrides();
    SmallVector<OpFoldResult> sizes, strides;
    for (int64_t label : labels) {
      if (label >= 0) {
        sizes.push_back(remapFoldResult(oldSizes[label], mapping));
        strides.push_back(remapFoldResult(oldStrides[label], mapping));
        continue;
      }
      // An axis the view already carried is relabelled, not inserted: it
      // keeps the size and stride of its old position, which is where the
      // real data lies. Only a newly inserted axis gets stride 0.
      Axis axis = markerAxis(label);
      if (std::optional<unsigned> oldPosition = oldLayout.get(axis)) {
        sizes.push_back(remapFoldResult(oldSizes[*oldPosition], mapping));
        strides.push_back(remapFoldResult(oldStrides[*oldPosition], mapping));
        continue;
      }
      std::optional<int64_t> extent = table.getAxisSize(axis);
      if (!extent)
        llvm_unreachable("inserted axis has no static extent");
      sizes.push_back(builder.getIndexAttr(*extent));
      strides.push_back(builder.getIndexAttr(0));
    }
    OpFoldResult offset =
        remapFoldResult(view.getMixedOffsets().front(), mapping);
    Operation *rebuilt;
    if (oldLayout == target) {
      // No layout change: keep the original result type, so a reader kept in
      // place, such as a load chain's memref.copy, still
      // verifies against it.
      rebuilt = builder.create<memref::ReinterpretCastOp>(
          view.getLoc(), view.getType(), newOperands[0], offset, sizes,
          strides);
    } else {
      rebuilt = builder.create<memref::ReinterpretCastOp>(
          view.getLoc(), newOperands[0], offset, sizes, strides);
    }
    rebuilt->setDiscardableAttrs(view->getDiscardableAttrDictionary());
    return rebuilt;
  }

  //===--- bufferization.to_tensor ----------------------------------------===//

  bool canStrip(bufferization::ToTensorOp toTensor, Axis axis) const {
    return hasStaticResults(toTensor) && resultPos(toTensor, axis).has_value();
  }

  bool canInsert(bufferization::ToTensorOp toTensor, Axis axis) const {
    // Req 7.10: the result follows the rebuilt memref operand, and only a
    // memref.reinterpret_cast has an insert rule, so a to_tensor of anything
    // else, such as the alloc of a load chain that keeps its copy, cannot
    // take the axis.
    return hasStaticResults(toTensor) && table.getAxisSize(axis).has_value() &&
           isa_and_present<memref::ReinterpretCastOp>(
               toTensor.getBuffer().getDefiningOp());
  }

  Operation *rebuild(OpBuilder &builder, bufferization::ToTensorOp toTensor,
                     ArrayRef<Value> newOperands, AxisLayout target) const {
    // No layout change, as for a lookup table read through a dynamically
    // shaped view (Req 7.12): the result type stays as it is.
    if (AxisLayout::of(toTensor.getResult(), tracker) == target) {
      auto rebuilt = builder.create<bufferization::ToTensorOp>(
          toTensor.getLoc(), toTensor.getResult().getType(), newOperands[0],
          toTensor.getRestrict(), toTensor.getWritable());
      rebuilt->setDiscardableAttrs(toTensor->getDiscardableAttrDictionary());
      return rebuilt;
    }
    // Req 7.10: the result's rank and shape follow the rebuilt memref
    // operand.
    auto bufferType = dyn_cast<MemRefType>(newOperands[0].getType());
    if (!bufferType || !bufferType.hasStaticShape())
      llvm_unreachable("to_tensor operand is not a static ranked memref");
    auto oldType = cast<RankedTensorType>(toTensor.getResult().getType());
    auto resultType = RankedTensorType::get(bufferType.getShape(),
                                            bufferType.getElementType(),
                                            oldType.getEncoding());
    AxisLayout oldLayout = AxisLayout::of(toTensor.getResult(), tracker);
    if (resultType !=
        relabelType(oldType, relabel(oldType.getRank(), oldLayout, target)))
      llvm_unreachable("to_tensor operand does not match the rebuilt result");
    auto rebuilt = builder.create<bufferization::ToTensorOp>(
        toTensor.getLoc(), resultType, newOperands[0], toTensor.getRestrict(),
        toTensor.getWritable());
    rebuilt->setDiscardableAttrs(toTensor->getDiscardableAttrDictionary());
    return rebuilt;
  }

  //===--- Operand positions (Req 6.8) ------------------------------------===//
  //
  // The operand of a non-generic op carries exactly the axes of the result.
  // A kept axis stays at the operand position it has in the input, a
  // stripped one is removed, and an inserted one is placed in the operand
  // right before the operand dim corresponding to the result dim that
  // follows it, or right after the one corresponding to the result dim that
  // precedes it. `toSource` maps an old result dim to the first and last old
  // operand dims it corresponds to, so an inserted axis never lands inside a
  // reassociation group.

  using DimRange = std::pair<int64_t, int64_t>;

  AxisLayout correspondingLayout(
      Operation *op, Value operand, AxisLayout target,
      function_ref<std::optional<DimRange>(int64_t)> toSource) const {
    AxisLayout oldResult = AxisLayout::of(op->getResult(0), tracker);
    AxisLayout oldSource = AxisLayout::of(operand, tracker);
    unsigned resultRank =
        cast<ShapedType>(op->getResult(0).getType()).getRank();
    unsigned sourceRank = cast<ShapedType>(operand.getType()).getRank();
    SmallVector<int64_t> resultLabels = relabel(resultRank, oldResult, target);
    SmallVector<int64_t> sourceLabels;
    for (unsigned i = 0; i < sourceRank; ++i)
      sourceLabels.push_back(i);
    for (Axis axis : kAxes)
      if (std::optional<unsigned> pos = oldSource.get(axis);
          pos && !target.get(axis))
        sourceLabels.erase(
            llvm::find(sourceLabels, static_cast<int64_t>(*pos)));
    auto mapLabel = [&](int64_t label) -> std::optional<DimRange> {
      if (label >= 0)
        return toSource(label);
      Axis axis = markerAxis(label);
      if (std::optional<unsigned> old = oldResult.get(axis))
        return toSource(*old);
      return DimRange(label, label);
    };
    auto placed = [&](int64_t label) {
      return llvm::is_contained(sourceLabels, label);
    };
    SmallVector<std::pair<unsigned, Axis>> inserted;
    for (Axis axis : kAxes)
      if (target.get(axis) && !oldResult.get(axis))
        inserted.push_back({*target.get(axis), axis});
    llvm::sort(inserted, [](auto x, auto y) { return x.first < y.first; });
    for (auto [position, axis] : inserted) {
      unsigned at = 0;
      bool found = false;
      for (int64_t label :
           ArrayRef<int64_t>(resultLabels).drop_front(position + 1))
        if (std::optional<DimRange> source = mapLabel(label);
            source && placed(source->first)) {
          at = indexOf(sourceLabels, source->first);
          found = true;
          break;
        }
      if (!found)
        for (int64_t label : llvm::reverse(
                 ArrayRef<int64_t>(resultLabels).take_front(position)))
          if (std::optional<DimRange> source = mapLabel(label);
              source && placed(source->second)) {
            at = indexOf(sourceLabels, source->second) + 1;
            break;
          }
      sourceLabels.insert(sourceLabels.begin() + at, axisMarker(axis));
    }
    AxisLayout layout;
    for (Axis axis : kAxes) {
      if (!target.get(axis))
        continue;
      std::optional<unsigned> old = oldSource.get(axis);
      if (oldResult.get(axis) && !old)
        llvm_unreachable("kept axis missing from the operand");
      layout.set(axis, indexOf(sourceLabels, old ? static_cast<int64_t>(*old)
                                                 : axisMarker(axis)));
    }
    return layout;
  }

  /// linalg.transpose: a kept axis is read through the permutation, and an
  /// inserted axis takes the same position in the input as in the result,
  /// the rebuilt permutation mapping it to itself.
  AxisLayout transposeLayout(linalg::TransposeOp transpose,
                             AxisLayout target) const {
    AxisLayout oldResult = AxisLayout::of(transpose->getResult(0), tracker);
    AxisLayout oldInput = AxisLayout::of(transpose.getInput(), tracker);
    SmallVector<int64_t> labels;
    for (unsigned i = 0, e = transpose.getPermutation().size(); i < e; ++i)
      labels.push_back(i);
    for (Axis axis : kAxes)
      if (std::optional<unsigned> pos = oldInput.get(axis);
          pos && !target.get(axis))
        labels.erase(llvm::find(labels, static_cast<int64_t>(*pos)));
    SmallVector<std::pair<unsigned, Axis>> inserted;
    for (Axis axis : kAxes)
      if (target.get(axis) && !oldResult.get(axis))
        inserted.push_back({*target.get(axis), axis});
    llvm::sort(inserted, [](auto x, auto y) { return x.first < y.first; });
    for (auto [position, axis] : inserted)
      labels.insert(labels.begin() + std::min<size_t>(position, labels.size()),
                    axisMarker(axis));
    AxisLayout layout;
    for (Axis axis : kAxes) {
      if (!target.get(axis))
        continue;
      std::optional<unsigned> old = oldInput.get(axis);
      if (oldResult.get(axis) && !old)
        llvm_unreachable("kept axis missing from the transpose input");
      layout.set(axis, indexOf(labels, old ? static_cast<int64_t>(*old)
                                           : axisMarker(axis)));
    }
    return layout;
  }

  /// tensor.reshape: an inserted axis goes to the first source position
  /// whose prefix extent equals the result's prefix extent ahead of it, the
  /// reshape analogue of a new singleton group. Returns nullopt when no such
  /// position exists.
  std::optional<AxisLayout> reshapeLayout(tensor::ReshapeOp reshape,
                                          AxisLayout target) const {
    AxisLayout oldResult = AxisLayout::of(reshape.getResult(), tracker);
    AxisLayout oldSource = AxisLayout::of(reshape.getSource(), tracker);
    auto sourceType = cast<RankedTensorType>(reshape.getSource().getType());
    RankedTensorType resultType = relabelType(
        reshape.getResult().getType(),
        relabel(reshape.getResult().getType().getRank(), oldResult, target));
    AxisLayout sourceLayout =
        AxisLayout::kept(reshape.getSource(), target.get(Axis::M).has_value(),
                         target.get(Axis::N).has_value(), tracker);
    SmallVector<std::pair<unsigned, Axis>> inserted;
    for (Axis axis : kAxes)
      if (target.get(axis) && !oldResult.get(axis))
        inserted.push_back({*target.get(axis), axis});
    llvm::sort(inserted, [](auto x, auto y) { return x.first < y.first; });
    for (auto [position, axis] : inserted) {
      // Current source shape: old dims minus stripped, plus placed axes.
      RankedTensorType current = relabelType(
          sourceType, relabel(sourceType.getRank(), oldSource, sourceLayout));
      int64_t wanted = product(resultType.getShape().take_front(position));
      std::optional<unsigned> at;
      for (unsigned q = 0; q <= current.getRank(); ++q)
        if (product(current.getShape().take_front(q)) == wanted) {
          at = q;
          break;
        }
      if (!at)
        return std::nullopt;
      sourceLayout = sourceLayout.withInserted(axis, *at);
    }
    return sourceLayout;
  }

  /// Whether `axis`, absent from `op`'s result, can be inserted at
  /// `position` of it: a reshape needs a group boundary there.
  bool canInsertAt(Operation *op, Axis axis, unsigned position) const {
    auto resultType = dyn_cast<ShapedType>(op->getResult(0).getType());
    if (!resultType || !resultType.hasRank() || position > resultType.getRank())
      return false;
    AxisLayout oldResult = AxisLayout::of(op->getResult(0), tracker);
    AxisLayout target = oldResult.withInserted(axis, position);
    if (auto expand = dyn_cast<tensor::ExpandShapeOp>(op)) {
      // The result dim after the inserted axis must open a group.
      SmallVector<ReassociationIndices> groups =
          expand.getReassociationIndices();
      SmallVector<int64_t> labels =
          relabel(expand.getResultType().getRank(), oldResult, target);
      for (int64_t label : ArrayRef<int64_t>(labels).drop_front(position + 1)) {
        int64_t dim = label;
        if (label < 0) {
          std::optional<unsigned> old = oldResult.get(markerAxis(label));
          if (!old)
            continue;
          dim = *old;
        }
        std::optional<unsigned> group = findGroup(groups, dim);
        return group && groups[*group].front() == dim;
      }
      return true;
    }
    if (auto reshape = dyn_cast<tensor::ReshapeOp>(op))
      return reshapeLayout(reshape, target).has_value();
    return true;
  }

  std::optional<AxisLayout> operandLayout(Operation *op, unsigned number,
                                          AxisLayout target,
                                          ArrayRef<AffineMap> maps) const {
    Value operand = op->getOperand(number);
    if (!isa<ShapedType>(operand.getType()))
      return std::nullopt;
    auto dps = dyn_cast<DestinationStyleOpInterface>(op);
    if (dps && dps.isDpsInit(&op->getOpOperand(number)))
      return target;
    return llvm::TypeSwitch<Operation *, std::optional<AxisLayout>>(op)
        .Case<linalg::GenericOp>(
            [&](auto generic) -> std::optional<AxisLayout> {
              if (!isa<RankedTensorType>(operand.getType()))
                return std::nullopt;
              AffineMap map = maps.empty()
                                  ? generic.getIndexingMapsArray()[number]
                                  : maps[number];
              return operandLayout(generic, map, target);
            })
        .Case<linalg::ReduceOp>([&](auto reduce) -> std::optional<AxisLayout> {
          return correspondingLayout(
              reduce, operand, target,
              [&](int64_t dim) -> std::optional<DimRange> {
                std::optional<unsigned> input = keptInputDim(reduce, dim);
                if (!input)
                  return std::nullopt;
                return DimRange(*input, *input);
              });
        })
        .Case<linalg::TransposeOp>(
            [&](auto transpose) -> std::optional<AxisLayout> {
              return transposeLayout(transpose, target);
            })
        .Case<tensor::ExpandShapeOp>(
            [&](auto expand) -> std::optional<AxisLayout> {
              SmallVector<ReassociationIndices> groups =
                  expand.getReassociationIndices();
              return correspondingLayout(
                  expand, operand, target,
                  [&](int64_t dim) -> std::optional<DimRange> {
                    std::optional<unsigned> group = findGroup(groups, dim);
                    if (!group)
                      return std::nullopt;
                    return DimRange(*group, *group);
                  });
            })
        .Case<tensor::CollapseShapeOp>(
            [&](auto collapse) -> std::optional<AxisLayout> {
              SmallVector<ReassociationIndices> groups =
                  collapse.getReassociationIndices();
              return correspondingLayout(
                  collapse, operand, target,
                  [&](int64_t dim) -> std::optional<DimRange> {
                    return DimRange(groups[dim].front(), groups[dim].back());
                  });
            })
        .Case<tensor::ReshapeOp>(
            [&](auto reshape) -> std::optional<AxisLayout> {
              if (operand != reshape.getSource())
                return std::nullopt;
              std::optional<AxisLayout> layout = reshapeLayout(reshape, target);
              if (!layout)
                llvm_unreachable("reshape axis without a shared boundary");
              return layout;
            })
        .Case<tensor::ExtractSliceOp>(
            [&](auto slice) -> std::optional<AxisLayout> {
              if (operand != slice.getSource())
                return std::nullopt;
              SmallVector<int64_t> toReduced = fullToReduced(
                  slice.getDroppedDims(), slice.getSourceType().getRank());
              return correspondingLayout(
                  slice, operand, target,
                  [&](int64_t dim) -> std::optional<DimRange> {
                    int64_t full = indexOf(toReduced, dim);
                    return DimRange(full, full);
                  });
            })
        .Case<tensor::InsertSliceOp>(
            [&](auto slice) -> std::optional<AxisLayout> {
              if (operand != slice.getSource())
                return std::nullopt;
              SmallVector<int64_t> toReduced = fullToReduced(
                  slice.getDroppedDims(), slice.getDestType().getRank());
              return correspondingLayout(
                  slice, operand, target,
                  [&](int64_t dim) -> std::optional<DimRange> {
                    if (toReduced[dim] < 0)
                      return std::nullopt;
                    return DimRange(toReduced[dim], toReduced[dim]);
                  });
            })
        // The memref operand reads the axis where the result has it; the
        // reinterpret_cast rule then inserts it as a stride-0 dim.
        .Case<bufferization::ToTensorOp>(
            [&](auto) -> std::optional<AxisLayout> { return target; })
        // A view's source is the base memref and a fill's input a scalar:
        // neither carries a tracked axis.
        .Case<memref::ReinterpretCastOp, tensor::EmptyOp, linalg::FillOp>(
            [&](auto) -> std::optional<AxisLayout> { return std::nullopt; })
        // A kind without a rule never changes layout (R2 / R3), so its
        // operands keep their input layout.
        .Default([&](Operation *) -> std::optional<AxisLayout> {
          if (!isa<RankedTensorType, MemRefType>(operand.getType()))
            return std::nullopt;
          return AxisLayout::of(operand, tracker);
        });
  }

  //===--- Kept axis ------------------------------------------------------===//

  /// For an op that already carries `axis` (Req 6.8): the axis stays at its
  /// existing position, which every supported kind can carry. A reshape that
  /// groups the axis with a non-unit dim has no rule (Req 7.6); one whose
  /// other group members are all unit carries it like a singleton group.
  bool keepsAxis(Operation *op, Axis axis) const {
    return llvm::TypeSwitch<Operation *, bool>(op)
        .Case<linalg::GenericOp, linalg::ReduceOp, linalg::TransposeOp,
              tensor::ReshapeOp, tensor::ExtractSliceOp, tensor::InsertSliceOp,
              tensor::EmptyOp, linalg::FillOp, memref::ReinterpretCastOp,
              bufferization::ToTensorOp>([](Operation *) { return true; })
        .Case<tensor::ExpandShapeOp>([&](auto typed) {
          return !tracker.isGrouped(typed, axis) ||
                 hasOnlyUnitPartners(typed, axis,
                                     typed.getReassociationIndices(),
                                     /*resultIsExpanded=*/true);
        })
        .Case<tensor::CollapseShapeOp>([&](auto typed) {
          return !tracker.isGrouped(typed, axis) ||
                 hasOnlyUnitPartners(typed, axis,
                                     typed.getReassociationIndices(),
                                     /*resultIsExpanded=*/false);
        })
        .Default([](Operation *) { return false; });
  }

  //===--- tensor.empty / linalg.fill -------------------------------------===//

  bool canStrip(tensor::EmptyOp empty, Axis axis) const {
    return hasStaticResults(empty) && resultPos(empty, axis);
  }

  bool canInsert(tensor::EmptyOp empty, Axis axis) const {
    return hasStaticResults(empty) && table.getAxisSize(axis).has_value();
  }

  Operation *rebuild(OpBuilder &builder, tensor::EmptyOp empty,
                     AxisLayout target) const {
    AxisLayout oldLayout = AxisLayout::of(empty.getResult(), tracker);
    RankedTensorType type = relabelType(
        empty.getType(), relabel(empty.getType().getRank(), oldLayout, target));
    auto rebuilt = builder.create<tensor::EmptyOp>(
        empty.getLoc(), type.getShape(), type.getElementType());
    rebuilt->setDiscardableAttrs(empty->getDiscardableAttrDictionary());
    return rebuilt;
  }

  bool canStrip(linalg::FillOp fill, Axis axis) const {
    return hasStaticResults(fill) && resultPos(fill, axis);
  }

  bool canInsert(linalg::FillOp fill, Axis axis) const {
    return hasStaticResults(fill) && table.getAxisSize(axis).has_value();
  }

  Operation *rebuild(OpBuilder &builder, linalg::FillOp fill,
                     ArrayRef<Value> newOperands, AxisLayout target) const {
    AxisLayout oldLayout = AxisLayout::of(fill->getResult(0), tracker);
    auto oldType = cast<RankedTensorType>(fill->getResult(0).getType());
    RankedTensorType type =
        relabelType(oldType, relabel(oldType.getRank(), oldLayout, target));
    IRMapping mapping;
    mapping.map(fill->getOperands(), newOperands);
    Operation *rebuilt = builder.clone(*fill, mapping);
    rebuilt->getResult(0).setType(
        cast<linalg::FillOp>(rebuilt).getDpsInitOperand(0)->get().getType());
    if (rebuilt->getResult(0).getType() != type)
      llvm_unreachable("fill init does not match the rebuilt result");
    return rebuilt;
  }

private:
  const RuleTable &table;
  const AxisTracker &tracker;
};

} // namespace

//===----------------------------------------------------------------------===//
// Dispatch
//===----------------------------------------------------------------------===//

bool RuleTable::canStrip(Operation *op, Axis axis) const {
  Rules rules(*this, tracker);
  return llvm::TypeSwitch<Operation *, bool>(op)
      .Case<linalg::GenericOp>(
          [&](auto typed) { return rules.canStrip(typed, axis); })
      .Case<linalg::ReduceOp>(
          [&](auto typed) { return rules.canStrip(typed, axis); })
      .Case<linalg::TransposeOp>(
          [&](auto typed) { return rules.canStrip(typed, axis); })
      .Case<tensor::ExpandShapeOp>([&](auto typed) {
        return rules.canStripReshape(typed, axis,
                                     typed.getReassociationIndices(),
                                     /*resultIsExpanded=*/true);
      })
      .Case<tensor::CollapseShapeOp>([&](auto typed) {
        return rules.canStripReshape(typed, axis,
                                     typed.getReassociationIndices(),
                                     /*resultIsExpanded=*/false);
      })
      .Case<tensor::ReshapeOp>(
          [&](auto typed) { return rules.canStrip(typed, axis); })
      .Case<tensor::ExtractSliceOp>(
          [&](auto typed) { return rules.canStrip(typed, axis); })
      .Case<tensor::InsertSliceOp>(
          [&](auto typed) { return rules.canStrip(typed, axis); })
      .Case<tensor::EmptyOp>(
          [&](auto typed) { return rules.canStrip(typed, axis); })
      .Case<linalg::FillOp>(
          [&](auto typed) { return rules.canStrip(typed, axis); })
      .Case<memref::ReinterpretCastOp>(
          [&](auto typed) { return rules.canStrip(typed, axis); })
      .Case<bufferization::ToTensorOp>(
          [&](auto typed) { return rules.canStrip(typed, axis); })
      // Unsupported kinds: the caller pins the op and raises R2.
      .Default([](Operation *) { return false; });
}

bool RuleTable::canInsert(Operation *op, Axis axis, unsigned position) const {
  Rules rules(*this, tracker);
  // An op that already carries the axis keeps it at its existing position
  // (Req 6.8), independent of canStrip and of the derived position.
  if (op->getNumResults() == 1 &&
      tracker.getAxisPosition(op->getResult(0), axis))
    return rules.keepsAxis(op, axis);
  if (op->getNumResults() == 0 || !rules.canInsertAt(op, axis, position))
    return false;
  return llvm::TypeSwitch<Operation *, bool>(op)
      .Case<linalg::GenericOp>(
          [&](auto typed) { return rules.canInsert(typed, axis); })
      .Case<linalg::ReduceOp>(
          [&](auto typed) { return rules.canInsert(typed, axis); })
      .Case<linalg::TransposeOp>(
          [&](auto typed) { return rules.canInsert(typed, axis); })
      .Case<tensor::ExpandShapeOp>(
          [&](auto typed) { return rules.canInsertReshape(typed, axis); })
      .Case<tensor::CollapseShapeOp>(
          [&](auto typed) { return rules.canInsertReshape(typed, axis); })
      .Case<tensor::ReshapeOp>(
          [&](auto typed) { return rules.canInsert(typed, axis); })
      .Case<tensor::ExtractSliceOp>(
          [&](auto typed) { return rules.canInsert(typed, axis); })
      .Case<tensor::InsertSliceOp>(
          [&](auto typed) { return rules.canInsert(typed, axis); })
      .Case<tensor::EmptyOp>(
          [&](auto typed) { return rules.canInsert(typed, axis); })
      .Case<linalg::FillOp>(
          [&](auto typed) { return rules.canInsert(typed, axis); })
      .Case<memref::ReinterpretCastOp>(
          [&](auto typed) { return rules.canInsert(typed, axis); })
      .Case<bufferization::ToTensorOp>(
          [&](auto typed) { return rules.canInsert(typed, axis); })
      // Unsupported kinds: the caller excludes the op and raises R3.
      .Default([](Operation *) { return false; });
}

std::optional<AxisLayout>
RuleTable::operandLayout(Operation *op, unsigned operandNumber,
                         AxisLayout target, ArrayRef<AffineMap> maps) const {
  return Rules(*this, tracker).operandLayout(op, operandNumber, target, maps);
}

Operation *RuleTable::rebuild(OpBuilder &builder, Operation *op,
                              ArrayRef<Value> newOperands,
                              ArrayRef<AxisLayout> operandLayouts,
                              AxisLayout target) const {
  assert(newOperands.size() == op->getNumOperands() &&
         operandLayouts.size() == op->getNumOperands() &&
         "one new operand and layout per operand");
  Rules rules(*this, tracker);
  return llvm::TypeSwitch<Operation *, Operation *>(op)
      .Case<linalg::GenericOp, linalg::ReduceOp, linalg::TransposeOp,
            tensor::ExpandShapeOp, tensor::CollapseShapeOp, tensor::ReshapeOp,
            tensor::ExtractSliceOp, tensor::InsertSliceOp>([&](auto typed) {
        return rules.rebuild(builder, typed, newOperands, operandLayouts,
                             target);
      })
      .Case<tensor::EmptyOp>(
          [&](auto typed) { return rules.rebuild(builder, typed, target); })
      .Case<linalg::FillOp, memref::ReinterpretCastOp,
            bufferization::ToTensorOp>([&](auto typed) {
        return rules.rebuild(builder, typed, newOperands, target);
      })
      .Default([&](Operation *) -> Operation * {
        // A kind without a rule is pinned (R2) or excluded (R3), so its
        // layout never changes: clone it with the remapped operands. A
        // layout change here means R2/R3 failed to stop it (Req 7.6, 7.7).
        if (op->getNumResults() == 0 ||
            !(AxisLayout::of(op->getResult(0), tracker) == target))
          llvm_unreachable("layout change requested for a kind without a rule");
        for (auto [operand, value] : llvm::zip(op->getOperands(), newOperands))
          if (operand.getType() != value.getType())
            llvm_unreachable("operand of a kind without a rule changed type");
        IRMapping mapping;
        mapping.map(op->getOperands(), newOperands);
        return builder.clone(*op, mapping);
      });
}

} // namespace qgemm_coupling
} // namespace buddy
} // namespace mlir
