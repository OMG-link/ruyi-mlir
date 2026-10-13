//===- Rebuilder.cpp - Rebuild of one QGEMM root in its target placement --===//
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
// The rebuild walks G(r) in canonical topological order. For every node it
// resolves each operand to its rebuilt value, looking through the pure
// broadcasts of the input, applies the separation plan of that operand
// (unit-axis collapse, operand-map replacement), inserts the pure broadcasts
// the operand needs to carry the node's target axes, and asks the rule table
// to rebuild the node with the target layout.
//
// Axis positions (Req 6.8) are fixed before any clone is built, in one
// reverse-topological pass: a node that gains an axis takes the layout its
// users read, through each user's position rule, so every consumer reads its
// rebuilt operand as it read the original and no consumer's operand map
// changes. A node that gains no axis keeps the axes it has where they are.
// Every broadcast inserts its axis at the position its consumer reads.
//
// Insertion point (Req 6.9): every clone goes immediately before its original
// op, and the helpers created for it (tensor.empty, recreated linalg.fill,
// unit-axis collapse) immediately before the clone. Every pure broadcast goes
// immediately after the definition of its input value, so the one cached
// broadcast per (value, axis, position) dominates all of its consumers,
// Outside_Users included, wherever those sit relative to Omega(r). Originals
// stay in place until the Eraser runs.
//
//===----------------------------------------------------------------------===//

#include "Rebuilder.h"
#include "Conversion/QGEMMRecognize/QGEMMAnnotation.h"
#include "RuleTable.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#include <tuple>

#define DEBUG_TYPE "qgemm-decompress-coupling"

namespace mlir {
namespace buddy {
namespace qgemm_coupling {

static constexpr Axis kAxes[] = {Axis::M, Axis::N};

namespace {

/// Number of tracked axes of `layout` sitting before `position`.
unsigned axesBefore(const AxisLayout &layout, unsigned position) {
  unsigned count = 0;
  for (Axis axis : kAxes)
    if (std::optional<unsigned> pos = layout.get(axis))
      count += *pos < position;
  return count;
}

bool isAxisPosition(const AxisLayout &layout, unsigned position) {
  return llvm::any_of(kAxes,
                      [&](Axis axis) { return layout.get(axis) == position; });
}

/// Position of the `ordinal`-th non-axis dim of a value of rank `rank`.
unsigned nonAxisPosition(const AxisLayout &layout, unsigned rank,
                         unsigned ordinal) {
  for (unsigned i = 0; i < rank; ++i) {
    if (isAxisPosition(layout, i))
      continue;
    if (ordinal-- == 0)
      return i;
  }
  llvm_unreachable("non-axis dim lost while rebuilding");
}

/// One planned axis insertion: `axis` must end at `finalPosition`.
struct AxisInsert {
  Axis axis;
  unsigned finalPosition;
};

class Rebuilder {
public:
  Rebuilder(const SetVector<Operation *> &omega, const AxisTracker &tracker,
            const SeparationPlan &separation, const LogicalGraph &graph,
            const PlacementResult &placement)
      : omega(omega), tracker(tracker), separation(separation), graph(graph),
        placement(placement), rules(tracker),
        builder(omega.front()->getContext()) {
    for (Operation *pb : separation.existingPBs)
      existingPBs.insert(pb);
  }

  RebuildResult run() {
    ArrayRef<LNode> nodes = graph.getNodes();
    computeTargets();
    result.clones.resize(nodes.size(), nullptr);
    for (auto [id, node] : llvm::enumerate(nodes))
      result.clones[id] =
          keepsInPlace(id, node) ? keepNode(node.op) : rebuildNode(id, node.op);
    rewireOutsideUsers();
    return std::move(result);
  }

private:
  //===--- Memref nodes -------------------------------------------------===//
  /// A node with no tensor result whose layout the placement leaves as it
  /// is, such as the memref.alloc and memref.copy of a load chain that keeps
  /// its copy (Req 14.8) or a memref view op that gains no axis, is reused in
  /// place rather than cloned. A memref is not rebuilt through a pure
  /// broadcast, and a result-less op has no layout to change.
  bool keepsInPlace(unsigned id, const LNode &node) const {
    if (llvm::any_of(node.op->getResultTypes(),
                     [](Type type) { return isa<TensorType>(type); }))
      return false;
    if (node.op->getNumResults() == 0)
      return true;
    return llvm::all_of(kAxes, [&](Axis axis) {
      return placement.getLifted(axis).test(id) ==
             node.hasAxis[static_cast<unsigned>(axis)];
    });
  }
  //===--- Outside uses ---------------------------------------------------===//

  /// Block-level ancestors of the users of `value` outside Omega(r), looking
  /// through the input's pure broadcasts.
  void collectOutsideUses(Value value, SmallVectorImpl<OpOperand *> &uses) {
    Block *block = omega.front()->getBlock();
    for (OpOperand &use : value.getUses()) {
      Operation *user = block->findAncestorOpInBlock(*use.getOwner());
      if (user && omega.contains(user)) {
        if (existingPBs.contains(user))
          collectOutsideUses(user->getResult(0), uses);
        continue;
      }
      uses.push_back(&use);
    }
  }

  //===--- Value resolution -----------------------------------------------===//

  /// Look through the input's pure broadcasts to the logical value.
  Value lookThrough(Value value) const {
    while (Operation *def = value.getDefiningOp()) {
      if (!existingPBs.contains(def))
        break;
      value = def->getOperand(0);
    }
    return value;
  }

  bool isNodeValue(Value value) const {
    Operation *def = value.getDefiningOp();
    return def && graph.getNodeId(def).has_value();
  }

  /// Rebuilt value of `old` (a node result after looking through PBs) or the
  /// value itself when it comes from outside Omega(r).
  Value resolve(Value old) {
    Value logical = lookThrough(old);
    if (isNodeValue(logical)) {
      auto it = valueMap.find(logical);
      assert(it != valueMap.end() &&
             "operand node not rebuilt before its user");
      return it->second;
    }
    if (!layouts.count(logical))
      layouts[logical] = AxisLayout::of(logical, tracker);
    return logical;
  }

  AxisLayout layoutOf(Value value) const {
    auto it = layouts.find(value);
    assert(it != layouts.end() && "rebuilt value without a layout");
    return it->second;
  }

  //===--- Types ----------------------------------------------------------===//

  /// `oldType` with its axes moved from `from` to `to`.
  RankedTensorType relabelType(Type oldType, AxisLayout from,
                               AxisLayout to) const {
    auto type = cast<RankedTensorType>(oldType);
    SmallVector<int64_t> shape(type.getShape());
    SmallVector<unsigned> removed;
    for (Axis axis : kAxes)
      if (std::optional<unsigned> pos = from.get(axis))
        removed.push_back(*pos);
    llvm::sort(removed, std::greater<unsigned>());
    for (unsigned pos : removed)
      shape.erase(shape.begin() + pos);
    SmallVector<std::pair<unsigned, Axis>> inserted;
    for (Axis axis : kAxes)
      if (std::optional<unsigned> pos = to.get(axis))
        inserted.push_back({*pos, axis});
    llvm::sort(inserted, [](auto a, auto b) { return a.first < b.first; });
    for (auto [pos, axis] : inserted) {
      std::optional<int64_t> size = rules.getAxisSize(axis);
      if (!size || pos > shape.size())
        llvm_unreachable("cannot place a rebuilt axis");
      shape.insert(shape.begin() + pos, *size);
    }
    return RankedTensorType::get(shape, type.getElementType(),
                                 type.getEncoding());
  }

  //===--- Created ops ----------------------------------------------------===//

  /// PB(value, axis) at `position`, reused per (value, axis, position)
  /// (Req 6.3, 6.4).
  Value getOrCreatePB(Value value, Axis axis, unsigned position) {
    auto key = std::make_tuple(value, static_cast<unsigned>(axis), position);
    auto it = pbCache.find(key);
    if (it != pbCache.end())
      return it->second;
    AxisLayout from = layoutOf(value);
    assert(!from.get(axis) && "broadcast of an axis the value already has");
    auto type = cast<RankedTensorType>(value.getType());
    SmallVector<int64_t> shape(type.getShape());
    std::optional<int64_t> size = rules.getAxisSize(axis);
    if (!size || position > shape.size())
      llvm_unreachable("cannot place a broadcast axis");
    shape.insert(shape.begin() + position, *size);

    // Right after the definition of `value` (Req 6.9), so the cached
    // broadcast dominates every consumer of (value, axis, position).
    OpBuilder::InsertionGuard guard(builder);
    if (Operation *def = value.getDefiningOp())
      builder.setInsertionPointAfter(def);
    else
      builder.setInsertionPointToStart(cast<BlockArgument>(value).getOwner());

    Location loc = value.getLoc();
    MLIRContext *context = builder.getContext();
    unsigned rank = shape.size();
    auto empty =
        builder.create<tensor::EmptyOp>(loc, shape, type.getElementType());
    result.createdHelpers.push_back(empty);
    SmallVector<AffineExpr> inputExprs;
    for (unsigned i = 0; i < rank; ++i)
      if (i != position)
        inputExprs.push_back(getAffineDimExpr(i, context));
    SmallVector<AffineMap> maps = {
        AffineMap::get(rank, 0, inputExprs, context),
        AffineMap::getMultiDimIdentityMap(rank, context)};
    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);
    auto pb = builder.create<linalg::GenericOp>(
        loc, TypeRange{empty.getType()}, ValueRange{value},
        ValueRange{empty.getResult()}, maps, iterators,
        [](OpBuilder &nested, Location nestedLoc, ValueRange args) {
          nested.create<linalg::YieldOp>(nestedLoc, args[0]);
        });
    result.createdPBs.push_back(pb);

    AxisLayout to;
    for (Axis other : kAxes)
      if (std::optional<unsigned> pos = from.get(other))
        to.set(other, *pos >= position ? *pos + 1 : *pos);
    to.set(axis, position);
    Value broadcast = pb->getResult(0);
    layouts[broadcast] = to;
    pbCache[key] = broadcast;
    return broadcast;
  }

  /// Insert the axes of `inserts` into `value`, m before n (Req 6.3). Each
  /// broadcast's position accounts for the axes inserted after it.
  Value insertAxes(Value value, ArrayRef<AxisInsert> inserts) {
    for (auto [index, insert] : llvm::enumerate(inserts)) {
      unsigned position = insert.finalPosition;
      for (const AxisInsert &later : inserts.drop_front(index + 1))
        position -= later.finalPosition < insert.finalPosition;
      value = getOrCreatePB(value, insert.axis, position);
    }
    return value;
  }

  /// Drop the unit axes `unitDrops` of `oldOperand`'s type from the rebuilt
  /// `value` through one tensor.collapse_shape (Req 2.4).
  Value collapseUnitAxes(Value value, Value oldOperand,
                         ArrayRef<unsigned> unitDrops) {
    if (unitDrops.empty())
      return value;
    AxisLayout oldLayout = AxisLayout::of(oldOperand, tracker);
    AxisLayout layout = layoutOf(value);
    auto type = cast<RankedTensorType>(value.getType());
    unsigned rank = type.getRank();
    llvm::SmallDenseSet<unsigned> dropped;
    for (unsigned position : unitDrops) {
      if (isAxisPosition(oldLayout, position))
        llvm_unreachable("unit-axis drop on a tracked parallel axis");
      unsigned ordinal = position - axesBefore(oldLayout, position);
      unsigned newPosition = nonAxisPosition(layout, rank, ordinal);
      assert(type.getDimSize(newPosition) == 1 && "dropped dim is not unit");
      dropped.insert(newPosition);
    }
    SmallVector<ReassociationIndices> groups;
    ReassociationIndices pending;
    SmallVector<int64_t> shape;
    for (unsigned i = 0; i < rank; ++i) {
      pending.push_back(i);
      if (dropped.contains(i))
        continue;
      groups.push_back(pending);
      pending.clear();
      shape.push_back(type.getDimSize(i));
    }
    if (!pending.empty() && !groups.empty())
      llvm::append_range(groups.back(), pending);
    auto collapse = builder.create<tensor::CollapseShapeOp>(
        value.getLoc(),
        RankedTensorType::get(shape, type.getElementType(), type.getEncoding()),
        value, groups);
    result.createdHelpers.push_back(collapse);
    AxisLayout to;
    for (Axis axis : kAxes)
      if (std::optional<unsigned> pos = layout.get(axis)) {
        unsigned shift = 0;
        for (unsigned d : dropped)
          shift += d < *pos;
        to.set(axis, *pos - shift);
      }
    layouts[collapse.getResult()] = to;
    return collapse.getResult();
  }

  /// Fresh init of `type` for an init operand produced outside Omega(r):
  /// tensor.empty and linalg.fill are recreated with the rebuilt shape.
  Value createInit(Value old, RankedTensorType type) {
    Operation *def = old.getDefiningOp();
    if (auto empty = dyn_cast_if_present<tensor::EmptyOp>(def)) {
      if (!empty.getDynamicSizes().empty())
        llvm_unreachable("dynamic tensor.empty init");
      auto created = builder.create<tensor::EmptyOp>(
          empty.getLoc(), type.getShape(), type.getElementType());
      result.createdHelpers.push_back(created);
      return created.getResult();
    }
    if (auto fill = dyn_cast_if_present<linalg::FillOp>(def)) {
      Value init = createInit(fill.getDpsInitOperand(0)->get(), type);
      auto created = builder.create<linalg::FillOp>(
          fill.getLoc(), fill.getDpsInputOperand(0)->get(), init);
      result.createdHelpers.push_back(created);
      return created->getResult(0);
    }
    if (old.getType() == type)
      return old;
    llvm_unreachable("init produced outside Omega(r) by an unsupported op");
  }

  //===--- Nodes ----------------------------------------------------------===//

  static bool isInitOperand(Operation *op, OpOperand &operand) {
    auto dps = dyn_cast<DestinationStyleOpInterface>(op);
    return dps && dps.isDpsInit(&operand);
  }

  /// Result type of `op`'s init-matching result `index` under `target`.
  RankedTensorType rebuiltResultType(Operation *op, unsigned index,
                                     AxisLayout target) const {
    AxisLayout from = AxisLayout::of(op->getResult(0), tracker);
    return relabelType(op->getResult(index).getType(), from, target);
  }

  /// Apply the operand-map replacements of the separation plan to `op` in
  /// place and return the maps to restore afterwards.
  std::optional<ArrayAttr> applyIntroductionMaps(Operation *op) {
    auto generic = dyn_cast<linalg::GenericOp>(op);
    if (!generic)
      return std::nullopt;
    SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
    bool changed = false;
    for (const Introduction &introduction : separation.introductions) {
      if (introduction.consumer != op)
        continue;
      maps[introduction.operandNumber] = introduction.newOperandMap;
      changed = true;
    }
    if (!changed)
      return std::nullopt;
    ArrayAttr original = generic.getIndexingMaps();
    generic.setIndexingMapsAttr(builder.getAffineMapArrayAttr(maps));
    return original;
  }

  const Introduction *findIntroduction(Operation *op, unsigned number) const {
    for (const Introduction &introduction : separation.introductions)
      if (introduction.consumer == op && introduction.operandNumber == number)
        return &introduction;
    return nullptr;
  }

  /// Rebuilt operand `operand` of node `op` in the layout `op` reads it
  /// with when rebuilt with `target` (Req 6.8). Missing axes are inserted by
  /// pure broadcasts at the positions `op` reads them; the operand map of
  /// `op` is never adjusted.
  Value rebuildOperand(Operation *op, OpOperand &operand, AxisLayout target) {
    Value old = operand.get();
    Value logical = lookThrough(old);
    std::optional<AxisLayout> required =
        rules.operandLayout(op, operand.getOperandNumber(), target, {});
    if (!required)
      llvm_unreachable("tensor operand without a position rule");
    // A tensor.empty / linalg.fill outside every Scope (an init chain entering
    // Omega(r), e.g. a fill expanded by a reshape of Omega(r)) is recreated
    // with the axes the consumer needs instead of being broadcast or
    // stripped.
    if (!isNodeValue(logical) &&
        isa_and_present<tensor::EmptyOp, linalg::FillOp>(
            logical.getDefiningOp())) {
      Value created = createInit(
          logical, relabelType(logical.getType(),
                               AxisLayout::of(logical, tracker), *required));
      layouts[created] = *required;
      return created;
    }
    Value value = resolve(old);
    if (const Introduction *introduction =
            findIntroduction(op, operand.getOperandNumber()))
      value = collapseUnitAxes(value, old, introduction->unitDrops);
    AxisLayout layout = layoutOf(value);
    SmallVector<AxisInsert> inserts;
    AxisLayout present = *required;
    for (Axis axis : kAxes) {
      if (layout.get(axis) && !required->get(axis))
        llvm_unreachable("producer carries an axis its consumer lacks");
      if (required->get(axis) && !layout.get(axis)) {
        inserts.push_back({axis, *required->get(axis)});
        present = present.without(axis);
      }
    }
    if (!(present == layout))
      llvm_unreachable("producer and consumer disagree on an axis position");
    return insertAxes(value, inserts);
  }

  /// A node kept in place is reused as it stands: its results map to
  /// themselves with their existing axis layout, so its consumers see the
  /// input form. Nothing is cloned, so a result-less op such as memref.copy
  /// never reaches the rule table.
  Operation *keepNode(Operation *op) {
    for (Value value : op->getResults()) {
      valueMap[value] = value;
      layouts[value] = AxisLayout::of(value, tracker);
    }
    result.keptOps.push_back(op);
    return op;
  }

  /// Users of `value` at block level, looking through the input's pure
  /// broadcasts. Uses nested in a region (captured from above) are skipped.
  void forEachDirectUse(Value value,
                        function_ref<void(OpOperand &, Operation *)> fn) {
    for (OpOperand &use : value.getUses()) {
      Operation *owner = use.getOwner();
      if (existingPBs.contains(owner)) {
        forEachDirectUse(owner->getResult(0), fn);
        continue;
      }
      if (owner->getBlock() == value.getParentBlock())
        fn(use, owner);
    }
  }

  /// Target layout of every node, consumers before producers (Req 6.8). A
  /// node that gains no axis keeps its axes where they are. A node that
  /// gains one takes the layout its users read it with; the placement has
  /// excluded every node whose users disagree on the gained axis, and an
  /// op and its destination-style init derive their layouts from the same
  /// consumer.
  void computeTargets() {
    ArrayRef<LNode> nodes = graph.getNodes();
    targets.assign(nodes.size(), AxisLayout());
    for (unsigned id = nodes.size(); id-- > 0;) {
      const LNode &node = nodes[id];
      bool has[2];
      bool gains = false;
      for (Axis axis : kAxes) {
        unsigned a = static_cast<unsigned>(axis);
        has[a] = placement.getLifted(axis).test(id);
        gains |= has[a] && !node.hasAxis[a];
      }
      if (node.op->getNumResults() == 0)
        continue;
      if (!gains) {
        targets[id] =
            AxisLayout::kept(node.op->getResult(0), has[0], has[1], tracker);
        continue;
      }
      std::optional<AxisLayout> chosen;
      for (Value value : node.op->getResults())
        forEachDirectUse(value, [&](OpOperand &use, Operation *user) {
          std::optional<unsigned> userId = graph.getNodeId(user);
          if (!userId)
            llvm_unreachable("lifted node with an Outside_User");
          std::optional<AxisLayout> required = rules.operandLayout(
              user, use.getOperandNumber(), targets[*userId],
              getSeparatedMaps(user, separation));
          if (!required)
            return;
          // Axes the user carries but this node does not get are inserted
          // by a broadcast on the edge.
          for (Axis axis : kAxes)
            if (required->get(axis) && !has[static_cast<unsigned>(axis)])
              required = required->without(axis);
          if (!chosen)
            chosen = required;
          else if (!(*chosen == *required))
            llvm_unreachable("users of a lifted node disagree on its layout");
        });
      if (!chosen)
        llvm_unreachable("lifted node without a user requiring a layout");
      for (Axis axis : kAxes)
        if (has[static_cast<unsigned>(axis)] != chosen->get(axis).has_value())
          llvm_unreachable("derived layout does not match the placement");
      targets[id] = *chosen;
    }
  }

  Operation *rebuildNode(unsigned id, Operation *op) {
    assert(op->getNumResults() >= 1 && "G(r) node without a result");
    // The clone and its helpers go immediately before the original (Req 6.9).
    builder.setInsertionPoint(op);
    AxisLayout target = targets[id];
    std::optional<ArrayAttr> originalMaps = applyIntroductionMaps(op);
    auto generic = dyn_cast<linalg::GenericOp>(op);

    SmallVector<Value> operands;
    SmallVector<AxisLayout> operandLayouts;
    unsigned initIndex = 0;
    for (OpOperand &operand : op->getOpOperands()) {
      Value old = operand.get();
      if (!isa<RankedTensorType>(old.getType())) {
        Value logical = lookThrough(old);
        auto it = valueMap.find(logical);
        operands.push_back(it == valueMap.end() ? old : it->second);
        operandLayouts.push_back(AxisLayout());
        continue;
      }
      // The shape operand of a tensor.reshape is an extent list, not data:
      // the rule builds the rebuilt shape constant itself, so the operand is
      // neither resolved nor broadcast.
      if (auto reshape = dyn_cast<tensor::ReshapeOp>(op);
          reshape && &operand == &reshape.getShapeMutable()) {
        operands.push_back(old);
        operandLayouts.push_back(AxisLayout());
        continue;
      }
      bool isInit = isInitOperand(op, operand);
      unsigned resultIndex = isInit ? initIndex++ : 0;
      if (isInit && !isNodeValue(lookThrough(old))) {
        // Req 7.2: an init the body does not read is replaced by the rule.
        if (generic && !generic.payloadUsesValueFromOperand(&operand))
          operands.push_back(old);
        else
          operands.push_back(
              createInit(old, rebuiltResultType(op, resultIndex, target)));
        operandLayouts.push_back(target);
        continue;
      }
      if (isInit && generic && !generic.payloadUsesValueFromOperand(&operand)) {
        operands.push_back(old);
        operandLayouts.push_back(target);
        continue;
      }
      Value value = rebuildOperand(op, operand, target);
      operands.push_back(value);
      operandLayouts.push_back(layoutOf(value));
    }

    Operation *rebuilt =
        rules.rebuild(builder, op, operands, operandLayouts, target);
    if (originalMaps)
      cast<linalg::GenericOp>(op).setIndexingMapsAttr(*originalMaps);
    remapCapturedValues(rebuilt);
    for (auto [oldResult, newResult] :
         llvm::zip(op->getResults(), rebuilt->getResults())) {
      valueMap[oldResult] = newResult;
      layouts[newResult] = target;
    }
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] rebuilt #" << id << " " << op->getName()
               << " -> " << rebuilt->getResult(0).getType() << "\n");
    return rebuilt;
  }

  /// Values of Omega(r) captured from above inside a rebuilt region (e.g. a
  /// tensor.extract table) are redirected to their rebuilt counterparts.
  void remapCapturedValues(Operation *rebuilt) {
    rebuilt->walk([&](Operation *nested) {
      if (nested == rebuilt)
        return;
      for (OpOperand &operand : nested->getOpOperands()) {
        Value logical = lookThrough(operand.get());
        if (!isNodeValue(logical))
          continue;
        auto it = valueMap.find(logical);
        assert(it != valueMap.end() && "captured node not rebuilt yet");
        if (it->second.getType() != operand.get().getType())
          llvm_unreachable("captured value changed type in the rebuild");
        operand.set(it->second);
      }
    });
  }

  //===--- Outside_Users
  //----------------------------------------------------===//

  /// Feed every Outside_User the rebuilt value with its original operand
  /// type (Req 6.5, 6.7).
  void rewireOutsideUsers() {
    for (const LNode &node : graph.getNodes()) {
      for (Value oldResult : node.op->getResults()) {
        SmallVector<OpOperand *> uses;
        collectOutsideUses(oldResult, uses);
        if (uses.empty())
          continue;
        Value rebuilt = valueMap.lookup(oldResult);
        assert(rebuilt && "node with an Outside_User was not rebuilt");
        for (OpOperand *use : uses) {
          Value old = use->get();
          AxisLayout layout = layoutOf(rebuilt);
          SmallVector<AxisInsert> inserts;
          for (Axis axis : kAxes) {
            std::optional<unsigned> expected =
                tracker.getAxisPosition(old, axis);
            assert(!(layout.get(axis) && !expected) &&
                   "Outside_User operand would gain a parallel dimension");
            if (expected && !layout.get(axis))
              inserts.push_back({axis, *expected});
          }
          Value value = insertAxes(rebuilt, inserts);
          assert(value.getType() == old.getType() &&
                 "Outside_User operand type changed by the rebuild");
          use->set(value);
        }
      }
    }
  }

  const SetVector<Operation *> &omega;
  const AxisTracker &tracker;
  const SeparationPlan &separation;
  const LogicalGraph &graph;
  const PlacementResult &placement;
  RuleTable rules;
  OpBuilder builder;
  DenseSet<Operation *> existingPBs;
  /// Old node result -> rebuilt value.
  DenseMap<Value, Value> valueMap;
  /// Axis layout of every rebuilt or created value and of outside operands.
  DenseMap<Value, AxisLayout> layouts;
  DenseMap<std::tuple<Value, unsigned, unsigned>, Value> pbCache;
  /// Target layout of every node's result, by node id.
  SmallVector<AxisLayout> targets;
  RebuildResult result;
};

} // namespace

RebuildResult rebuildRoot(const SetVector<Operation *> &omega,
                          const AxisTracker &tracker,
                          const SeparationPlan &separation,
                          const LogicalGraph &graph,
                          const PlacementResult &placement) {
  return Rebuilder(omega, tracker, separation, graph, placement).run();
}

//===--- AnnotationWriter ---------------------------------------------===//

static DictionaryAttr makeNodeAttr(MLIRContext *context, unsigned rootId,
                                   StringRef stage) {
  SmallVector<NamedAttribute, 2> fields = {
      NamedAttribute(StringAttr::get(context, kQGEMMNodeRootField),
                     IntegerAttr::get(IntegerType::get(context, 64), rootId)),
      NamedAttribute(StringAttr::get(context, kQGEMMNodeStageField),
                     StringAttr::get(context, stage))};
  return DictionaryAttr::get(context, fields);
}

static DictionaryAttr makeRootAttr(MLIRContext *context, unsigned mDim,
                                   unsigned nDim) {
  auto i64 = IntegerType::get(context, 64);
  SmallVector<NamedAttribute, 2> fields = {
      NamedAttribute(StringAttr::get(context, kQGEMMRootMDimField),
                     IntegerAttr::get(i64, mDim)),
      NamedAttribute(StringAttr::get(context, kQGEMMRootNDimField),
                     IntegerAttr::get(i64, nDim))};
  return DictionaryAttr::get(context, fields);
}

void writeAnnotations(const RebuildResult &rebuild, const LogicalGraph &graph,
                      const PlacementResult &placement, Operation *finalReduce,
                      unsigned rootId, unsigned mDim, unsigned nDim) {
  std::optional<unsigned> finalId = graph.getNodeId(finalReduce);
  assert(finalId && "final reduction is not a node of G(r)");
  MLIRContext *context = finalReduce->getContext();
  const llvm::BitVector &liftedM =
      placement.lifted[static_cast<unsigned>(Axis::M)];
  const llvm::BitVector &liftedN =
      placement.lifted[static_cast<unsigned>(Axis::N)];
  DictionaryAttr computeAttr =
      makeNodeAttr(context, rootId, kQGEMMComputeStage);
  DictionaryAttr dataPrepAttr =
      makeNodeAttr(context, rootId, kQGEMMDataPrepStage);

  // Rebuilt ops: the stage is a function of L_m and L_n only (Req 8.1, 8.2).
  // Rule clones may have copied the original discardable attributes, so the
  // node attribute is overwritten and any copied qgemm.root dropped.
  for (auto [id, clone] : llvm::enumerate(rebuild.clones)) {
    assert(clone && "G(r) node without a rebuilt op");
    bool compute = liftedM.test(id) && liftedN.test(id);
    clone->setAttr(kQGEMMNodeAttr, compute ? computeAttr : dataPrepAttr);
    if (id == *finalId)
      clone->setAttr(kQGEMMRootAttr, makeRootAttr(context, mDim, nDim));
    else
      clone->removeAttr(kQGEMMRootAttr);
  }
  // Every created pure broadcast is compute (Req 8.3).
  for (Operation *pb : rebuild.createdPBs)
    pb->setAttr(kQGEMMNodeAttr, computeAttr);
  // Helpers: empty/fill stay outside the scope (Req 8.5); a unit-axis
  // collapse takes the stage of its consumer, which is annotated by now.
  for (Operation *helper : rebuild.createdHelpers) {
    if (isa<tensor::EmptyOp, linalg::FillOp>(helper))
      continue;
    auto collapse = dyn_cast<tensor::CollapseShapeOp>(helper);
    if (!collapse)
      llvm_unreachable("unexpected created helper kind");
    Attribute stage;
    for (Operation *user : collapse->getUsers())
      if ((stage = user->getAttr(kQGEMMNodeAttr)))
        break;
    if (!stage)
      llvm_unreachable("unit-axis collapse without an annotated consumer");
    collapse->setAttr(kQGEMMNodeAttr, stage);
  }
  LLVM_DEBUG({
    // Stage summary over G(r): `lifted` counts nodes that become compute
    // without carrying both axes before, `demoted` the converse.
    const llvm::BitVector &curM = graph.getCurrentPlacement(Axis::M);
    const llvm::BitVector &curN = graph.getCurrentPlacement(Axis::N);
    unsigned compute = 0, lifted = 0, demoted = 0;
    for (unsigned id = 0, e = rebuild.clones.size(); id < e; ++id) {
      bool now = liftedM.test(id) && liftedN.test(id);
      bool before = curM.test(id) && curN.test(id);
      compute += now;
      lifted += now && !before;
      demoted += before && !now;
    }
    llvm::dbgs() << "[" DEBUG_TYPE "] root " << rootId
                 << " annotation: nodes compute=" << compute
                 << " data_prep=" << rebuild.clones.size() - compute
                 << " pbs=" << rebuild.createdPBs.size()
                 << " lifted-to-compute=" << lifted
                 << " demoted-to-data_prep=" << demoted << "\n";
  });
}

//===--- Eraser -------------------------------------------------------===//

/// An out-of-scope op the eraser may remove once it has no users.
static bool isErasableOrphanKind(Operation *op) {
  return !op->hasAttr(kQGEMMNodeAttr) &&
         (isa<tensor::EmptyOp, linalg::FillOp>(op) ||
          op->hasTrait<OpTrait::ConstantLike>());
}

void eraseOriginals(const SetVector<Operation *> &omega,
                    const RebuildResult &rebuild) {
  SetVector<Operation *> worklist;
  auto eraseOp = [&](Operation *op) {
    for (Value operand : op->getOperands())
      if (Operation *def = operand.getDefiningOp())
        worklist.insert(def);
    // Captured-from-above operands of nested regions also lose a user.
    op->walk([&](Operation *nested) {
      if (nested == op)
        return;
      for (Value operand : nested->getOperands())
        if (Operation *def = operand.getDefiningOp())
          if (!op->isAncestor(def))
            worklist.insert(def);
    });
    worklist.remove(op);
    op->erase();
  };

  // Originals of Omega(r), consumers before producers.  Nodes kept in place
  // are not originals: they are part of the rebuilt operator.
  DenseSet<Operation *> kept;
  kept.insert(rebuild.keptOps.begin(), rebuild.keptOps.end());
  SmallVector<Operation *> originals;
  for (Operation *op : omega)
    if (!kept.contains(op))
      originals.push_back(op);
  llvm::sort(originals,
             [](Operation *a, Operation *b) { return b->isBeforeInBlock(a); });
  for (Operation *op : originals) {
    if (!op->use_empty())
      llvm_unreachable("original op of Omega(r) still used after the rebuild");
    eraseOp(op);
  }

  // Created ops left unused by the rebuild itself (Req 6.6). This happens
  // when a node of G(r) only feeds an init its consumer's body does not read
  // (e.g. a tensor.expand_shape of a tensor.empty as a cast's outs): the node
  // is cloned before its consumer, whose rule then takes a fresh
  // tensor.empty (Req 7.2), leaving the clone and its created init dead.
  DenseSet<Operation *> created;
  created.insert(rebuild.clones.begin(), rebuild.clones.end());
  for (Operation *op : rebuild.keptOps)
    created.erase(op);
  created.insert(rebuild.createdPBs.begin(), rebuild.createdPBs.end());
  created.insert(rebuild.createdHelpers.begin(), rebuild.createdHelpers.end());
  SmallVector<Operation *> createdOrder;
  for (Operation *op : rebuild.clones)
    if (!kept.contains(op))
      createdOrder.push_back(op);
  llvm::append_range(createdOrder, rebuild.createdPBs);
  llvm::append_range(createdOrder, rebuild.createdHelpers);
  llvm::sort(createdOrder,
             [](Operation *a, Operation *b) { return b->isBeforeInBlock(a); });
  DenseSet<Operation *> erasedCreated;
  for (Operation *op : createdOrder) {
    if (!op->use_empty())
      continue;
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] erase unused created op " << op->getName()
               << " : " << op->getResultTypes() << "\n");
    erasedCreated.insert(op);
    eraseOp(op);
  }

  // Orphan out-of-scope empties/fills/constants, to a fixed point.
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (erasedCreated.contains(op))
      continue;
    if (!op->use_empty())
      continue;
    // Created helpers already covered above; anything else must be an
    // out-of-scope empty/fill/constant to be removable.
    if (!created.contains(op) && !isErasableOrphanKind(op))
      continue;
    if (created.contains(op))
      erasedCreated.insert(op);
    eraseOp(op);
  }
}

} // namespace qgemm_coupling
} // namespace buddy
} // namespace mlir
