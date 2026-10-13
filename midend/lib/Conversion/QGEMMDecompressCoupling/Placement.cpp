//===- Placement.cpp - Target placement for QGEMM coupling ----------------===//
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
// `pre` / `on-the-fly` placement, the H_d sweep with consumer-to-producer
// position propagation (Req 6.8), the frontier broadcast-position
// consistency check and the skip check of Req 6.1.
//
// Axis position (Req 6.8): a node that already carries d keeps it where it
// is. A node lifted into H_d takes d where its users in L_d read it, so
// every user reads the rebuilt node as it read the original and no
// consumer's operand map changes. The sweep visits users before producers,
// so each user's position is known when its producer is decided. Positions
// are computed in the frame of the separated input with only d changed: the
// other axis stays where the input has it, so deciding L_n never reads L_m
// and vice versa (Req 1.6). A candidate whose users require d at different
// positions, or whose kind cannot take d at the required position, is
// excluded from H_d with Remark R3 (Req 5.7).
//
//===----------------------------------------------------------------------===//
#include "Placement.h"
#include "RuleTable.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
namespace mlir {
namespace buddy {
namespace qgemm_coupling {
static constexpr Axis kAxes[] = {Axis::M, Axis::N};
namespace {
class PlacementBuilder {
public:
  PlacementBuilder(const LogicalGraph &graph, const AxisTracker &tracker,
                   const SeparationPlan &separation, bool onTheFly)
      : graph(graph), tracker(tracker), separation(separation),
        onTheFly(onTheFly), rules(tracker) {
    for (const LNode &node : graph.getNodes())
      nodeOps.insert(node.op);
    for (Operation *pb : separation.existingPBs)
      pbOps.insert(pb);
  }
  PlacementResult run() {
    PlacementResult result;
    unsigned count = graph.getNodes().size();
    for (Axis d : kAxes) {
      unsigned a = static_cast<unsigned>(d);
      result.lifted[a] = graph.getVariance(d);
      result.hoisted[a] = llvm::BitVector(count);
      result.excluded[a] = llvm::BitVector(count);
      result.positions[a].assign(count, std::nullopt);
      if (onTheFly)
        sweep(d, result);
      checkFixedPositions(d, result);
    }
    result.unchanged =
        separation.rewrites == 0 && llvm::all_of(kAxes, [&](Axis d) {
          return result.getLifted(d) == graph.getCurrentPlacement(d);
        });
    return result;
  }

private:
  /// Visit every use of `value`, looking through pure broadcasts of Omega(r).
  /// Uses nested in a region are reported on their ancestor in the block.
  void forEachLogicalUse(Value value,
                         function_ref<void(OpOperand &, Operation *)> fn) {
    for (OpOperand &use : value.getUses()) {
      Operation *owner = use.getOwner();
      if (pbOps.contains(owner)) {
        forEachLogicalUse(owner->getResult(0), fn);
        continue;
      }
      Operation *ancestor =
          value.getParentBlock()->findAncestorOpInBlock(*owner);
      fn(use, ancestor);
    }
  }
  /// Layout of node `id`'s result in the d frame: the separated input's
  /// layout, with d inserted at its recorded position when the node is in
  /// H_d and lacked d.
  AxisLayout frameLayout(unsigned id, Axis d,
                         const PlacementResult &result) const {
    const LNode &node = graph.getNodes()[id];
    if (node.op->getNumResults() == 0)
      return AxisLayout();
    AxisLayout layout = AxisLayout::of(node.op->getResult(0), tracker);
    if (!layout.get(d))
      if (std::optional<unsigned> pos =
              result.positions[static_cast<unsigned>(d)][id])
        layout = layout.withInserted(d, *pos);
    return layout;
  }

  /// `layout` restricted to the axes `value` carries in the input besides
  /// `d`: the frame of a producer that gains only d.
  AxisLayout toProducerFrame(AxisLayout layout, Value value, Axis d) const {
    for (Axis other : kAxes)
      if (other != d && layout.get(other) &&
          !tracker.getAxisPosition(value, other))
        layout = layout.without(other);
    return layout;
  }

  /// Position of d each user of `value` reads, in the producer's d frame.
  /// Users in L_d read it through their position rule; with
  /// `includeOutside`, an Outside_User reads it at its original operand
  /// position. Users carrying no requirement are skipped.
  void collectRequiredPositions(
      Value value, Axis d, const PlacementResult &result, bool includeOutside,
      function_ref<void(Operation *, unsigned, unsigned)> fn) {
    const llvm::BitVector &lifted = result.getLifted(d);
    forEachLogicalUse(value, [&](OpOperand &use, Operation *user) {
      if (use.getOwner() != user)
        return;
      std::optional<unsigned> userId =
          nodeOps.contains(user) ? graph.getNodeId(user) : std::nullopt;
      std::optional<AxisLayout> required;
      if (!userId) {
        if (!includeOutside)
          return;
        required = AxisLayout::of(use.get(), tracker);
      } else {
        if (!lifted.test(*userId))
          return;
        SmallVector<AffineMap> maps = getSeparatedMaps(user, separation);
        required = rules.operandLayout(user, use.getOperandNumber(),
                                       frameLayout(*userId, d, result), maps);
      }
      if (!required || !required->get(d))
        return;
      AxisLayout frame = toProducerFrame(*required, value, d);
      fn(user, use.getOperandNumber(), *frame.get(d));
    });
  }

  /// Position of d required of candidate `id`, or nullopt when its users
  /// disagree or none requires one.
  std::optional<unsigned> derivePosition(unsigned id, Axis d,
                                         const PlacementResult &result) {
    const LNode &node = graph.getNodes()[id];
    // A result-less op, such as a kept memref.copy, has no position to
    // carry d at; the rule table excludes it (Req 14.8).
    if (node.op->getNumResults() == 0)
      return std::nullopt;
    if (std::optional<unsigned> existing =
            tracker.getAxisPosition(node.op->getResult(0), d))
      return existing;
    std::optional<unsigned> position;
    bool conflict = false;
    for (Value value : node.op->getResults())
      collectRequiredPositions(value, d, result, /*includeOutside=*/false,
                               [&](Operation *, unsigned, unsigned required) {
                                 if (position && *position != required)
                                   conflict = true;
                                 position = required;
                               });
    if (conflict)
      return std::nullopt;
    return position;
  }
  /// True when `op` has at least one use and every use of its results is
  /// nested inside the region of a user rather than an operand of it.
  bool isOnlyCapturedFromAbove(Operation *op) {
    bool any = false, direct = false;
    for (Value value : op->getResults())
      forEachLogicalUse(value, [&](OpOperand &use, Operation *user) {
        any = true;
        direct |= use.getOwner() == user;
      });
    return any && !direct;
  }
  /// One reverse-topological sweep: users are decided before their
  /// producers, so H_d is the largest set satisfying Req 5.2.
  void sweep(Axis d, PlacementResult &result) {
    unsigned a = static_cast<unsigned>(d);
    ArrayRef<LNode> nodes = graph.getNodes();
    llvm::BitVector &lifted = result.lifted[a];
    for (unsigned id = nodes.size(); id-- > 0;) {
      const LNode &node = nodes[id];
      // A memref view op is a Boundary_Load and a candidate all the same:
      // the inserted dimension gets stride 0, so the lift reads no more
      // memory than the original load (Req 5.2).
      if (lifted.test(id) || node.hasOutsideUser)
        continue;
      if (node.isBoundaryLoad && !node.isMemrefView)
        continue;
      // A value read only from inside its users' regions, such as a lookup
      // table read by tensor.extract, is indexed explicitly and never needs
      // the parallel dim (Req 7.3, 7.12): it is not a candidate and draws no
      // remark.
      if (isOnlyCapturedFromAbove(node.op))
        continue;
      if (!llvm::all_of(node.users,
                        [&](unsigned user) { return lifted.test(user); }))
        continue;
      // Users requiring different positions degrade the node: one R3, no
      // consumer operand-map adjustment (Req 5.7, 6.8).
      std::optional<unsigned> position = derivePosition(id, d, result);
      if (position && rules.canInsert(node.op, d, *position)) {
        lifted.set(id);
        result.hoisted[a].set(id);
        result.positions[a][id] = *position;
      } else {
        result.excluded[a].set(id);
      }
    }
  }
  /// Phase A check behind Req 6.4: the users in L_d and the Outside_Users
  /// of one (value, d) on the frontier must read d at the same position.
  /// A candidate is resolved by the R3 degradation of the sweep; a producer
  /// outside the candidate set has position-fixed users, and a disagreement
  /// among them is an introduction form outside Req 2.1-2.4 (R1).
  void checkFixedPositions(Axis d, PlacementResult &result) {
    const llvm::BitVector &lifted = result.getLifted(d);
    const llvm::BitVector &excluded = result.excluded[static_cast<unsigned>(d)];
    for (auto [id, node] : llvm::enumerate(graph.getNodes())) {
      if (lifted.test(id) || excluded.test(id))
        continue;
      for (Value value : node.op->getResults()) {
        if (!isa<RankedTensorType>(value.getType()))
          continue;
        std::optional<unsigned> fixed;
        bool reported = false;
        collectRequiredPositions(
            value, d, result, /*includeOutside=*/true,
            [&](Operation *user, unsigned number, unsigned required) {
              if (reported)
                return;
              if (!fixed) {
                fixed = required;
                return;
              }
              if (*fixed == required)
                return;
              result.r1Diagnostics.push_back(
                  {user, d, number,
                   UnsupportedForm::InconsistentBroadcastPositions});
              reported = true;
            });
      }
    }
  }
  const LogicalGraph &graph;
  const AxisTracker &tracker;
  const SeparationPlan &separation;
  bool onTheFly;
  RuleTable rules;
  DenseSet<Operation *> nodeOps;
  DenseSet<Operation *> pbOps;
};
} // namespace
PlacementResult computePlacement(const LogicalGraph &graph,
                                 const AxisTracker &tracker,
                                 const SeparationPlan &separation,
                                 bool onTheFly) {
  return PlacementBuilder(graph, tracker, separation, onTheFly).run();
}
void PlacementResult::emitR3Remarks(const LogicalGraph &graph) const {
  for (auto [id, node] : llvm::enumerate(graph.getNodes()))
    for (Axis axis : kAxes)
      if (excluded[static_cast<unsigned>(axis)].test(id))
        node.op->emitRemark()
            << "qgemm-decompress-coupling: '" << getAxisName(axis)
            << "' cannot be inserted into op; kept outside the '"
            << getAxisName(axis) << "' space";
}
void PlacementResult::print(llvm::raw_ostream &os,
                            const LogicalGraph &graph) const {
  os << "  |L_m|=" << lifted[0].count() << " |L_n|=" << lifted[1].count()
     << " |H_m|=" << hoisted[0].count() << " |H_n|=" << hoisted[1].count()
     << " excluded(m,n)=" << excluded[0].count() << "," << excluded[1].count()
     << " R1=" << r1Diagnostics.size()
     << " skip=" << (unchanged ? "unchanged" : "rebuild") << "\n";
  for (auto [id, node] : llvm::enumerate(graph.getNodes())) {
    os << "    #" << id << " " << node.op->getName()
       << " L=" << (lifted[0].test(id) ? "m" : "-")
       << (lifted[1].test(id) ? "n" : "-")
       << " H=" << (hoisted[0].test(id) ? "m" : "-")
       << (hoisted[1].test(id) ? "n" : "-")
       << (excluded[0].test(id) || excluded[1].test(id) ? " excluded" : "");
    for (Axis axis : kAxes)
      if (std::optional<unsigned> pos =
              positions[static_cast<unsigned>(axis)][id])
        os << " pos_" << getAxisName(axis) << "=" << *pos;
    os << "\n";
  }
}
} // namespace qgemm_coupling
} // namespace buddy
} // namespace mlir
