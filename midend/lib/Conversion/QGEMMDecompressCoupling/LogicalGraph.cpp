//===- LogicalGraph.cpp - Logical graph and variance for QGEMM ------------===//
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
// G(r) construction, boundary-load and Outside_User detection, deterministic
// topological ordering and the forward variance sweep (Req 3.1, 3.2, 4).
//
// An op that is invariant along d but whose d axis the rule table cannot strip
// is pinned in V_d with Remark R2 (Req 3.2). Pins are also counted as pin
// candidates in the debug print.
//
//===----------------------------------------------------------------------===//
#include "LogicalGraph.h"
#include "Conversion/QGEMMRecognize/QGEMMAnnotation.h"
#include "RuleTable.h"
#include "Separator.h"
#include "Utils/MemrefView.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
#include <tuple>
namespace mlir {
namespace buddy {
namespace qgemm_coupling {
static constexpr Axis kAxes[] = {Axis::M, Axis::N};
bool canStripAxis(Operation *op, Axis d, const AxisTracker &tracker) {
  return RuleTable(tracker).canStrip(op, d);
}
namespace {
/// Tie-break key of the canonical topological order. Independent of where
/// pure broadcasts sit: boundary-load ancestor ranks, op name, then the
/// lexicographically smallest operand-index path from the final reduction
/// (unreachable nodes last). The Omega(r) index makes the order total.
struct OrderKey {
  SmallVector<unsigned> ancestorLoads;
  StringRef name;
  bool unreachable;
  SmallVector<unsigned> path;
  unsigned omegaIndex;
  bool operator<(const OrderKey &other) const {
    return std::tie(ancestorLoads, name, unreachable, path, omegaIndex) <
           std::tie(other.ancestorLoads, other.name, other.unreachable,
                    other.path, other.omegaIndex);
  }
};
class GraphBuilder {
public:
  GraphBuilder(const SetVector<Operation *> &omega, const AxisTracker &tracker,
               Operation *finalReduce)
      : omega(omega), tracker(tracker), finalReduce(finalReduce) {}
  void run(SmallVector<LNode> &outNodes,
           DenseMap<Operation *, unsigned> &outIds) {
    for (Operation *op : omega) {
      if (isPureBroadcast(op, tracker))
        continue;
      tempIds[op] = ops.size();
      ops.push_back(op);
    }
    unsigned count = ops.size();
    temp.resize(count);
    useEdges.resize(count);
    for (unsigned id = 0; id < count; ++id) {
      temp[id].op = ops[id];
      buildEdges(id);
      for (Axis axis : kAxes)
        if (ops[id]->getNumResults() > 0)
          temp[id].hasAxis[static_cast<unsigned>(axis)] =
              tracker.getAxisPosition(ops[id]->getResult(0), axis).has_value();
      temp[id].hasOutsideUser = hasOutsideUser(ops[id]);
      temp[id].isMemrefView = isMemrefView(ops[id]);
    }
    for (unsigned id = 0; id < count; ++id)
      temp[id].isBoundaryLoad = isBoundaryLoad(ops[id]);
    unsigned rank = 0;
    for (unsigned id = 0; id < count; ++id)
      if (temp[id].isBoundaryLoad)
        loadRank[id] = rank++;
    order(outNodes, outIds);
  }

private:
  /// Follow `value` up through pure broadcasts of Omega(r).
  Value lookThroughPBs(Value value) const {
    while (Operation *producer = value.getDefiningOp()) {
      if (!omega.contains(producer) || !isPureBroadcast(producer, tracker))
        break;
      value = producer->getOperand(0);
    }
    return value;
  }
  /// Node id of the op producing `value` (PBs looked through), if in G(r).
  std::optional<unsigned> getProducerNode(Value value) const {
    Operation *producer = lookThroughPBs(value).getDefiningOp();
    if (!producer)
      return std::nullopt;
    auto it = tempIds.find(producer);
    if (it == tempIds.end())
      return std::nullopt;
    return it->second;
  }
  void addEdge(unsigned id, unsigned producer, unsigned operandNumber) {
    useEdges[producer].push_back({id, operandNumber});
    if (!llvm::is_contained(temp[id].inputs, producer))
      temp[id].inputs.push_back(producer);
    if (!llvm::is_contained(temp[producer].users, id))
      temp[producer].users.push_back(id);
  }
  /// A memref operand the op only writes, such as the target of a
  /// memref.copy, carries no data into the op, so it is not a data edge:
  /// the op does not vary with the buffer's contents and is not a user that
  /// needs the buffer's axes.
  static bool isWriteOnlyOperand(Operation *op, Value value) {
    auto effects = dyn_cast<MemoryEffectOpInterface>(op);
    if (!effects || !isa<BaseMemRefType>(value.getType()))
      return false;
    return effects.getEffectOnValue<MemoryEffects::Write>(value) &&
           !effects.getEffectOnValue<MemoryEffects::Read>(value);
  }
  void buildEdges(unsigned id) {
    for (OpOperand &operand : ops[id]->getOpOperands()) {
      std::optional<unsigned> producer = getProducerNode(operand.get());
      if (!producer)
        continue;
      if (isWriteOnlyOperand(ops[id], operand.get()))
        continue;
      addEdge(id, *producer, operand.getOperandNumber());
    }
    // A value of Omega(r) captured from above inside a region, such as the
    // iq2_xxs lookup table read by a tensor.extract, is a dependency too: the
    // rebuild remaps it, so it must be ordered before its consumer. The
    // operand number of the enclosing op is used as the path key, since the
    // nested use has none of its own.
    for (Region &region : ops[id]->getRegions())
      region.walk([&](Operation *nested) {
        for (Value operand : nested->getOperands()) {
          std::optional<unsigned> producer = getProducerNode(operand);
          // Values defined inside the region have no node id, so only
          // captured values of Omega(r) produce an edge.
          if (!producer || *producer == id)
            continue;
          addEdge(id, *producer, ops[id]->getNumOperands());
        }
      });
  }
  /// True when some result of `op` reaches, directly or through pure
  /// broadcasts of Omega(r), a user outside Omega(r). A PB's Outside_User is
  /// charged to the PB's input node.
  bool hasOutsideUser(Operation *op) const {
    Block *block = op->getBlock();
    for (Value result : op->getResults()) {
      for (OpOperand &use : result.getUses()) {
        Operation *user = block->findAncestorOpInBlock(*use.getOwner());
        if (!user || !omega.contains(user))
          return true;
        if (isPureBroadcast(user, tracker) && hasOutsideUser(user))
          return true;
      }
    }
    return false;
  }
  /// True when `value` derives from a function argument through ops that
  /// belong to no Scope.
  bool reachesFunctionArgument(Value value,
                               DenseSet<Operation *> &visited) const {
    if (auto argument = dyn_cast<BlockArgument>(value))
      return isa<func::FuncOp>(argument.getOwner()->getParentOp());
    Operation *producer = value.getDefiningOp();
    if (getQGEMMNodeRoot(producer) || !visited.insert(producer).second)
      return false;
    auto walkOperands = [&](Operation *op) {
      return llvm::any_of(op->getOperands(), [&](Value operand) {
        return operand != value && reachesFunctionArgument(operand, visited);
      });
    };
    if (walkOperands(producer))
      return true;
    // A buffer's contents come from the out-of-scope ops writing it, such as
    // the memref.copy that fills a memref.alloc before bufferization.to_tensor.
    if (!isa<MemRefType>(value.getType()))
      return false;
    return llvm::any_of(value.getUsers(), [&](Operation *user) {
      auto effects = dyn_cast<MemoryEffectOpInterface>(user);
      if (!effects || getQGEMMNodeRoot(user) ||
          !effects.getEffectOnValue<MemoryEffects::Write>(value))
        return false;
      if (!visited.insert(user).second)
        return false;
      return walkOperands(user);
    });
  }
  /// Boundary load: no operand produced inside Omega(r) (PBs looked through)
  /// and at least one operand deriving from a function argument through
  /// out-of-scope ops.
  bool isBoundaryLoad(Operation *op) const {
    SmallVector<Value> sources;
    for (Value operand : op->getOperands()) {
      // Scalar and index operands are not traceable: a view op whose dynamic
      // offset comes from an index_cast outside the Scope is still a
      // boundary load.
      if (!isa<TensorType, BaseMemRefType>(operand.getType()))
        continue;
      Value source = lookThroughPBs(operand);
      if (Operation *producer = source.getDefiningOp())
        if (omega.contains(producer))
          return false;
      sources.push_back(source);
    }
    DenseSet<Operation *> visited;
    return llvm::any_of(sources, [&](Value source) {
      return reachesFunctionArgument(source, visited);
    });
  }
  const SmallVector<unsigned> &getAncestorLoads(unsigned id) {
    auto it = ancestorLoads.find(id);
    if (it != ancestorLoads.end())
      return it->second;
    SmallVector<unsigned> loads;
    if (temp[id].isBoundaryLoad)
      loads.push_back(loadRank.lookup(id));
    for (unsigned input : temp[id].inputs)
      llvm::append_range(loads, getAncestorLoads(input));
    llvm::sort(loads);
    loads.erase(std::unique(loads.begin(), loads.end()), loads.end());
    return ancestorLoads[id] = std::move(loads);
  }
  /// Smallest operand-index path from the final reduction, or nullopt when
  /// the node does not reach it.
  const std::optional<SmallVector<unsigned>> &getRootPath(unsigned id) {
    auto it = rootPaths.find(id);
    if (it != rootPaths.end())
      return it->second;
    std::optional<SmallVector<unsigned>> best;
    if (ops[id] == finalReduce) {
      best.emplace();
    } else {
      for (auto [user, operandNumber] : useEdges[id]) {
        const std::optional<SmallVector<unsigned>> &userPath =
            getRootPath(user);
        if (!userPath)
          continue;
        SmallVector<unsigned> candidate = *userPath;
        candidate.push_back(operandNumber);
        if (!best || candidate < *best)
          best = std::move(candidate);
      }
    }
    return rootPaths[id] = std::move(best);
  }
  /// Kahn's algorithm with OrderKey tie-breaking, then renumbering.
  void order(SmallVector<LNode> &outNodes,
             DenseMap<Operation *, unsigned> &outIds) {
    unsigned count = ops.size();
    SmallVector<OrderKey> keys;
    keys.reserve(count);
    for (unsigned id = 0; id < count; ++id) {
      const std::optional<SmallVector<unsigned>> &path = getRootPath(id);
      keys.push_back({getAncestorLoads(id), ops[id]->getName().getStringRef(),
                      !path.has_value(), path ? *path : SmallVector<unsigned>(),
                      id});
    }
    auto compare = [&](unsigned a, unsigned b) { return keys[a] < keys[b]; };
    std::set<unsigned, decltype(compare)> ready(compare);
    SmallVector<unsigned> indegree(count);
    for (unsigned id = 0; id < count; ++id) {
      indegree[id] = temp[id].inputs.size();
      if (indegree[id] == 0)
        ready.insert(id);
    }
    SmallVector<unsigned> sorted;
    while (!ready.empty()) {
      unsigned id = *ready.begin();
      ready.erase(ready.begin());
      sorted.push_back(id);
      for (unsigned user : temp[id].users)
        if (--indegree[user] == 0)
          ready.insert(user);
    }
    assert(sorted.size() == count && "cyclic logical graph");
    SmallVector<unsigned> newId(count);
    for (auto [index, id] : llvm::enumerate(sorted))
      newId[id] = index;
    for (unsigned id : sorted) {
      LNode node = temp[id];
      for (unsigned &input : node.inputs)
        input = newId[input];
      for (unsigned &user : node.users)
        user = newId[user];
      llvm::sort(node.users);
      outIds[node.op] = outNodes.size();
      outNodes.push_back(std::move(node));
    }
  }
  const SetVector<Operation *> &omega;
  const AxisTracker &tracker;
  Operation *finalReduce;
  SmallVector<Operation *> ops;
  DenseMap<Operation *, unsigned> tempIds;
  SmallVector<LNode> temp;
  /// (user node, operand number) per producer node.
  SmallVector<SmallVector<std::pair<unsigned, unsigned>>> useEdges;
  DenseMap<unsigned, unsigned> loadRank;
  DenseMap<unsigned, SmallVector<unsigned>> ancestorLoads;
  DenseMap<unsigned, std::optional<SmallVector<unsigned>>> rootPaths;
};
/// Iteration dimension of `generic` carrying `axis`, read through the output
/// map at the axis position of the first result.
std::optional<unsigned> getAxisLoop(linalg::GenericOp generic,
                                    const AxisTracker &tracker, Axis axis) {
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
/// Req 4.2: the body reads the d coordinate through linalg.index of the d
/// loop. A tensor.extract index depending on such an index reads it too; one
/// depending on a V_d operand is covered by Req 4.1.
bool readsAxisCoordinate(Operation *op, const AxisTracker &tracker, Axis d) {
  auto generic = dyn_cast<linalg::GenericOp>(op);
  if (!generic)
    return false;
  std::optional<unsigned> loop = getAxisLoop(generic, tracker, d);
  if (!loop)
    return false;
  WalkResult walk = generic.getRegion().walk([&](linalg::IndexOp index) {
    return index.getDim() == *loop ? WalkResult::interrupt()
                                   : WalkResult::advance();
  });
  return walk.wasInterrupted();
}
/// A source node whose result is a memref, such as the memref.alloc of a
/// load chain that keeps its copy: its contents are written through side
/// effects rather than data edges, so its axes cannot be derived from its
/// inputs and are taken as varying.
bool readsMemoryContents(Operation *op) {
  return op->getNumResults() == 1 &&
         isa<BaseMemRefType>(op->getResult(0).getType());
}
} // namespace
LogicalGraph LogicalGraph::build(const SetVector<Operation *> &omega,
                                 const AxisTracker &tracker,
                                 Operation *finalReduce) {
  LogicalGraph graph;
  GraphBuilder(omega, tracker, finalReduce).run(graph.nodes, graph.nodeIds);
  for (Axis axis : kAxes)
    graph.computeVariance(axis, tracker);
  return graph;
}
/// One forward sweep in topological order: inputs are final before their
/// users, so the result is the least set closed under Req 4.1-4.4. Reads only
/// axis `d`; V_m and V_n are independent (Req 1.6).
void LogicalGraph::computeVariance(Axis d, const AxisTracker &tracker) {
  unsigned a = static_cast<unsigned>(d);
  unsigned count = nodes.size();
  variance[a] = llvm::BitVector(count);
  current[a] = llvm::BitVector(count);
  pinCandidates[a] = llvm::BitVector(count);
  for (unsigned id = 0; id < count; ++id) {
    LNode &node = nodes[id];
    if (node.hasAxis[a])
      current[a].set(id);
    bool varies =
        llvm::any_of(node.inputs,
                     [&](unsigned input) { return variance[a].test(input); }) ||
        readsAxisCoordinate(node.op, tracker, d) ||
        ((node.isBoundaryLoad || readsMemoryContents(node.op)) &&
         node.hasAxis[a]);
    if (varies) {
      variance[a].set(id);
      continue;
    }
    if (node.hasAxis[a] && !canStripAxis(node.op, d, tracker)) {
      pinCandidates[a].set(id);
      node.pinned[a] = true;
      variance[a].set(id);
    }
    // Otherwise the node is invariant along d; its d axis, if any, is
    // stripped in G(r) (Req 3.1).
  }
}
void LogicalGraph::emitR2Remarks() const {
  for (const LNode &node : nodes)
    for (Axis axis : kAxes)
      if (node.pinned[static_cast<unsigned>(axis)])
        node.op->emitRemark()
            << "qgemm-decompress-coupling: op is invariant along '"
            << getAxisName(axis) << "' but '" << getAxisName(axis)
            << "' cannot be stripped; pinned in the '" << getAxisName(axis)
            << "' space";
}
void LogicalGraph::print(llvm::raw_ostream &os) const {
  unsigned loads = 0, outside = 0;
  for (const LNode &node : nodes) {
    loads += node.isBoundaryLoad;
    outside += node.hasOutsideUser;
  }
  os << "  |nodes|=" << nodes.size() << " |V_m|=" << variance[0].count()
     << " |V_n|=" << variance[1].count() << " |C_m|=" << current[0].count()
     << " |C_n|=" << current[1].count() << " boundary-loads=" << loads
     << " outside-user-nodes=" << outside
     << " pin-candidates(m,n)=" << pinCandidates[0].count() << ","
     << pinCandidates[1].count() << "\n";
  for (auto [id, node] : llvm::enumerate(nodes)) {
    os << "    #" << id << " " << node.op->getName()
       << " V=" << (variance[0].test(id) ? "m" : "-")
       << (variance[1].test(id) ? "n" : "-")
       << " C=" << (node.hasAxis[0] ? "m" : "-")
       << (node.hasAxis[1] ? "n" : "-") << (node.isBoundaryLoad ? " load" : "")
       << (node.isMemrefView ? " memref-view" : "")
       << (node.hasOutsideUser ? " outside-user" : "") << " "
       << node.op->getLoc() << " in=[";
    llvm::interleaveComma(node.inputs, os);
    os << "]\n";
  }
}
} // namespace qgemm_coupling
} // namespace buddy
} // namespace mlir
