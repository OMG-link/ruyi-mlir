//===- LogicalGraph.h - Logical graph and variance for QGEMM ----*- C++ -*-===//
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
// The logical graph G(r) is the separated Omega(r) with every pure broadcast
// erased and its users connected directly to its input. This file builds
// G(r) in a deterministic topological order, detects boundary loads and
// Outside_Users, and computes for each parallel axis d the set V_d of nodes
// that vary along d together with the current placement C_d.
//
//===----------------------------------------------------------------------===//
#ifndef BUDDY_QGEMM_DECOMPRESS_COUPLING_LOGICAL_GRAPH_H
#define BUDDY_QGEMM_DECOMPRESS_COUPLING_LOGICAL_GRAPH_H
#include "AxisTracker.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
namespace mlir {
namespace buddy {
namespace qgemm_coupling {
/// One node of G(r): a non-pure-broadcast op of Omega(r).
struct LNode {
  /// Representative op in the separated plan.
  Operation *op;
  /// Carries m / n in the separated plan (current placement).
  bool hasAxis[2] = {false, false};
  /// Pinned in V_d by Req 3.2 (Remark R2).
  bool pinned[2] = {false, false};
  bool isBoundaryLoad = false;
  /// A memref view op. A Boundary_Load, but an H_d candidate all the same,
  /// because folding a parallel dimension into it gives that dimension
  /// stride 0 and reads nothing extra (Req 5.2).
  bool isMemrefView = false;
  /// Node ids of the producers of this node's operands, pure broadcasts
  /// looked through, unique, in operand order.
  SmallVector<unsigned> inputs;
  /// Node ids of this node's users, pure broadcasts looked through, unique.
  SmallVector<unsigned> users;
  /// Some result, possibly through pure broadcasts, is used by an
  /// Outside_User.
  bool hasOutsideUser = false;
};
/// Whether the rule table can strip axis `d` from `op`.
bool canStripAxis(Operation *op, Axis d, const AxisTracker &tracker);
/// G(r) together with V_d and C_d for both parallel axes.
class LogicalGraph {
public:
  LogicalGraph() = default;
  /// Build G(r) over `omega` (Omega(r) in block order) and compute V_m, V_n
  /// and C_m, C_n. Reads no stage field and no operand role.
  static LogicalGraph build(const SetVector<Operation *> &omega,
                            const AxisTracker &tracker, Operation *finalReduce);
  /// Emit one Remark R2 per pinned (node, d) pair (Req 3.2).
  void emitR2Remarks() const;
  /// Nodes in canonical topological order; node ids index this vector.
  ArrayRef<LNode> getNodes() const { return nodes; }
  std::optional<unsigned> getNodeId(Operation *op) const {
    auto it = nodeIds.find(op);
    if (it == nodeIds.end())
      return std::nullopt;
    return it->second;
  }
  /// V_d (Req 4).
  const llvm::BitVector &getVariance(Axis d) const {
    return variance[static_cast<unsigned>(d)];
  }
  /// C_d: nodes whose result type currently carries d.
  const llvm::BitVector &getCurrentPlacement(Axis d) const {
    return current[static_cast<unsigned>(d)];
  }
  /// Nodes that would have been pinned in V_d by R2 if pinning were enabled.
  const llvm::BitVector &getPinCandidates(Axis d) const {
    return pinCandidates[static_cast<unsigned>(d)];
  }
  /// Count summary on one line plus one line per node.
  void print(llvm::raw_ostream &os) const;

private:
  void computeVariance(Axis d, const AxisTracker &tracker);
  SmallVector<LNode> nodes;
  DenseMap<Operation *, unsigned> nodeIds;
  llvm::BitVector variance[2];
  llvm::BitVector current[2];
  llvm::BitVector pinCandidates[2];
};
} // namespace qgemm_coupling
} // namespace buddy
} // namespace mlir
#endif // BUDDY_QGEMM_DECOMPRESS_COUPLING_LOGICAL_GRAPH_H
