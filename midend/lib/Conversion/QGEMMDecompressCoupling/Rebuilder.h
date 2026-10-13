//===- Rebuilder.h - Rebuild of one QGEMM root in its target placement ----===//
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
// Phase B rebuild of one QGEMM root (Req 6.2-6.5, 6.7, 6.8): every node of
// G(r) is cloned in canonical topological order with exactly the parallel
// axes of its placement, the separation plan is applied on the way, pure
// broadcasts are inserted and reused on frontier edges, and Outside_Users are
// rewired to the rebuilt values with their original operand types. The
// original ops are left in place; erasure happens afterwards.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_QGEMM_DECOMPRESS_COUPLING_REBUILDER_H
#define BUDDY_QGEMM_DECOMPRESS_COUPLING_REBUILDER_H

#include "AxisTracker.h"
#include "LogicalGraph.h"
#include "Placement.h"
#include "Separator.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace buddy {
namespace qgemm_coupling {

/// What the rebuild created, for annotation writing and erasure.
struct RebuildResult {
  /// Rebuilt op per G(r) node id.  A node kept in place is its own entry:
  /// it is reused in place rather than cloned.
  SmallVector<Operation *> clones;
  /// Nodes reused in place (memref nodes whose layout is unchanged), in node
  /// order.
  SmallVector<Operation *> keptOps;
  /// Pure broadcasts created by the rebuild, in creation order.
  SmallVector<Operation *> createdPBs;
  /// tensor.empty / linalg.fill / tensor.collapse_shape helpers created as
  /// inits, broadcast outputs or unit-axis drops, in creation order.
  SmallVector<Operation *> createdHelpers;
};

/// Rebuild Omega(r) in the placement of `placement`. `omega` is Omega(r) in
/// block order; the other arguments are the Phase A results of the root.
RebuildResult rebuildRoot(const SetVector<Operation *> &omega,
                          const AxisTracker &tracker,
                          const SeparationPlan &separation,
                          const LogicalGraph &graph,
                          const PlacementResult &placement);

/// AnnotationWriter (Req 8.1-8.5, 1.10). Writes `qgemm.node = {root =
/// rootId, stage}` on every rebuilt op (`compute` iff its node is in
/// L_m and L_n, `data_prep` otherwise) and on every created pure broadcast
/// (`compute`), and `qgemm.root = {m_dim, n_dim}` on the rebuilt final
/// reduction only; stale `qgemm.root` copied onto other clones is removed.
/// Created tensor.empty / linalg.fill helpers carry no `qgemm.node`. A
/// created unit-axis tensor.collapse_shape is a data transform on an operand
/// edge of a rule clone, so it is annotated as part of Omega'(r) with the
/// stage of its consumer, which keeps it inside the same tiling decision as
/// that consumer.
void writeAnnotations(const RebuildResult &rebuild, const LogicalGraph &graph,
                      const PlacementResult &placement, Operation *finalReduce,
                      unsigned rootId, unsigned mDim, unsigned nDim);

/// Eraser (Req 6.6, 1.5). Erases the original ops of `omega` (including the
/// input's pure broadcasts) in reverse block order, which is a reverse
/// topological order since Omega(r) lives in one block; every one of them
/// must be unused by now. Created ops of `rebuild` left without users are
/// erased as well. A worklist then erases `tensor.empty`, `linalg.fill` and
/// ConstantLike ops outside every Scope (no `qgemm.node`) that lost their last
/// user, re-queuing their operands' producers, to a fixed point.
void eraseOriginals(const SetVector<Operation *> &omega,
                    const RebuildResult &rebuild);
} // namespace qgemm_coupling
} // namespace buddy
} // namespace mlir

#endif // BUDDY_QGEMM_DECOMPRESS_COUPLING_REBUILDER_H
