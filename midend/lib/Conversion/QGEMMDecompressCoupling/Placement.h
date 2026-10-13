//===- Placement.h - Target placement for QGEMM coupling --------*- C++ -*-===//
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
// Target placement (L_m, L_n) of one QGEMM root: `pre` sets L_d = V_d,
// `on-the-fly` adds the hoist set H_d computed by one reverse-topological
// sweep over G(r) (Req 5). Phase A only; no IR is created or modified.
//
//===----------------------------------------------------------------------===//
#ifndef BUDDY_QGEMM_DECOMPRESS_COUPLING_PLACEMENT_H
#define BUDDY_QGEMM_DECOMPRESS_COUPLING_PLACEMENT_H
#include "AxisTracker.h"
#include "LogicalGraph.h"
#include "Separator.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
namespace mlir {
namespace buddy {
namespace qgemm_coupling {
/// Result of the placement step. Data only.
struct PlacementResult {
  /// L_d per axis, indexed by node id.
  llvm::BitVector lifted[2];
  /// H_d per axis (empty under `pre`).
  llvm::BitVector hoisted[2];
  /// Candidates excluded from H_d because `d` cannot be inserted (R3).
  llvm::BitVector excluded[2];
  /// pos_d per node: the position of d in the rebuilt result of a node in
  /// H_d, derived from its users (Req 6.8), in the frame of the separated
  /// input with only d changed. Unset for nodes outside H_d.
  SmallVector<std::optional<unsigned>> positions[2];
  /// R1 `inconsistent broadcast positions` diagnostics.
  SmallVector<R1Diagnostic> r1Diagnostics;
  /// Skip check (Req 6.1): target equals the current placement and the
  /// separation plan has no rewrite.
  bool unchanged = false;
  const llvm::BitVector &getLifted(Axis d) const {
    return lifted[static_cast<unsigned>(d)];
  }
  /// Emit one Remark R3 per excluded (node, d) pair (Req 5.7), gated.
  void emitR3Remarks(const LogicalGraph &graph) const;
  void print(llvm::raw_ostream &os, const LogicalGraph &graph) const;
};
/// Compute the target placement. `onTheFly` selects `on-the-fly`, otherwise
/// `pre`. Each axis is decided without reading the other's placement.
PlacementResult computePlacement(const LogicalGraph &graph,
                                 const AxisTracker &tracker,
                                 const SeparationPlan &separation,
                                 bool onTheFly);
} // namespace qgemm_coupling
} // namespace buddy
} // namespace mlir
#endif // BUDDY_QGEMM_DECOMPRESS_COUPLING_PLACEMENT_H
