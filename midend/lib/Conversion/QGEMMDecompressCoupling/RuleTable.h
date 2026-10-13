//===- RuleTable.h - Per-op axis rules for QGEMM coupling -------*- C++ -*-===//
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
// The rule table answers, per op kind, whether a parallel axis m or n can be
// stripped from or inserted into an op, and rebuilds the op with exactly the
// axes of a target layout (Req 7).
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_QGEMM_DECOMPRESS_COUPLING_RULE_TABLE_H
#define BUDDY_QGEMM_DECOMPRESS_COUPLING_RULE_TABLE_H

#include "AxisTracker.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"

#include <optional>

namespace mlir {
namespace buddy {
namespace qgemm_coupling {

/// Positions `m` and `n` occupy in a value's type, or none.
struct AxisLayout {
  std::optional<unsigned> position[2];

  std::optional<unsigned> get(Axis axis) const {
    return position[static_cast<unsigned>(axis)];
  }
  void set(Axis axis, std::optional<unsigned> pos) {
    position[static_cast<unsigned>(axis)] = pos;
  }
  bool operator==(const AxisLayout &other) const {
    return position[0] == other.position[0] && position[1] == other.position[1];
  }

  /// Layout of `value` as read from the axis tracker.
  static AxisLayout of(Value value, const AxisTracker &tracker);
  /// Layout of a rebuilt `value` that keeps the axes of `keep` it already
  /// carries and strips the others, with no axis inserted: every kept axis
  /// stays between the same dims as in the input.
  static AxisLayout kept(Value value, bool keepM, bool keepN,
                         const AxisTracker &tracker);
  /// `layout` with `axis` inserted at `position`: axes at or after it shift.
  AxisLayout withInserted(Axis axis, unsigned position) const;
  /// `layout` with `axis` removed: axes after it shift back.
  AxisLayout without(Axis axis) const;
};

/// Per-op-kind rules. One explicit TypeSwitch case per supported kind;
/// unsupported kinds answer false so the caller raises R2 / R3 (Req 7.6).
class RuleTable {
public:
  explicit RuleTable(const AxisTracker &tracker) : tracker(tracker) {}

  /// Whether `axis` can be removed from `op`'s result. `axis` must be a
  /// parallel dimension of `op` (asserted where the kind has iterators).
  bool canStrip(Operation *op, Axis axis) const;
  /// Whether `op` can carry `axis` in L_d. An op that already carries it
  /// keeps its existing position, which every supported kind can carry.
  /// Otherwise `axis` is inserted at `position` of the result, the position
  /// derived from the op's consumers (Req 6.8), and the kind must support an
  /// axis there.
  bool canInsert(Operation *op, Axis axis, unsigned position) const;

  /// Layout operand `operandNumber` of `op` must carry when `op`'s result
  /// is rebuilt with layout `target` (Req 6.8): the per-kind position rule
  /// of the design's Axis position table. `maps` are the indexing maps of a
  /// linalg.generic after separation; ignored for other kinds. Returns
  /// std::nullopt for an operand that carries no tracked axis (a memref
  /// base, a reshape shape operand) and for a kind without a rule.
  std::optional<AxisLayout> operandLayout(Operation *op, unsigned operandNumber,
                                          AxisLayout target,
                                          ArrayRef<AffineMap> maps) const;

  /// Rebuild `op` so its result carries exactly the axes of `target`.
  /// `newOperands` replace `op`'s operands one-for-one and carry the layouts
  /// `operandLayouts`. Only called with layout changes the kind answered
  /// true for; any other request is a caller bug (llvm_unreachable).
  Operation *rebuild(OpBuilder &builder, Operation *op,
                     ArrayRef<Value> newOperands,
                     ArrayRef<AxisLayout> operandLayouts,
                     AxisLayout target) const;

  /// Static extent of `axis` in the final reduction's result, or nullopt.
  std::optional<int64_t> getAxisSize(Axis axis) const;

private:
  const AxisTracker &tracker;
};

} // namespace qgemm_coupling
} // namespace buddy
} // namespace mlir

#endif // BUDDY_QGEMM_DECOMPRESS_COUPLING_RULE_TABLE_H
