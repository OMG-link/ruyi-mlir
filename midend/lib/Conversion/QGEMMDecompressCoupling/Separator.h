//===- Separator.h - Separation plan for QGEMM coupling ---------*- C++ -*-===//
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
// The separator finds every introduction of a parallel dimension m or n inside
// Omega(r) and records how it is turned into pure broadcasts. It produces a
// plan only; no IR is created or modified (Phase A).
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_QGEMM_DECOMPRESS_COUPLING_SEPARATOR_H
#define BUDDY_QGEMM_DECOMPRESS_COUPLING_SEPARATOR_H

#include "AxisTracker.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace buddy {
namespace qgemm_coupling {

/// Printable name of a parallel axis ("m" or "n").
StringRef getAxisName(Axis axis);

/// Returns the axis that `op` inserts when it is a pure broadcast (generic
/// form or single-dimension linalg.broadcast form), or std::nullopt when `op`
/// is a business op.
std::optional<Axis> getPureBroadcastAxis(Operation *op,
                                         const AxisTracker &tracker);

/// True when `op` is a pure broadcast inserting exactly one tracked m or n.
inline bool isPureBroadcast(Operation *op, const AxisTracker &tracker) {
  return getPureBroadcastAxis(op, tracker).has_value();
}

/// Why an introduction cannot be separated (Remark R1).
enum class UnsupportedForm {
  NonProjectedPermutationMap,
  NonZeroConstantMapResult,
  NonGenericIntroducer,
  InconsistentBroadcastPositions,
  UnsupportedForm,
};

StringRef getUnsupportedFormName(UnsupportedForm form);

/// One introduction of m and/or n on the edge (operand value, consumer) of a
/// business linalg.generic, planned as a chain of pure broadcasts.
struct Introduction {
  Operation *consumer;
  unsigned operandNumber;
  /// Inserted axes in chain order: m before n (Req 2.2).
  SmallVector<Axis, 2> axes;
  /// Positions of the operand type's unit axes read through a constant-0 map
  /// result; dropped by one tensor.collapse_shape before the broadcasts
  /// (Req 2.4).
  SmallVector<unsigned, 2> unitDrops;
  /// Operand indexing map after the rewrite: constant-0 results dropped and
  /// each inserted axis read at its Broadcast_Position (Req 2.6).
  AffineMap newOperandMap;
  /// Broadcast_Position of each axis of `axes` in the operand the consumer
  /// reads after the rewrite, i.e. in the result of the last PB of the chain.
  SmallVector<unsigned, 2> positions;
  /// True when `newOperandMap` is the identity over the iteration domain
  /// (Req 2.3).
  bool identityOperandMap;
};

/// One Remark R1 to emit on `op`.
struct R1Diagnostic {
  Operation *op;
  Axis axis;
  unsigned operandNumber;
  UnsupportedForm form;
};

/// Result of separating Omega(r). Data only.
struct SeparationPlan {
  SmallVector<Introduction> introductions;
  /// Pure broadcasts already present in the input, in Omega(r) order.
  SmallVector<Operation *> existingPBs;
  /// Number of separation rewrites: business-op introductions plus
  /// single-dimension linalg.broadcast ops (rebuilt as generics).
  unsigned rewrites = 0;
  SmallVector<R1Diagnostic> r1Diagnostics;

  bool hasR1() const { return !r1Diagnostics.empty(); }
  void print(llvm::raw_ostream &os) const;
};

/// Compute the separation plan of Omega(r). Does not touch IR.
SeparationPlan computeSeparationPlan(const SetVector<Operation *> &omega,
                                     const AxisTracker &tracker);

/// Indexing maps of `op` after separation: each introduction's operand map
/// replaced by its new map. Empty when `op` is not a linalg.generic.
SmallVector<AffineMap> getSeparatedMaps(Operation *op,
                                        const SeparationPlan &plan);

/// Emit every R1 of `plan` on its op (Req 2.5).
void emitR1Remarks(const SeparationPlan &plan, unsigned rootId);

} // namespace qgemm_coupling
} // namespace buddy
} // namespace mlir

#endif // BUDDY_QGEMM_DECOMPRESS_COUPLING_SEPARATOR_H
