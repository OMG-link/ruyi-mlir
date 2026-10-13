//===- LoadChain.h - QGEMM operand load chain walk -------------*- C++ -*-===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
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
// This file declares the one membership predicate for an operand's memref
// source chain.  The dimension-transform pass uses it to build its coarse
// scope and the recognizer uses it to bring the chain into the operator, so
// the two definitions cannot drift apart.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_MLIR_QGEMM_LOAD_CHAIN_H
#define BUDDY_MLIR_QGEMM_LOAD_CHAIN_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace mlir {
class Operation;

namespace buddy {

/// Visitation state of a load-chain walk.  Several boundaries of the same
/// function share one state so that a chain op reached twice is recorded
/// once.
struct QGEMMLoadChainState {
  llvm::DenseSet<Value> visitedValues;
  llvm::DenseSet<Operation *> visitedOperations;
};

/// Follow the memory-side source path hidden behind a tensor boundary.
///
/// OperatorWalk intentionally stops at bufferization.to_tensor and memref
/// results. The layout expression for an input is nevertheless part of the
/// operator's data preparation. In the captured IR the source path commonly
/// has the form
///
///   memref.reinterpret_cast -> memref.copy -> memref.alloc
///       -> bufferization.to_tensor
///
/// so the walk below follows the defining operands and, for an allocation,
/// the copy that fills it. It only crosses the source-producing memref ops
/// listed here; arbitrary users and unrelated memory computations remain out
/// of scope, as do the scalar, index and memref.dim operations that compute a
/// view's dynamic offsets and sizes.
///
/// `isAdmitted` restricts the walk to the ops the caller owns (in practice
/// the ops of one function).  `record` receives every chain op exactly once.
void collectQGEMMLoadChain(Value value, Operation *producer,
                           llvm::function_ref<bool(Operation *)> isAdmitted,
                           llvm::function_ref<void(Operation *)> record,
                           QGEMMLoadChainState &state);

} // namespace buddy
} // namespace mlir

#endif // BUDDY_MLIR_QGEMM_LOAD_CHAIN_H
