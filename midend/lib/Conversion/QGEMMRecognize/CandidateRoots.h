//===- CandidateRoots.h - QGEMM candidate root enumeration ------*- C++ -*-===//
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
// This file declares the reverse-scan enumeration of candidate QGEMM
// reduction roots.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_MLIR_QGEMM_CANDIDATE_ROOTS_H
#define BUDDY_MLIR_QGEMM_CANDIDATE_ROOTS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace buddy {

/// A candidate final reduction and its stable per-function IR-order number.
struct QGEMMCandidateRoot {
  linalg::ReduceOp reduce;
  unsigned index;
};

/// Enumerate additive reduction roots in a function.
///
/// A candidate is an additive linalg.reduce whose result does not reach any
/// other additive linalg.reduce in the same block through the supported
/// non-boundary subgraph. Results are ordered in forward IR order, and their
/// indices consequently start at zero for each function.
SmallVector<QGEMMCandidateRoot> findQGEMMCandidateRoots(func::FuncOp funcOp);

} // namespace buddy
} // namespace mlir

#endif // BUDDY_MLIR_QGEMM_CANDIDATE_ROOTS_H
