//===- OperatorWalk.h - QGEMM backwards boundary traversal -----*- C++ -*-===//
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
// This file declares the backwards traversal from a candidate QGEMM reduction
// to its upstream boundaries.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_MLIR_QGEMM_OPERATOR_WALK_H
#define BUDDY_MLIR_QGEMM_OPERATOR_WALK_H

#include "CandidateRoots.h"

#include "mlir/IR/Value.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class Operation;

namespace buddy {

enum class QGEMMBoundaryKind {
  BlockArgument,
  CrossBlockValue,
  BufferizationToTensor,
  MemrefResult,
  NotAdmitted,
};

/// A value at which the backwards walk stops.
struct QGEMMBoundary {
  Value value;
  Operation *producer;
  QGEMMBoundaryKind kind;
};

/// The portion of the candidate DAG reached by the backwards walk.
///
/// `operations` contains admitted computational operations exactly once and
/// in first-visit order. `boundaries` contains the values at which traversal
/// stops, each with its QGEMMBoundaryKind.
struct QGEMMOperatorWalk {
  QGEMMCandidateRoot candidate;
  llvm::SetVector<Operation *> operations;
  llvm::SmallVector<QGEMMBoundary> boundaries;
};

/// Walk one candidate reduction backwards to its upstream boundaries.
QGEMMOperatorWalk walkQGEMMOperator(QGEMMCandidateRoot candidate);

} // namespace buddy
} // namespace mlir

#endif // BUDDY_MLIR_QGEMM_OPERATOR_WALK_H
