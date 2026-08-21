//===- Recognition.h - QGEMM recognition model ------------------*- C++ -*-===//
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
// This file declares the recognition result built from OperatorWalk.  The
// result records scope, upstream boundaries and the positional stage split;
// it deliberately does not encode the meaning of element-wise operations
// between reductions.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_MLIR_QGEMM_RECOGNITION_H
#define BUDDY_MLIR_QGEMM_RECOGNITION_H

#include "OperatorWalk.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"

#include <optional>

namespace mlir {
namespace buddy {

/// The recognition result for one candidate reduction root.
struct Recognition {
  linalg::ReduceOp finalReduce;
  llvm::SetVector<Operation *> ops;
  llvm::SetVector<Operation *> boundaries;
  llvm::DenseSet<Operation *> dataPrepOps;
  unsigned mDim = 0;
  unsigned nDim = 1;
  llvm::DenseMap<Operation *, StringRef> stages;
};

/// The first failed legality condition for one candidate.
struct Rejection {
  Operation *at;
  StringRef reason;
  unsigned section;
};

/// Return true for a named reduction or for a linalg.generic carrying a
/// reduction iterator.  The latter is the form produced when a non-root
/// reduction is rebuilt by dimension-transform simplification.
bool isQGEMMReduction(Operation *op);

/// Build scope, boundary and stage information from one OperatorWalk result.
Recognition buildQGEMMRecognition(const QGEMMOperatorWalk &walk);

/// Check the recognition conditions owned by this pass.
///
/// The three per-candidate checks are evaluated here in their fixed order.
/// The cross-candidate ownership check is applied by the pass after all
/// candidate records have been reconstructed.
std::optional<Rejection> checkQGEMMLegality(const Recognition &recognition);

} // namespace buddy
} // namespace mlir

#endif // BUDDY_MLIR_QGEMM_RECOGNITION_H
