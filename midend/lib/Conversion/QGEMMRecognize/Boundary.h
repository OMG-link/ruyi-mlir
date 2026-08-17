//===- Boundary.h - QGEMM operator boundary predicate ---------*- C++ -*-===//
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
// This file declares the shared upstream boundary predicate used by QGEMM
// recognition traversals.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_MLIR_QGEMM_BOUNDARY_H
#define BUDDY_MLIR_QGEMM_BOUNDARY_H

namespace mlir {
class Operation;

namespace buddy {

/// Return true when an operation is outside the traversable QGEMM subgraph.
///
/// The admitted operation set is the canonical-form Linalg/tensor shape set
/// plus arithmetic operations whose results are ranked tensors. Operations
/// without ranked tensor results are boundaries. tensor.empty is deliberately
/// a boundary because it supplies a destination rather than computation.
bool isQGEMMOperatorBoundary(Operation *op);

} // namespace buddy
} // namespace mlir

#endif // BUDDY_MLIR_QGEMM_BOUNDARY_H
