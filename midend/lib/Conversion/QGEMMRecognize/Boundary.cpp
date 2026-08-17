//===- Boundary.cpp - QGEMM operator boundary predicate -------*- C++ -*-===//
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
// This file implements the shared upstream boundary predicate used by QGEMM
// recognition traversals.
//
//===----------------------------------------------------------------------===//

#include "Boundary.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace mlir {
namespace buddy {
namespace {

static bool hasOnlyRankedTensorResults(Operation *op) {
  if (op->getNumResults() == 0)
    return false;
  return llvm::all_of(op->getResultTypes(),
                      [](Type type) { return isa<RankedTensorType>(type); });
}

static bool isAdmittedQGEMMOperation(Operation *op) {
  if (isa<linalg::GenericOp, linalg::ReduceOp, linalg::TransposeOp,
          linalg::FillOp, tensor::ExpandShapeOp, tensor::CollapseShapeOp,
          tensor::ReshapeOp, tensor::ExtractSliceOp, tensor::InsertSliceOp>(
          op))
    return true;

  // Arith operations are elementwise for tensor operands/results. Matching by
  // dialect keeps this list open to all current and future standalone tensor-
  // level arith operations without admitting their scalar index plumbing.
  return isa<arith::ArithDialect>(op->getDialect());
}

} // namespace

bool isQGEMMOperatorBoundary(Operation *op) {
  if (!op || isa<tensor::EmptyOp>(op))
    return true;
  if (!hasOnlyRankedTensorResults(op))
    return true;
  return !isAdmittedQGEMMOperation(op);
}

} // namespace buddy
} // namespace mlir
