//===- FinalConsumerFinder.cpp - Find tiling root ---------------*- C++ -*-===//
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
// This file implements the FinalConsumerFinder utility. It walks a FuncOp for
// bufferization::MaterializeInDestinationOp ops and traces the source value
// backward to find the first op that implements TilingInterface.
//
//===----------------------------------------------------------------------===//

#include "FinalConsumerFinder.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "qgemm-tiling"

using namespace mlir;

namespace mlir {
namespace buddy {

std::optional<FinalConsumerInfo> findFinalConsumer(func::FuncOp funcOp) {
  std::optional<FinalConsumerInfo> result;

  funcOp.walk([&](bufferization::MaterializeInDestinationOp matOp) {
    // Already found a valid match — skip further processing.
    if (result.has_value())
      return WalkResult::interrupt();

    // Get the source value of the materialize_in_destination op.
    Value source = matOp.getSource();

    // Trace backward through the source value to find the first op that
    // implements TilingInterface. If the source is directly produced by a
    // TilingInterface op, that's the root. Otherwise, walk through
    // non-TilingInterface ops (unlikely but handled for robustness).
    Value current = source;
    while (current) {
      Operation *defOp = current.getDefiningOp();
      if (!defOp)
        break;

      if (isa<linalg::LinalgOp>(defOp)) {
        LLVM_DEBUG(llvm::dbgs() << "Found tiling root: " << *defOp << "\n");
        result = FinalConsumerInfo{cast<linalg::LinalgOp>(defOp)};
        return WalkResult::interrupt();
      }

      // If the op has exactly one operand that is a tensor result, trace
      // through it. This handles non-TilingInterface ops that sit between
      // the tiling root and the materialize_in_destination (unlikely but
      // possible).
      Value nextValue = nullptr;
      for (Value operand : defOp->getOperands()) {
        if (isa<RankedTensorType>(operand.getType())) {
          nextValue = operand;
          break;
        }
      }
      current = nextValue;
    }

    return WalkResult::advance();
  });

  LLVM_DEBUG({
    if (!result.has_value())
      llvm::dbgs() << "No tiling root found in function\n";
  });

  return result;
}

} // namespace buddy
} // namespace mlir
