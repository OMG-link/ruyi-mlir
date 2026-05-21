//===- FinalConsumerFinder.h - Find tiling root -----------------*- C++ -*-===//
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
// This file declares the FinalConsumerFinder utility, which locates the tiling
// root — the op closest to the output store that implements TilingInterface.
//
//===----------------------------------------------------------------------===//

#ifndef QGEMM_TILING_FINAL_CONSUMER_FINDER_H
#define QGEMM_TILING_FINAL_CONSUMER_FINDER_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Operation.h"
#include <optional>

namespace mlir {
namespace buddy {

/// Result of final consumer search.
struct FinalConsumerInfo {
  /// The LinalgOp closest to the store that implements TilingInterface.
  /// Guaranteed to be a valid LinalgOp (generic, reduce, etc.).
  linalg::LinalgOp tilingRoot;
};

/// Find the tiling root in a FuncOp.
///
/// Walks the FuncOp for `bufferization::MaterializeInDestinationOp` ops and
/// traces the source value backward to find the first op that implements
/// `TilingInterface`.
///
/// Returns std::nullopt if no eligible pattern is found.
std::optional<FinalConsumerInfo> findFinalConsumer(func::FuncOp funcOp);

} // namespace buddy
} // namespace mlir

#endif // QGEMM_TILING_FINAL_CONSUMER_FINDER_H
