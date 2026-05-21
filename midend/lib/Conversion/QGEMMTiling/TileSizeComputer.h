//===- TileSizeComputer.h - Compute tile sizes for M/N dims -------*- C++
//-*-===//
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
// This file declares the computeTileSizes utility, which maps dimension roles
// (M, N, Batch, Reduction) to concrete tile sizes based on pass parameters
// and static dimension extents.
//
//===----------------------------------------------------------------------===//

#ifndef QGEMM_TILING_TILE_SIZE_COMPUTER_H
#define QGEMM_TILING_TILE_SIZE_COMPUTER_H

#include "DimensionClassifier.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace buddy {

/// Compute the tile size vector for the tiling root.
///
/// This is the top-level entry point for Step 4. It:
/// 1. Gets the iteration space sizes from the tiling root (via
///    getStaticLoopRanges).
/// 2. Checks that M/N dimensions to be tiled are static (emits remark and
///    returns failure if not).
/// 3. Computes tile sizes based on dimension roles and pass parameters.
/// 4. Returns failure if all tile sizes are 0 (nothing to tile).
///
/// Returns the tile size vector on success, or failure to indicate bail-out.
FailureOr<SmallVector<OpFoldResult>>
computeTileSizes(linalg::LinalgOp tilingRoot,
                 const DimensionClassification &classification, int64_t tileM,
                 int64_t tileN);

} // namespace buddy
} // namespace mlir

#endif // QGEMM_TILING_TILE_SIZE_COMPUTER_H
