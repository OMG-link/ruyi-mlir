//===- TilingExecutor.h - Execute tiling via SCF infrastructure ---*- C++
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
// This file declares the executeTiling utility, which applies tiling to the
// tiling root using MLIR's SCF tile-and-fuse infrastructure and fuses all
// producers within the contraction subgraph.
//
//===----------------------------------------------------------------------===//

#ifndef QGEMM_TILING_TILING_EXECUTOR_H
#define QGEMM_TILING_TILING_EXECUTOR_H

#include "ContractionSubgraph.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace buddy {

/// Execute tiling on the tiling root, fusing all producers in the subgraph.
///
/// This configures SCFTileAndFuseOptions with:
/// - The provided tile sizes
/// - ForOp loop type
/// - A fusionControlFn that fuses only producers within subgraph.ops
///
/// On success, replaces all uses of the original consumer's results with the
/// tiled replacements and erases the original tiling root if it has no
/// remaining uses.
///
/// Crashes (llvm_unreachable) if the tiling utility returns failure — this
/// indicates a compiler bug since all preconditions are validated before
/// calling this function.
void executeTiling(linalg::LinalgOp tilingRoot,
                   const ContractionSubgraph &subgraph,
                   ArrayRef<OpFoldResult> tileSizes);

} // namespace buddy
} // namespace mlir

#endif // QGEMM_TILING_TILING_EXECUTOR_H
