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
// annotated tiling root using MLIR's SCF tile-and-fuse infrastructure.
//
//===----------------------------------------------------------------------===//

#ifndef QGEMM_TILING_TILING_EXECUTOR_H
#define QGEMM_TILING_TILING_EXECUTOR_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir {
namespace buddy {

/// Execute tiling on the annotated tiling root, fusing only producers that
/// carry the same qgemm.node root identity. A tensor-semantics linalg.fill
/// reached through a destination/outs edge is the sole root-neutral exception.
///
/// On success, replaces all uses of the original consumer's results with the
/// tiled replacements and erases the original tiling root if it has no
/// remaining uses.
///
/// Crashes (llvm_unreachable) if the tiling utility returns failure — this
/// indicates a compiler bug since all preconditions are validated before
/// calling this function.
void executeTiling(linalg::LinalgOp tilingRoot, unsigned rootId,
                   ArrayRef<OpFoldResult> tileSizes);

} // namespace buddy
} // namespace mlir

#endif // QGEMM_TILING_TILING_EXECUTOR_H
