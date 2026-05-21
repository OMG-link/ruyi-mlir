//===- ContractionSubgraph.h - Collect contraction subgraph -------*- C++
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
// This file declares the ContractionSubgraph data structure and the
// collectSubgraph utility, which performs backward BFS from the tiling root
// to collect all ops in the contraction pattern, compute dimension mappings,
// and detect broadcast dimensions.
//
//===----------------------------------------------------------------------===//

#ifndef QGEMM_TILING_CONTRACTION_SUBGRAPH_H
#define QGEMM_TILING_CONTRACTION_SUBGRAPH_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
namespace buddy {

/// Per-op dimension mapping to the tiling root's iteration space.
/// dimMap[i] = j means dim i in this op's iteration space corresponds to
/// dim j in the tiling root's iteration space.
/// dimMap[i] = std::nullopt means dim i is eliminated before reaching
/// the tiling root (it is a reduction dimension handled by an inner reduce).
///
/// "Iteration space" is the domain of the op's indexing maps — the (d0, d1,
/// ..., dN) dimensions that indexing maps map from. For linalg.generic this
/// equals numLoops; for linalg.reduce it equals the input rank (parallel +
/// reduction dims); for linalg.transpose it equals the input/output rank.
using DimMapping = SmallVector<std::optional<unsigned>>;

/// The collected contraction subgraph.
struct ContractionSubgraph {
  /// The tiling root (closest LinalgOp to the store that implements
  /// TilingInterface).
  linalg::LinalgOp tilingRoot;

  /// All operations in the contraction subgraph (element-wise generics,
  /// broadcast generics, reduces, transposes, fills, empty tensors).
  /// Used by fusionControlFn to decide what to fuse.
  llvm::SetVector<Operation *> ops;

  /// All linalg.reduce ops in the subgraph, ordered from innermost to
  /// outermost. Useful for future K-dimension tiling.
  SmallVector<linalg::ReduceOp> reduces;

  /// Per-op dimension mapping to the tiling root's iteration space.
  /// Computed incrementally during BFS traversal.
  llvm::DenseMap<Operation *, DimMapping> dimMappings;

  /// Per-dim broadcast detection for the tiling root's output dimensions.
  /// hasBroadcast[d] = true if at least one input in the subgraph is
  /// broadcast on output dim d.
  SmallVector<bool> hasBroadcast;
};

/// Collect the contraction subgraph backward from the tiling root.
/// Returns failure if the subgraph is inconsistent. On failure, a remark
/// diagnostic is emitted on the offending op with the specific reason.
FailureOr<ContractionSubgraph> collectSubgraph(linalg::LinalgOp tilingRoot);

} // namespace buddy
} // namespace mlir

#endif // QGEMM_TILING_CONTRACTION_SUBGRAPH_H
