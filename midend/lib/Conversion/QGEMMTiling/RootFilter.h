//===- RootFilter.h - Diamond detection and candidate filtering ---*- C++
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
// This file declares the TilingCandidate struct and the filterTilingCandidates
// function, which detects diamond overlap (shared ops) between tiling
// candidates and excludes conflicting ones.
//
//===----------------------------------------------------------------------===//

#ifndef QGEMM_TILING_ROOT_FILTER_H
#define QGEMM_TILING_ROOT_FILTER_H

#include "ContractionSubgraph.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir {
namespace buddy {

/// A validated tiling candidate: a root with its successfully-collected
/// subgraph.
struct TilingCandidate {
  linalg::LinalgOp tilingRoot;
  ContractionSubgraph subgraph;
};

/// Filter candidates by removing those with diamond overlap.
///
/// Input: candidates with successfully-collected subgraphs (failed collections
///        are already excluded before calling this function).
///
/// Uses a two-pass approach:
///   Pass 1: Build a per-op reference count across all candidates' subgraphs
///           (excluding linalg::FillOp which is trivially clonable).
///   Pass 2: Any candidate whose subgraph contains an op with refcount > 1
///           is excluded (with warning diagnostic on the root).
///
/// Returns only the non-conflicting candidates.
SmallVector<TilingCandidate>
filterTilingCandidates(SmallVectorImpl<TilingCandidate> &candidates);

} // namespace buddy
} // namespace mlir

#endif // QGEMM_TILING_ROOT_FILTER_H
