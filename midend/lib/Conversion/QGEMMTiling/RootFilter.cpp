//===- RootFilter.cpp - Diamond detection and candidate filtering ---------===//
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
// This file implements the filterTilingCandidates function, which detects
// diamond overlap (shared ops) between tiling candidates and excludes
// conflicting ones with a warning diagnostic.
//
//===----------------------------------------------------------------------===//

#include "RootFilter.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::buddy;

SmallVector<TilingCandidate> mlir::buddy::filterTilingCandidates(
    SmallVectorImpl<TilingCandidate> &candidates) {
  // --- Debug-only sanity check: no tiling root should appear inside
  //     another root's subgraph. If this fires, it means the reduce chain
  //     analysis failed to correctly identify inner reduces as non-roots. ---
#ifndef NDEBUG
  {
    llvm::DenseSet<Operation *> allRoots;
    for (auto &candidate : candidates)
      allRoots.insert(candidate.tilingRoot.getOperation());
    for (auto &candidate : candidates) {
      for (Operation *op : candidate.subgraph.ops) {
        if (op == candidate.tilingRoot.getOperation())
          continue; // A root is always in its own subgraph — that's fine.
        assert(!allRoots.contains(op) &&
               "A tiling root appears inside another root's subgraph — "
               "reduce chain analysis failed to filter inner reduces");
      }
    }
  }
#endif

  // --- Pass 1: Build per-op reference counts ---
  // Exclude linalg.fill which is trivially clonable and doesn't represent
  // a real data dependency between chains. (tensor.empty is already excluded
  // from subgraphs by isBoundaryOp, so it won't appear here.)
  llvm::DenseMap<Operation *, unsigned> opRefCount;
  for (auto &candidate : candidates) {
    for (Operation *op : candidate.subgraph.ops) {
      if (isa<linalg::FillOp>(op))
        continue;
      opRefCount[op]++;
    }
  }

  // --- Pass 2: Exclude candidates with shared ops (diamond) ---
  SmallVector<TilingCandidate> result;
  for (auto &candidate : candidates) {
    bool hasConflict = false;
    for (Operation *op : candidate.subgraph.ops) {
      if (isa<linalg::FillOp>(op))
        continue;
      if (opRefCount[op] > 1) {
        hasConflict = true;
        break;
      }
    }
    if (hasConflict) {
      candidate.tilingRoot->emitWarning(
          "qgemm-tiling: subgraph shares ops with another reduce chain "
          "root (diamond); skipping");
      continue;
    }
    result.push_back(std::move(candidate));
  }

  return result;
}
