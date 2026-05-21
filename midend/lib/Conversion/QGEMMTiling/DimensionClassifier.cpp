//===- DimensionClassifier.cpp - Classify M/N/Batch dimensions ----*- C++
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
// This file implements the DimensionClassifier utility. It examines the
// hasBroadcast array from a ContractionSubgraph to determine which output
// dimensions of the tiling root correspond to M, N, and Batch roles.
//
//===----------------------------------------------------------------------===//

#include "DimensionClassifier.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "qgemm-tiling"

using namespace mlir;
using namespace mlir::buddy;

FailureOr<DimensionClassification>
mlir::buddy::classifyDimensions(const ContractionSubgraph &subgraph) {
  // Collect indices of dimensions that have broadcast inputs.
  SmallVector<unsigned> broadcastDims;
  for (unsigned d = 0, e = subgraph.hasBroadcast.size(); d < e; ++d) {
    if (subgraph.hasBroadcast[d])
      broadcastDims.push_back(d);
  }

  unsigned numBroadcast = broadcastDims.size();

  // Case 0: No broadcast detected — not a dot-like structure.
  // Silent bail out (no diagnostic).
  if (numBroadcast == 0)
    return failure();

  // Case 1: Only one dimension has broadcast — cannot determine M/N pair.
  if (numBroadcast == 1) {
    LLVM_DEBUG(llvm::dbgs()
               << "qgemm-tiling: only one broadcast dimension (dim "
               << broadcastDims[0]
               << ") detected; cannot determine M/N pair\n");
    return failure();
  }

  // Case 2: Exactly two broadcast dimensions — valid dot structure.
  if (numBroadcast == 2) {
    unsigned outputRank = subgraph.hasBroadcast.size();
    SmallVector<DimRole> roles(outputRank, DimRole::Batch);
    roles[broadcastDims[0]] = DimRole::M;
    roles[broadcastDims[1]] = DimRole::N;

    LLVM_DEBUG(llvm::dbgs() << "DimensionClassification: M=" << broadcastDims[0]
                            << " N=" << broadcastDims[1] << "\n");

    DimensionClassification result;
    result.roles = std::move(roles);
    result.mDim = broadcastDims[0];
    result.nDim = broadcastDims[1];
    return result;
  }

  // Case >2: More than two broadcast dims — violates single-M single-N
  // assumption. Emit remark diagnostic on the tiling root.
  subgraph.tilingRoot->emitRemark()
      << "qgemm-tiling: found " << numBroadcast
      << " broadcast dimensions; expected exactly 2 (M and N)";
  return failure();
}
