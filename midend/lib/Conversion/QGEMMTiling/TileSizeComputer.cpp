//===- TileSizeComputer.cpp - Compute tile sizes for M/N dims -----*- C++
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
// This file implements the computeTileSizes utility. Given the dimensions
// recorded on an annotated QGEMM root, it produces a tile size vector suitable
// for passing to tileConsumerAndFuseProducersUsingSCF.
//
//===----------------------------------------------------------------------===//

#include "TileSizeComputer.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "qgemm-tiling"

using namespace mlir;
using namespace mlir::buddy;

FailureOr<SmallVector<OpFoldResult>>
mlir::buddy::computeTileSizes(linalg::LinalgOp tilingRoot, unsigned mDim,
                              unsigned nDim, int64_t tileM, int64_t tileN,
                              bool explain) {
  SmallVector<int64_t> loopRanges = tilingRoot.getStaticLoopRanges();
  unsigned numDims = loopRanges.size();

  // Malformed hand-written annotations are not recognition input. A tiling
  // consumer must skip them rather than trying to reconstruct recognition.
  if (mDim >= numDims || nDim >= numDims)
    return failure();

  // Bail out if a dimension we intend to tile is dynamic. Diagnostics for
  // this expected skip are explanatory and therefore opt-in.
  if ((tileM != 0 && ShapedType::isDynamic(loopRanges[mDim])) ||
      (tileN != 0 && ShapedType::isDynamic(loopRanges[nDim]))) {
    if (explain)
      tilingRoot->emitRemark(
          "qgemm-tiling: dynamic M/N dimension in tiling root; skipping");
    return failure();
  }

  // Compute tile sizes. Every dimension other than the annotated M/N pair is
  // intentionally left untiled, including reduction and batch dimensions.
  OpBuilder builder(tilingRoot->getContext());
  SmallVector<OpFoldResult> tileSizes(numDims);

  for (unsigned i = 0; i < numDims; ++i) {
    int64_t tileSize = 0;
    if (i == mDim)
      tileSize = (loopRanges[i] <= tileM) ? 0 : tileM;
    else if (i == nDim)
      tileSize = (loopRanges[i] <= tileN) ? 0 : tileN;

    tileSizes[i] = builder.getIndexAttr(tileSize);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "TileSizeComputer: [";
    for (unsigned i = 0; i < numDims; ++i) {
      if (i > 0)
        llvm::dbgs() << ", ";
      auto attr = llvm::dyn_cast<Attribute>(tileSizes[i]);
      llvm::dbgs() << llvm::cast<IntegerAttr>(attr).getInt();
    }
    llvm::dbgs() << "]\n";
  });

  return tileSizes;
}
