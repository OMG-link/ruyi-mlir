//===- TilingExecutor.cpp - Execute tiling via SCF infrastructure -*- C++
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
// This file implements the executeTiling utility, which applies tiling to the
// annotated tiling root using MLIR's SCF tile-and-fuse infrastructure and
// identity-controlled producer fusion.
//
//===----------------------------------------------------------------------===//

#include "TilingExecutor.h"

#include "Conversion/QGEMMRecognize/QGEMMAnnotation.h"

#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"

#define DEBUG_TYPE "qgemm-tiling"

using namespace mlir;

namespace mlir {
namespace buddy {

void executeTiling(linalg::LinalgOp tilingRoot, unsigned rootId,
                   ArrayRef<OpFoldResult> tileSizes) {
  // If all tile sizes are 0, nothing to tile — early return.
  bool allZero = llvm::all_of(tileSizes, [](OpFoldResult ofr) {
    auto attr = dyn_cast<Attribute>(ofr);
    return attr && cast<IntegerAttr>(attr).getInt() == 0;
  });
  if (allZero)
    return;

  // Create a rewriter at the tiling root's location.
  IRRewriter rewriter(tilingRoot->getContext());
  rewriter.setInsertionPoint(tilingRoot);

  // Cast to TilingInterface (all LinalgOps implement TilingInterface).
  auto tilingInterface = cast<TilingInterface>(tilingRoot.getOperation());

  // Configure SCFTileAndFuseOptions.
  scf::SCFTileAndFuseOptions options;
  options.tilingOptions.setTileSizes(tileSizes);
  options.tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);

  // Measured in task 7.1 with basic_q80.mlir: the SCF utility preserves
  // discardable qgemm.node/qgemm.root attributes on cloned root-owned
  // operations. A destination/outs tensor-semantics linalg.fill clone stays
  // unannotated. Consequently this consumer deliberately has no
  // read-then-patch or re-attachment path.
  options.setFusionControlFn(
      [rootId](tensor::ExtractSliceOp, OpResult originalProducer,
               bool isDestinationOperand)
          -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
        Operation *producerOp = originalProducer.getOwner();
        if (auto producerRoot = getQGEMMNodeRoot(producerOp);
            producerRoot && *producerRoot == rootId)
          return scf::SCFTileAndFuseOptions::ControlFnResult{};

        // Only tensor-semantics fills have a tensor result that can reach this
        // callback. Memref-semantics fills therefore cannot qualify here.
        if (isDestinationOperand) {
          if (auto fill = dyn_cast<linalg::FillOp>(producerOp);
              fill && fill->getNumResults() != 0)
            return scf::SCFTileAndFuseOptions::ControlFnResult{};
        }
        return std::nullopt;
      });

  // Call tileConsumerAndFuseProducersUsingSCF.
  FailureOr<scf::SCFTileAndFuseResult> result =
      scf::tileConsumerAndFuseProducersUsingSCF(rewriter, tilingInterface,
                                                options);

  if (failed(result))
    llvm_unreachable("tileConsumerAndFuseProducersUsingSCF failed — this is a "
                     "compiler bug; all preconditions were validated before "
                     "calling the tiling utility");

  // On success: replace all uses of the original consumer's results with
  // the tiled replacements.
  for (auto [origValue, replacement] : result->replacements)
    rewriter.replaceAllUsesWith(origValue, replacement);

  // Erase the original tiling root if it has no remaining uses.
  if (tilingRoot->use_empty())
    rewriter.eraseOp(tilingRoot);
}

} // namespace buddy
} // namespace mlir
