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
// annotated tiling root using MLIR's SCF tile-and-fuse infrastructure. Ordinary
// producer fusion is controlled by both the qgemm root identity and the
// compute-stage boundary; the destination-fill exception is root-neutral.
//
//===----------------------------------------------------------------------===//

#include "TilingExecutor.h"

#include "Conversion/QGEMMRecognize/QGEMMAnnotation.h"

#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/ADT/SmallPtrSet.h"

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

  // Fusion is first admitted by structure: a tensor-semantics linalg.fill
  // reached through a destination/outs edge is the sole root-neutral
  // exception. Its qgemm.node/qgemm.root, if present, is redundant metadata
  // and does not change that safe tile-local cloning decision. The warning is
  // diagnostic only and does not prevent cloning. An ins-edge fill and a
  // memref-semantics fill do not enter this exception because the former is
  // not a destination edge and the latter has no tensor result here.
  //
  // SmallPtrSet is used only to avoid repeating that warning for the same fill
  // during this executeTiling invocation; it is not part of fusion ownership.
  llvm::SmallPtrSet<Operation *, 8> warnedDestinationFills;
  options.setFusionControlFn(
      [rootId, &warnedDestinationFills](tensor::ExtractSliceOp,
                                        OpResult originalProducer,
                                        bool isDestinationOperand)
          -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
        Operation *producerOp = originalProducer.getOwner();

        // Check the structural destination-fill exception before ordinary
        // root/stage matching, so annotation metadata cannot bypass the
        // warning or change the exception's semantics.
        if (isDestinationOperand) {
          if (auto fill = dyn_cast<linalg::FillOp>(producerOp);
              fill && fill->getNumResults() != 0) {
            if ((producerOp->hasAttr(kQGEMMNodeAttr) ||
                 producerOp->hasAttr(kQGEMMRootAttr)) &&
                warnedDestinationFills.insert(producerOp).second)
              producerOp->emitWarning()
                  << "qgemm-tiling: destination tensor fill carries QGEMM "
                     "annotation; continuing with tile-local cloning";
            return scf::SCFTileAndFuseOptions::ControlFnResult{};
          }
        }

        // Ordinary fusion is bounded by both the root identity and the
        // compute-stage boundary. Same-root data_prep producers, missing or
        // malformed stages, different-root producers, and unannotated
        // ordinary producers stay outside the tile.
        auto producerRoot = getQGEMMNodeRoot(producerOp);
        auto producerStage = getQGEMMNodeStage(producerOp);
        if (producerRoot && producerStage && *producerRoot == rootId &&
            *producerStage == kQGEMMComputeStage)
          return scf::SCFTileAndFuseOptions::ControlFnResult{};

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
