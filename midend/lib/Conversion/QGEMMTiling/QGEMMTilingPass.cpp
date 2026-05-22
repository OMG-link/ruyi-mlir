//===- QGEMMTilingPass.cpp - QGEMM Tiling Pass ------------------*- C++ -*-===//
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
// This file implements the QGEMM tiling pass that tiles M/N dimensions of
// multi-level contraction patterns in canonicalized Linalg IR.
//
//===----------------------------------------------------------------------===//

#include "ContractionSubgraph.h"
#include "DimensionClassifier.h"
#include "ReduceChainAnalysis.h"
#include "RootFilter.h"
#include "TileSizeComputer.h"
#include "TilingExecutor.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/RegionUtils.h"

#define DEBUG_TYPE "qgemm-tiling"

using namespace mlir;

namespace {

/// Find all tiling roots in the function.
/// Currently delegates to reduce-chain analysis. Future versions may
/// combine multiple strategies here.
static SmallVector<linalg::LinalgOp> findTilingRoots(func::FuncOp funcOp) {
  SmallVector<linalg::ReduceOp> reduceRoots =
      buddy::findReduceChainRoots(funcOp);
  SmallVector<linalg::LinalgOp> roots;
  roots.reserve(reduceRoots.size());
  for (linalg::ReduceOp r : reduceRoots)
    roots.push_back(cast<linalg::LinalgOp>(r.getOperation()));
  return roots;
}

struct QGEMMTilingPass
    : public PassWrapper<QGEMMTilingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QGEMMTilingPass)

  StringRef getArgument() const override { return "qgemm-tiling"; }
  StringRef getDescription() const override {
    return "Tile M/N dimensions of multi-level contraction patterns";
  }

  QGEMMTilingPass() = default;
  QGEMMTilingPass(const QGEMMTilingPass &other) : PassWrapper(other) {}

  Option<int64_t> tileM{
      *this, "tile-m",
      llvm::cl::desc("Tile size for M dimensions (0 = no tiling)"),
      llvm::cl::init(0)};
  Option<int64_t> tileN{
      *this, "tile-n",
      llvm::cl::desc("Tile size for N dimensions (0 = no tiling)"),
      llvm::cl::init(0)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect, arith::ArithDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // 1. Validate pass parameters.
    if (tileM < 0) {
      funcOp->emitError() << "invalid tile-m value "
                          << static_cast<int64_t>(tileM) << " (must be >= 0)";
      signalPassFailure();
      return;
    }
    if (tileN < 0) {
      funcOp->emitError() << "invalid tile-n value "
                          << static_cast<int64_t>(tileN) << " (must be >= 0)";
      signalPassFailure();
      return;
    }

    // If both tile sizes are 0, no tiling is requested — early return.
    if (tileM == 0 && tileN == 0)
      return;

    // 2. Find all tiling roots.
    SmallVector<linalg::LinalgOp> roots = findTilingRoots(funcOp);
    if (roots.empty())
      return; // No eligible roots found — leave IR unchanged.

    // 3. Collect subgraphs for each root (skip failed ones).
    SmallVector<buddy::TilingCandidate> rawCandidates;
    for (linalg::LinalgOp root : roots) {
      auto subgraph = buddy::collectSubgraph(root);
      if (failed(subgraph))
        continue; // Subgraph collection failed — skip silently.
      rawCandidates.push_back({root, std::move(*subgraph)});
    }
    if (rawCandidates.empty())
      return;

    // 4. Filter out diamond-conflicting roots.
    SmallVector<buddy::TilingCandidate> candidates =
        buddy::filterTilingCandidates(rawCandidates);

    // 5. Process each valid candidate sequentially.
    bool anyTiled = false;
    for (auto &candidate : candidates) {
      // Classify dimensions.
      auto classification = buddy::classifyDimensions(candidate.subgraph);
      if (failed(classification))
        continue; // Skip this root.

      // Compute tile sizes.
      auto tileSizes = buddy::computeTileSizes(candidate.tilingRoot,
                                               *classification, tileM, tileN);
      if (failed(tileSizes))
        continue; // Skip this root (dynamic dims).

      // Check if all tile sizes are 0 (no tiling needed).
      bool allZero = llvm::all_of(*tileSizes, [](OpFoldResult ofr) {
        auto attr = dyn_cast<Attribute>(ofr);
        return attr && cast<IntegerAttr>(attr).getInt() == 0;
      });
      if (allZero)
        continue; // Skip this root — extents ≤ tile sizes.

      // Execute tiling.
      buddy::executeTiling(candidate.tilingRoot, candidate.subgraph,
                           *tileSizes);
      anyTiled = true;
    }

    // 6. Run DCE and CSE if any tiling was performed.
    if (anyTiled) {
      IRRewriter rewriter(funcOp->getContext());
      // Remove dead ops left behind by tiling (originals replaced by clones).
      (void)runRegionDCE(rewriter, funcOp->getRegions());
      // Merge duplicate constants and common subexpressions introduced by
      // cloning producers into loop bodies.
      DominanceInfo domInfo(funcOp);
      eliminateCommonSubExpressions(rewriter, domInfo, funcOp);
    }
  }
};

} // namespace

namespace mlir {
namespace buddy {

void registerQGEMMTilingPass() { PassRegistration<QGEMMTilingPass>(); }

} // namespace buddy
} // namespace mlir
