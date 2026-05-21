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
#include "FinalConsumerFinder.h"
#include "TileSizeComputer.h"
#include "TilingExecutor.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "qgemm-tiling"

using namespace mlir;

namespace {

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
    // Validate pass parameters.
    if (tileM < 0) {
      getOperation()->emitError()
          << "invalid tile-m value " << static_cast<int64_t>(tileM)
          << " (must be >= 0)";
      signalPassFailure();
      return;
    }
    if (tileN < 0) {
      getOperation()->emitError()
          << "invalid tile-n value " << static_cast<int64_t>(tileN)
          << " (must be >= 0)";
      signalPassFailure();
      return;
    }

    // If both tile sizes are 0, no tiling is requested — early return.
    if (tileM == 0 && tileN == 0)
      return;

    // Step 1: Find the final consumer (tiling root).
    auto consumerInfo = buddy::findFinalConsumer(getOperation());
    if (!consumerInfo.has_value()) {
      getOperation()->emitRemark(
          "no materialize_in_destination found; skipping qgemm-tiling");
      return;
    }

    // Step 2: Collect the contraction subgraph via backward BFS.
    auto subgraph = buddy::collectSubgraph(consumerInfo->tilingRoot);
    if (failed(subgraph))
      return;

    // Step 3: Classify dimensions as M, N, or Batch.
    auto classificationResult = buddy::classifyDimensions(*subgraph);
    if (failed(classificationResult))
      return; // Bail out — diagnostics already emitted by classifyDimensions.
    auto &classification = *classificationResult;

    // Step 4: Compute tile sizes (includes dynamic shape check;
    // returns failure if M/N dims are dynamic).
    auto tileSizes = buddy::computeTileSizes(consumerInfo->tilingRoot,
                                             classification, tileM, tileN);
    if (failed(tileSizes))
      return;

    // Step 5: Execute tiling with producer fusion.
    buddy::executeTiling(consumerInfo->tilingRoot, *subgraph, *tileSizes);
  }
};

} // namespace

namespace mlir {
namespace buddy {

void registerQGEMMTilingPass() { PassRegistration<QGEMMTilingPass>(); }

} // namespace buddy
} // namespace mlir
