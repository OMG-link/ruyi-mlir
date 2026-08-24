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
// This file implements the QGEMM tiling pass. Recognition is deliberately
// outside this pass: annotated roots and node identities are its only input.
//
//===----------------------------------------------------------------------===//

#include "Conversion/QGEMMRecognize/QGEMMAnnotation.h"
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

/// Find every annotated tiling root in function IR order. The vector contains
/// operations rather than only Linalg ops so malformed hand-written
/// annotations can be skipped safely without re-deriving recognition.
static SmallVector<Operation *> collectAnnotatedRoots(func::FuncOp funcOp) {
  SmallVector<Operation *> roots;
  funcOp.walk([&](Operation *operation) {
    if (operation->hasAttr(buddy::kQGEMMRootAttr))
      roots.push_back(operation);
  });
  return roots;
}

struct QGEMMTilingPass
    : public PassWrapper<QGEMMTilingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QGEMMTilingPass)

  StringRef getArgument() const override { return "qgemm-tiling"; }
  StringRef getDescription() const override {
    return "Tile M/N dimensions of annotated QGEMM roots";
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
  Option<bool> explain{
      *this, "explain",
      llvm::cl::desc("Emit remarks for roots skipped by qgemm tiling"),
      llvm::cl::init(false)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect, arith::ArithDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // Validate pass parameters before inspecting or changing the function.
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

    // Recognition is the sole producer of qgemm.root. A function without one
    // is valid non-QGEMM input and is intentionally a no-op.
    SmallVector<Operation *> roots = collectAnnotatedRoots(funcOp);
    if (roots.empty())
      return;

    bool anyTiled = false;
    for (Operation *rootOperation : roots) {
      auto rootDims = buddy::getQGEMMRootDims(rootOperation);
      auto rootId = buddy::getQGEMMNodeRoot(rootOperation);
      auto tilingRoot = dyn_cast<linalg::LinalgOp>(rootOperation);
      if (!rootDims || !rootId || !tilingRoot)
        continue;

      // A root with a valid identity and dimensions remains recoverable even
      // when its stage metadata is absent, malformed, unknown, or data_prep.
      // Stage is advisory for locating the root; the fusion callback still
      // admits only same-root compute producers.
      auto stage = buddy::getQGEMMNodeStage(rootOperation);
      if (!stage || *stage != buddy::kQGEMMComputeStage)
        rootOperation->emitWarning()
            << "qgemm-tiling: tiling root does not carry stage = \"compute\"; "
               "continuing from qgemm.root";

      auto tileSizes = buddy::computeTileSizes(
          tilingRoot, rootDims->first, rootDims->second, tileM, tileN, explain);
      if (failed(tileSizes))
        continue;

      bool allZero = llvm::all_of(*tileSizes, [](OpFoldResult ofr) {
        auto attr = dyn_cast<Attribute>(ofr);
        return attr && cast<IntegerAttr>(attr).getInt() == 0;
      });
      if (allZero)
        continue;

      // Each located root is processed independently, in the collection order.
      // There is deliberately no overlap filter: ownership was established by
      // recognition, while root-neutral fills are controlled by the executor.
      buddy::executeTiling(tilingRoot, *rootId, *tileSizes);
      anyTiled = true;
    }

    // Remove dead ops left behind by tiling and merge duplicate constants and
    // common subexpressions introduced by cloning producers into loop bodies.
    if (anyTiled) {
      IRRewriter rewriter(funcOp->getContext());
      (void)runRegionDCE(rewriter, funcOp->getRegions());
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
