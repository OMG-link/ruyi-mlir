//===- QGEMMDecompressCouplingPass.cpp - QGEMM decompression coupling -----===//
//
// Licensed under the Apache License v2.0 (the "License");
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
// This file implements the QGEMM decompression-matmul coupling pass. For each
// annotated QGEMM root it chooses where the parallel dimensions m and n enter
// the operator: `pre` keeps decompression in (m,k) / (n,k) space (dynamic
// pre-decompression), `on-the-fly` moves the broadcast upstream so
// decompression happens in (m,n,k) space (dynamic decompression).
//
// Every root is processed in two phases. Phase A is pure analysis over Omega(r)
// and allocates no IR; Phase B mutates IR only after Phase A has completed.
//
//===----------------------------------------------------------------------===//

#include "Conversion/QGEMMDecompressCoupling/QGEMMDecompressCouplingPass.h"
#include "AxisTracker.h"
#include "Conversion/QGEMMRecognize/QGEMMAnnotation.h"
#include "ForwardReadOnlyCopy.h"
#include "LogicalGraph.h"
#include "Placement.h"
#include "Rebuilder.h"
#include "Separator.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include <memory>

#define DEBUG_TYPE "qgemm-decompress-coupling"

using namespace mlir;

namespace {

/// Placement strategy selected by the `decompress` option.
enum class DecompressMode { Pre, OnTheFly };

/// One annotated QGEMM root: its final reduction and identity.
struct QGEMMRootRef {
  Operation *rootOp;
  unsigned rootId;
};

/// Collect every op carrying qgemm.root, in function order.
static SmallVector<QGEMMRootRef> collectRoots(func::FuncOp funcOp) {
  SmallVector<QGEMMRootRef> roots;
  funcOp.walk([&](Operation *op) {
    if (!buddy::getQGEMMRootDims(op))
      return;
    std::optional<unsigned> rootId = buddy::getQGEMMNodeRoot(op);
    // The recognizer always writes qgemm.node next to qgemm.root; a root
    // without its node identity is an annotation-writer bug.
    assert(rootId && "qgemm.root op without qgemm.node root identity");
    roots.push_back({op, *rootId});
  });
  return roots;
}

/// Omega(r): every op whose qgemm.node root is `rootId`, whatever its stage.
static SmallVector<Operation *> collectOmega(func::FuncOp funcOp,
                                             unsigned rootId) {
  SmallVector<Operation *> omega;
  funcOp.walk([&](Operation *op) {
    std::optional<unsigned> id = buddy::getQGEMMNodeRoot(op);
    if (id && *id == rootId)
      omega.push_back(op);
  });
  return omega;
}

/// Per-root driver. Phase A analyses Omega(r) without touching IR; Phase B
/// applies the resulting plan. Each root's outcome is local.
class RootContext {
public:
  RootContext(func::FuncOp funcOp, QGEMMRootRef root, DecompressMode mode)
      : funcOp(funcOp), root(root), mode(mode) {}

  void run() {
    omega = collectOmega(funcOp, root.rootId);
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] root " << root.rootId
                            << ": |Omega| = " << omega.size() << "\n");
    if (!analyze())
      return;
    rewrite();
  }

private:
  /// Phase A: axis tracking, separation plan, logical graph, variance and
  /// placement. Returns false when the root must be left unchanged (R1 or
  /// the skip check).
  bool analyze() {
    auto rootDims = buddy::getQGEMMRootDims(root.rootOp);
    assert(rootDims && "collectRoots only returns qgemm.root ops");
    omegaSet.insert(omega.begin(), omega.end());
    trackerStorage = std::make_unique<buddy::qgemm_coupling::AxisTracker>(
        root.rootOp, omegaSet, rootDims->first, rootDims->second);
    buddy::qgemm_coupling::AxisTracker &tracker = *trackerStorage;
    tracker.run();
    LLVM_DEBUG({
      llvm::dbgs() << "[" DEBUG_TYPE "] root " << root.rootId
                   << " axis positions:\n";
      tracker.print(llvm::dbgs());
    });

    separation =
        buddy::qgemm_coupling::computeSeparationPlan(omegaSet, tracker);
    LLVM_DEBUG({
      llvm::dbgs() << "[" DEBUG_TYPE "] root " << root.rootId
                   << " separation plan:\n";
      separation.print(llvm::dbgs());
    });
    if (separation.hasR1()) {
      // R1 is the only bail-out: the root is left unchanged and no R2/R3 is
      // computed for it (Req 2.5).
      buddy::qgemm_coupling::emitR1Remarks(separation, root.rootId);
      return false;
    }
    graph = buddy::qgemm_coupling::LogicalGraph::build(omegaSet, tracker,
                                                       root.rootOp);
    LLVM_DEBUG({
      llvm::dbgs() << "[" DEBUG_TYPE "] root " << root.rootId
                   << " logical graph:\n";
      graph.print(llvm::dbgs());
    });
    placement = buddy::qgemm_coupling::computePlacement(
        graph, tracker, separation, mode == DecompressMode::OnTheFly);
    LLVM_DEBUG({
      llvm::dbgs() << "[" DEBUG_TYPE "] root " << root.rootId << " placement ("
                   << (mode == DecompressMode::OnTheFly ? "on-the-fly" : "pre")
                   << "):\n";
      placement.print(llvm::dbgs(), graph);
    });
    if (!placement.r1Diagnostics.empty()) {
      // Inconsistent broadcast positions are an unsupported introduction
      // form: same R1 path, emitted before any R2/R3 (Req 2.5).
      llvm::append_range(separation.r1Diagnostics, placement.r1Diagnostics);
      buddy::qgemm_coupling::emitR1Remarks(separation, root.rootId);
      return false;
    }
    graph.emitR2Remarks();
    placement.emitR3Remarks(graph);
    // Skip check (Req 6.1): target equals the current placement and
    // separation found no rewrite, so the root is left unchanged.
    return !placement.unchanged;
  }

  /// Phase B: rebuild, annotate, then erase the original Omega(r) and the
  /// out-of-scope inits it orphaned (Req 6.2-6.8, 8.1-8.5).
  void rewrite() {
    auto rootDims = buddy::getQGEMMRootDims(root.rootOp);
    rebuild = buddy::qgemm_coupling::rebuildRoot(omegaSet, *trackerStorage,
                                                 separation, graph, placement);
    buddy::qgemm_coupling::writeAnnotations(rebuild, graph, placement,
                                            root.rootOp, root.rootId,
                                            rootDims->first, rootDims->second);
    buddy::qgemm_coupling::eraseOriginals(omegaSet, rebuild);
  }

  func::FuncOp funcOp;
  QGEMMRootRef root;
  DecompressMode mode;
  SmallVector<Operation *> omega;
  SetVector<Operation *> omegaSet;
  /// Owned here because Phase B reads it after analyze() returns; it holds a
  /// reference to omegaSet.
  std::unique_ptr<buddy::qgemm_coupling::AxisTracker> trackerStorage;
  buddy::qgemm_coupling::SeparationPlan separation;
  buddy::qgemm_coupling::LogicalGraph graph;
  buddy::qgemm_coupling::PlacementResult placement;
  buddy::qgemm_coupling::RebuildResult rebuild;
};

struct QGEMMDecompressCouplingPass
    : public PassWrapper<QGEMMDecompressCouplingPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QGEMMDecompressCouplingPass)

  StringRef getArgument() const override { return "qgemm-decompress-coupling"; }

  StringRef getDescription() const override {
    return "Choose how QGEMM weight decompression couples to the matrix "
           "multiplication by placing the parallel-dimension broadcasts";
  }

  QGEMMDecompressCouplingPass() = default;
  QGEMMDecompressCouplingPass(const QGEMMDecompressCouplingPass &other)
      : PassWrapper(other) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    // Load_Copy_Forwarding runs over the whole function before roots are
    // collected, under both modes (Req 14.7, 14.11). Folding, constant CSE
    // and region simplification are off so that the driver changes nothing
    // but the forwarded load copies.
    {
      RewritePatternSet patterns(&getContext());
      buddy::populateForwardReadOnlyCopyPatterns(patterns);
      GreedyRewriteConfig config;
      config.enableFolding(false);
      config.enableConstantCSE(false);
      config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
      config.setStrictness(GreedyRewriteStrictness::ExistingOps);
      (void)applyPatternsGreedily(funcOp, std::move(patterns), config);
    }
    SmallVector<QGEMMRootRef> roots = collectRoots(funcOp);
    if (roots.empty())
      return;
    for (const QGEMMRootRef &root : roots)
      RootContext(funcOp, root, decompress).run();
  }

  Option<DecompressMode> decompress{
      *this, "decompress",
      llvm::cl::desc("Decompression-matmul coupling strategy"),
      llvm::cl::init(DecompressMode::Pre),
      llvm::cl::values(
          clEnumValN(DecompressMode::Pre, "pre",
                     "Dynamic pre-decompression: keep the broadcast next to "
                     "the matmul"),
          clEnumValN(DecompressMode::OnTheFly, "on-the-fly",
                     "Dynamic decompression: move the broadcast upstream of "
                     "the decompression chain"))};
};

} // namespace

namespace mlir {
namespace buddy {
void registerQGEMMDecompressCouplingPass() {
  PassRegistration<QGEMMDecompressCouplingPass>();
}
} // namespace buddy
} // namespace mlir
