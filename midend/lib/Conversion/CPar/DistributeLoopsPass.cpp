//===-- DistributeLoopsPass.cpp - Distribute marked loops over threads ----===//
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
// --cpar-distribute-loops (tensor level, before bufferization): every
// scf.forall whose iterations are independent work items (1-d, no
// shared_outs, no mapping; e.g. a Triton tl.dev.parallel loop) is rewritten
// into a one iteration per thread scf.forall {cpar.threads} whose threads run
// the iterations as the forall's "cpar.schedule" attribute says: "static"
// (the default: one contiguous range per thread, as OpenMP's
// schedule(static)) or "round-robin" (see ::buddy::cpar::distributeForall). The
// thread forall makes One-Shot Bufferize give every thread private buffers
// for the tensors it writes.
//
// --cpar-forall-to-parallel (after bufferization): turns these foralls into
// cpar.parallel regions.
//
//  * Only the outermost forall of a nest is distributed; foralls nested in it
//    become sequential scf.for loops inside each thread.
//  * Foralls with shared_outs or a mapping, or of rank > 1, are rejected.
//  * With num-threads=1 the foralls become sequential scf.for loops, so no
//    runtime call is generated.
//
//===----------------------------------------------------------------------===//

#include "CPar/CParDialect.h"
#include "CPar/CParTransforms.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct CParDistributeLoopsPass
    : public PassWrapper<CParDistributeLoopsPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CParDistributeLoopsPass)
  StringRef getArgument() const final { return "cpar-distribute-loops"; }
  StringRef getDescription() const final {
    return "Distribute the iterations of scf.forall loops over the threads "
           "of a one iteration per thread scf.forall (run before "
           "bufferization)";
  }
  CParDistributeLoopsPass() = default;
  CParDistributeLoopsPass(const CParDistributeLoopsPass &other)
      : PassWrapper(other) {}

  Option<int64_t> numThreads{
      *this, "num-threads",
      llvm::cl::desc("Number of threads of the generated regions (>= 1)"),
      llvm::cl::init(1)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<::buddy::cpar::CParDialect, arith::ArithDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() override {
    if (numThreads < 1) {
      getOperation()->emitError()
          << "cpar-distribute-loops: num-threads must be >= 1, got "
          << static_cast<int64_t>(numThreads);
      return signalPassFailure();
    }

    // Work-item foralls, outermost first. Thread foralls already created by
    // an earlier run are left alone.
    SmallVector<scf::ForallOp> roots, nested;
    bool failed = false;
    getOperation()->walk<WalkOrder::PreOrder>([&](scf::ForallOp forall) {
      if (forall->hasAttr(::buddy::cpar::kThreadForallAttrName))
        return;
      if (forall.getRank() != 1 || !forall.getOutputs().empty() ||
          forall.getMapping().has_value()) {
        forall.emitError("cpar-distribute-loops: expected a 1-d scf.forall "
                         "without shared_outs and mapping");
        failed = true;
        return;
      }
      for (Operation *parent = forall->getParentOp(); parent;
           parent = parent->getParentOp())
        if (isa<scf::ForallOp, ::buddy::cpar::ParallelOp>(parent)) {
          forall.emitRemark("cpar-distribute-loops: nested parallel loop is "
                            "executed sequentially by each thread");
          nested.push_back(forall);
          return;
        }
      roots.push_back(forall);
    });
    if (failed)
      return signalPassFailure();

    IRRewriter rewriter(&getContext());
    // Innermost first, so that every forall is rewritten before its parent.
    for (scf::ForallOp forall : llvm::reverse(nested)) {
      rewriter.setInsertionPoint(forall);
      if (::mlir::failed(scf::forallToForLoop(rewriter, forall))) {
        forall.emitError("cpar-distribute-loops: failed to sequentialize");
        return signalPassFailure();
      }
    }
    for (scf::ForallOp forall : roots) {
      std::optional<::buddy::cpar::Schedule> kind =
          ::buddy::cpar::getSchedule(forall);
      if (!kind) {
        forall.emitError("cpar-distribute-loops: ")
            << ::buddy::cpar::kScheduleAttrName
            << " must be \"static\" or \"round-robin\"";
        return signalPassFailure();
      }
      rewriter.setInsertionPoint(forall);
      LogicalResult result =
          numThreads == 1 ? scf::forallToForLoop(rewriter, forall)
                          : LogicalResult(::buddy::cpar::distributeForall(
                                rewriter, forall, numThreads, *kind));
      if (::mlir::failed(result)) {
        forall.emitError("cpar-distribute-loops: failed to distribute loop");
        return signalPassFailure();
      }
    }
  }
};

struct CParForallToParallelPass
    : public PassWrapper<CParForallToParallelPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CParForallToParallelPass)
  StringRef getArgument() const final { return "cpar-forall-to-parallel"; }
  StringRef getDescription() const final {
    return "Turn the scf.forall loops created by --cpar-distribute-loops "
           "into cpar.parallel regions (run after bufferization)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<::buddy::cpar::CParDialect, arith::ArithDialect>();
  }

  void runOnOperation() override {
    SmallVector<scf::ForallOp> foralls;
    getOperation()->walk([&](scf::ForallOp forall) {
      if (forall->hasAttr(::buddy::cpar::kThreadForallAttrName))
        foralls.push_back(forall);
    });
    IRRewriter rewriter(&getContext());
    for (scf::ForallOp forall : foralls) {
      if (forall.getNumResults() != 0) {
        forall.emitError("cpar-forall-to-parallel: the forall still has "
                         "tensor results; run it after bufferization");
        return signalPassFailure();
      }
      if (::mlir::failed(::buddy::cpar::forallToParallel(rewriter, forall))) {
        forall.emitError("cpar-forall-to-parallel: unsupported forall");
        return signalPassFailure();
      }
    }
  }
};

} // namespace

namespace mlir {
namespace buddy {
void registerCParDistributeLoopsPass() {
  PassRegistration<CParDistributeLoopsPass>();
}
void registerCParForallToParallelPass() {
  PassRegistration<CParForallToParallelPass>();
}
} // namespace buddy
} // namespace mlir
