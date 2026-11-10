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
// --cpar-distribute-loops (tensor level, before bufferization): every scf.for
// carrying the unit attribute "cpar.parallel" is rewritten into a one
// iteration per thread scf.forall {cpar.threads} whose threads run the
// iterations round robin (see ::buddy::cpar::distributeLoop).  The forall
// makes One-Shot Bufferize give every thread private buffers for the tensors
// it writes.
//
// --cpar-forall-to-parallel (after bufferization): turns these foralls into
// cpar.parallel regions.
//
//  * Only the outermost marked loop of a nest is distributed; marked loops
//    nested in it stay sequential inside each thread.
//  * Marked loops with loop-carried values are rejected: their iterations are
//    not independent.
//  * With num-threads=1 the markers are dropped and the loops stay as they
//    are, so no runtime call is generated.
//
//===----------------------------------------------------------------------===//

#include "CPar/CParDialect.h"
#include "CPar/CParTransforms.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct CParDistributeLoopsPass
    : public PassWrapper<CParDistributeLoopsPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CParDistributeLoopsPass)
  StringRef getArgument() const final { return "cpar-distribute-loops"; }
  StringRef getDescription() const final {
    return "Distribute the iterations of scf.for loops marked with "
           "'cpar.parallel' over the threads of a one iteration per thread "
           "scf.forall (run before bufferization)";
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

    // Outermost marked loops; markers of nested ones are dropped.
    SmallVector<scf::ForOp> roots;
    getOperation()->walk<WalkOrder::PreOrder>([&](scf::ForOp loop) {
      if (!loop->hasAttr(::buddy::cpar::kParallelLoopAttrName))
        return;
      Operation *parent = loop->getParentOp();
      bool nested = false;
      for (; parent; parent = parent->getParentOp()) {
        if (isa<::buddy::cpar::ParallelOp>(parent) ||
            parent->hasAttr(::buddy::cpar::kParallelLoopAttrName) ||
            parent->hasAttr(::buddy::cpar::kThreadForallAttrName)) {
          nested = true;
          break;
        }
      }
      if (nested) {
        loop.emitRemark("cpar-distribute-loops: nested parallel loop is "
                        "executed sequentially by each thread");
        loop->removeAttr(::buddy::cpar::kParallelLoopAttrName);
        return;
      }
      roots.push_back(loop);
    });

    IRRewriter rewriter(&getContext());
    bool failed = false;
    for (scf::ForOp loop : roots) {
      if (loop.getNumResults() != 0) {
        loop.emitError("cpar-distribute-loops: a parallel loop must not carry "
                       "values from one iteration to the next");
        failed = true;
        continue;
      }
      if (numThreads == 1) {
        loop->removeAttr(::buddy::cpar::kParallelLoopAttrName);
        continue;
      }
      if (::mlir::failed(
              ::buddy::cpar::distributeLoop(rewriter, loop, numThreads))) {
        loop.emitError("cpar-distribute-loops: failed to distribute loop");
        failed = true;
      }
    }
    if (failed)
      signalPassFailure();
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
