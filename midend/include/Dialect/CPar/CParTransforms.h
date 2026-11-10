//===-- CParTransforms.h - CPU parallel region transforms -------*- C++ -*-===//
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
// C++ entry points to create cpar.parallel regions from existing code, and
// the registration of the cpar passes.
//
//===----------------------------------------------------------------------===//

#ifndef CPAR_CPARTRANSFORMS_H
#define CPAR_CPARTRANSFORMS_H

#include "CPar/CParOps.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

namespace buddy {
namespace cpar {

/// Discardable unit attribute marking an scf.for whose iterations are
/// independent and may be executed by several threads.
constexpr llvm::StringLiteral kParallelLoopAttrName = "cpar.parallel";

/// Discardable unit attribute marking an scf.forall created by
/// distributeLoop, i.e. one iteration per thread, to be turned into a
/// cpar.parallel region by forallToParallel.
constexpr llvm::StringLiteral kThreadForallAttrName = "cpar.threads";

/// Name of the runtime entry point called by outlined regions.
constexpr llvm::StringLiteral kRuntimeFunctionName = "__cpar_parallel";

/// Wraps the ops [first, last] of one block into a cpar.parallel region with
/// `numThreads` threads.  Every thread executes the whole range (SPMD); the
/// ops may use the thread id / count of the returned op to split the work.
/// Fails if a value defined in the range is used after it.
///
/// cpar.parallel has no bufferization semantics: use this on buffer level IR
/// (after bufferization), where it is explicit which memory the threads
/// share.  On tensor level IR use distributeLoop.
mlir::FailureOr<ParallelOp> wrapInParallelRegion(mlir::RewriterBase &rewriter,
                                                 mlir::Operation *first,
                                                 mlir::Operation *last,
                                                 int64_t numThreads);

/// Rewrites `loop` (which must not have loop-carried values) into
///
///   scf.forall (%t) in (N) {
///     scf.for %iv = lb + t * step to ub step step * N { body }
///   } {cpar.threads}
///
/// i.e. the iterations are distributed round robin over `numThreads` threads.
/// Discardable attributes of the loop, except kParallelLoopAttrName, are kept
/// on the new loop.
///
/// The forall is what makes this safe on tensor level IR: One-Shot
/// Bufferize treats its body as a parallel region, so a tensor defined
/// outside of the loop and written inside it is bufferized out of place,
/// into a buffer private to the thread.  (Bufferizing the original scf.for
/// may legally reuse a single buffer for all iterations, which threads would
/// then share.)  Run forallToParallel after bufferization.
mlir::FailureOr<mlir::scf::ForallOp>
distributeLoop(mlir::RewriterBase &rewriter, mlir::scf::ForOp loop,
               int64_t numThreads);

/// Rewrites a buffer level `scf.forall (%t) in (N)` carrying
/// kThreadForallAttrName (1-d, normalized, no shared_outs) into
/// `cpar.parallel num_threads(N)`.
mlir::FailureOr<ParallelOp> forallToParallel(mlir::RewriterBase &rewriter,
                                             mlir::scf::ForallOp forall);

} // namespace cpar
} // namespace buddy

namespace mlir {
namespace buddy {
/// --cpar-distribute-loops: distributes loops marked with "cpar.parallel"
/// (tensor level, before bufferization).
void registerCParDistributeLoopsPass();
/// --cpar-forall-to-parallel: turns the foralls created by
/// --cpar-distribute-loops into cpar.parallel (after bufferization).
void registerCParForallToParallelPass();
/// --cpar-outline-to-runtime: outlines cpar.parallel regions (LLVM dialect)
/// into functions called through __cpar_parallel.
void registerCParOutlineToRuntimePass();
} // namespace buddy
} // namespace mlir

#endif // CPAR_CPARTRANSFORMS_H
