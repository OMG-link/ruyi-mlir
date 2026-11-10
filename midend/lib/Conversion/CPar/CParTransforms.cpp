//===-- CParTransforms.cpp - Create cpar.parallel regions -----------------===//
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

#include "CPar/CParTransforms.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;

namespace buddy {
namespace cpar {

FailureOr<ParallelOp> wrapInParallelRegion(RewriterBase &rewriter,
                                           Operation *first, Operation *last,
                                           int64_t numThreads) {
  if (numThreads < 1)
    return rewriter.notifyMatchFailure(first, "num_threads must be >= 1");
  Block *block = first->getBlock();
  if (last->getBlock() != block || last->isBeforeInBlock(first))
    return rewriter.notifyMatchFailure(first, "invalid op range");
  if (last->hasTrait<OpTrait::IsTerminator>())
    return rewriter.notifyMatchFailure(last, "cannot wrap a terminator");

  // The range must not define values that are used after it: every thread
  // computes its own copy, there is no single value to forward.
  auto end = std::next(Block::iterator(last));
  for (Operation &op : llvm::make_range(Block::iterator(first), end))
    for (Value result : op.getResults())
      for (Operation *user : result.getUsers()) {
        Operation *ancestor = block->findAncestorOpInBlock(*user);
        if (!ancestor || last->isBeforeInBlock(ancestor))
          return rewriter.notifyMatchFailure(
              &op, "a value defined in the range is used after it");
      }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(first);
  auto parallel = ParallelOp::create(rewriter, first->getLoc(), numThreads);
  Block &body = parallel.getBody().front();
  // Move the ops in front of the cpar.yield created by the builder.
  body.getOperations().splice(Block::iterator(body.getTerminator()),
                              block->getOperations(), Block::iterator(first),
                              end);
  return parallel;
}

// Converts an index value to the type of the loop bounds.
static Value castIndexToLoopType(OpBuilder &b, Location loc, Value value,
                                 Type type) {
  if (type.isIndex())
    return value;
  return arith::IndexCastOp::create(b, loc, type, value);
}

// Creates, at the insertion point,
//
//   scf.forall (%t) in (N) {
//     scf.for %iv = lb + t * step to ub step step * N { <empty> }
//   } {cpar.threads}
//
// and returns the forall and the (empty) scf.for. The bounds share one type.
static std::pair<scf::ForallOp, scf::ForOp>
createThreadForall(RewriterBase &rewriter, Location loc, Value lowerBound,
                   Value upperBound, Value step, int64_t numThreads) {
  Type type = lowerBound.getType();
  scf::ForOp distributed;
  auto forall = scf::ForallOp::create(
      rewriter, loc, ArrayRef<OpFoldResult>{rewriter.getIndexAttr(numThreads)},
      /*outputs=*/ValueRange{}, /*mapping=*/std::nullopt,
      [&](OpBuilder &b, Location loc, ValueRange ivs) {
        Value tid = castIndexToLoopType(b, loc, ivs[0], type);
        Value nt = castIndexToLoopType(
            b, loc, arith::ConstantIndexOp::create(b, loc, numThreads), type);
        Value offset = arith::MulIOp::create(b, loc, tid, step);
        Value lb = arith::AddIOp::create(b, loc, lowerBound, offset);
        Value newStep = arith::MulIOp::create(b, loc, step, nt);
        distributed = scf::ForOp::create(b, loc, lb, upperBound, newStep);
        scf::InParallelOp::create(b, loc);
      });
  forall->setAttr(kThreadForallAttrName, rewriter.getUnitAttr());
  return {forall, distributed};
}

FailureOr<scf::ForallOp> distributeLoop(RewriterBase &rewriter, scf::ForOp loop,
                                        int64_t numThreads) {
  if (numThreads < 1)
    return rewriter.notifyMatchFailure(loop, "num_threads must be >= 1");
  if (loop.getNumResults() != 0 || !loop.getInitArgs().empty())
    return rewriter.notifyMatchFailure(
        loop, "loops with loop-carried values cannot be distributed");

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(loop);
  auto [forall, distributed] =
      createThreadForall(rewriter, loop.getLoc(), loop.getLowerBound(),
                         loop.getUpperBound(), loop.getStep(), numThreads);

  // Keep the loop's attributes (e.g. tt.num_stages) on the new loop.
  for (NamedAttribute attr : loop->getDiscardableAttrs())
    distributed->setAttr(attr.getName(), attr.getValue());

  // Replace the empty body (and its implicit scf.yield) by the original one.
  Block *newBody = distributed.getBody();
  rewriter.eraseOp(newBody->getTerminator());
  rewriter.mergeBlocks(loop.getBody(), newBody,
                       {distributed.getInductionVar()});
  rewriter.eraseOp(loop);
  return forall;
}

FailureOr<scf::ForallOp> distributeForall(RewriterBase &rewriter,
                                          scf::ForallOp forall,
                                          int64_t numThreads) {
  if (numThreads < 1)
    return rewriter.notifyMatchFailure(forall, "num_threads must be >= 1");
  if (forall.getRank() != 1 || !forall.getOutputs().empty() ||
      forall.getNumResults() != 0 || forall.getMapping().has_value())
    return rewriter.notifyMatchFailure(
        forall, "expected a 1-d scf.forall without shared_outs and mapping");
  if (!forall.getTerminator().getYieldingOps().empty())
    return rewriter.notifyMatchFailure(
        forall, "expected an empty scf.forall.in_parallel terminator");

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(forall);
  Location loc = forall.getLoc();
  Value lb = getValueOrCreateConstantIndexOp(rewriter, loc,
                                             forall.getMixedLowerBound()[0]);
  Value ub = getValueOrCreateConstantIndexOp(rewriter, loc,
                                             forall.getMixedUpperBound()[0]);
  Value step =
      getValueOrCreateConstantIndexOp(rewriter, loc, forall.getMixedStep()[0]);
  auto [threads, distributed] =
      createThreadForall(rewriter, loc, lb, ub, step, numThreads);

  for (NamedAttribute attr : forall->getDiscardableAttrs())
    distributed->setAttr(attr.getName(), attr.getValue());

  Block *newBody = distributed.getBody();
  rewriter.eraseOp(newBody->getTerminator());
  Block *body = forall.getBody();
  rewriter.eraseOp(body->getTerminator());
  rewriter.setInsertionPointToEnd(body);
  scf::YieldOp::create(rewriter, loc);
  rewriter.mergeBlocks(body, newBody, {distributed.getInductionVar()});
  rewriter.eraseOp(forall);
  return threads;
}

FailureOr<ParallelOp> forallToParallel(RewriterBase &rewriter,
                                       scf::ForallOp forall) {
  if (forall.getRank() != 1 || !forall.getOutputs().empty() ||
      forall.getNumResults() != 0)
    return rewriter.notifyMatchFailure(
        forall, "expected a 1-d scf.forall without shared_outs");
  if (!forall.isNormalized())
    return rewriter.notifyMatchFailure(forall, "expected a normalized forall");
  std::optional<int64_t> numThreads =
      getConstantIntValue(forall.getMixedUpperBound()[0]);
  if (!numThreads || *numThreads < 1)
    return rewriter.notifyMatchFailure(
        forall, "expected a constant, positive thread count");
  if (!forall.getTerminator().getYieldingOps().empty())
    return rewriter.notifyMatchFailure(
        forall, "expected an empty scf.forall.in_parallel terminator");

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(forall);
  Location loc = forall.getLoc();
  auto parallel = ParallelOp::create(rewriter, loc, *numThreads);
  Block &body = parallel.getBody().front();
  rewriter.setInsertionPointToStart(&body);
  Value tid = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(),
                                         parallel.getThreadId());

  Block *forallBody = forall.getBody();
  rewriter.eraseOp(forallBody->getTerminator());
  rewriter.inlineBlockBefore(forallBody, body.getTerminator(), {tid});
  rewriter.eraseOp(forall);
  return parallel;
}

} // namespace cpar
} // namespace buddy
