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

FailureOr<scf::ForallOp> distributeLoop(RewriterBase &rewriter, scf::ForOp loop,
                                        int64_t numThreads) {
  if (numThreads < 1)
    return rewriter.notifyMatchFailure(loop, "num_threads must be >= 1");
  if (loop.getNumResults() != 0 || !loop.getInitArgs().empty())
    return rewriter.notifyMatchFailure(
        loop, "loops with loop-carried values cannot be distributed");

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(loop);
  Location loc = loop.getLoc();
  Type type = loop.getInductionVar().getType();
  scf::ForOp distributed;
  auto forall = scf::ForallOp::create(
      rewriter, loc, ArrayRef<OpFoldResult>{rewriter.getIndexAttr(numThreads)},
      /*outputs=*/ValueRange{}, /*mapping=*/std::nullopt,
      [&](OpBuilder &b, Location loc, ValueRange ivs) {
        Value tid = castIndexToLoopType(b, loc, ivs[0], type);
        Value nt = castIndexToLoopType(
            b, loc, arith::ConstantIndexOp::create(b, loc, numThreads), type);
        Value offset = arith::MulIOp::create(b, loc, tid, loop.getStep());
        Value lb = arith::AddIOp::create(b, loc, loop.getLowerBound(), offset);
        Value step = arith::MulIOp::create(b, loc, loop.getStep(), nt);
        distributed =
            scf::ForOp::create(b, loc, lb, loop.getUpperBound(), step);
        scf::InParallelOp::create(b, loc);
      });
  forall->setAttr(kThreadForallAttrName, rewriter.getUnitAttr());

  // Keep the loop's other attributes (e.g. tt.num_stages) on the new loop.
  for (NamedAttribute attr : loop->getDiscardableAttrs())
    if (attr.getName() != kParallelLoopAttrName)
      distributed->setAttr(attr.getName(), attr.getValue());

  // Replace the empty body (and its implicit scf.yield) by the original one.
  Block *newBody = distributed.getBody();
  rewriter.eraseOp(newBody->getTerminator());
  rewriter.mergeBlocks(loop.getBody(), newBody,
                       {distributed.getInductionVar()});
  rewriter.eraseOp(loop);
  return forall;
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
