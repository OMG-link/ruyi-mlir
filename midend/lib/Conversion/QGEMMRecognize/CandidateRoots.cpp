//===- CandidateRoots.cpp - QGEMM candidate root enumeration ----*- C++ -*-===//
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
// This file implements candidate-root enumeration for the QGEMM recognizer.
// The reverse-scan enumeration and additive-combiner logic remain independent
// of the original tiling analysis. Operator boundary classification is shared
// through Boundary.h.
//
//===----------------------------------------------------------------------===//

#include "CandidateRoots.h"
#include "Boundary.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineExpr.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>

using namespace mlir;

namespace mlir {
namespace buddy {
namespace {

/// Return true when an operand edge stops reverse propagation.
static bool isPropagationBoundaryEdge(Operation *consumer, OpOperand *operand) {
  Operation *definingOp = operand->get().getDefiningOp();
  if (!definingOp)
    return true;
  if (definingOp->getBlock() != consumer->getBlock())
    return true;

  auto generic = dyn_cast<linalg::GenericOp>(consumer);
  if (!generic)
    return false;

  AffineMap map = generic.getMatchingIndexingMap(operand);
  for (AffineExpr expr : map.getResults()) {
    if (isa<AffineConstantExpr>(expr))
      return true;
  }
  return false;
}

/// Return true only for a combiner with one additive operation.
static bool hasSingleAdditiveCombiner(linalg::ReduceOp reduce) {
  unsigned additiveOps = 0;
  for (Operation &op : reduce.getCombiner().front().without_terminator()) {
    if (isa<arith::AddIOp, arith::AddFOp>(op))
      ++additiveOps;
  }
  return additiveOps == 1;
}

/// Propagate the downstream-additive-reduce mark through supported edges.
static void propagateUpstream(Operation *op,
                              llvm::DenseSet<Operation *> &hasDownstreamReduce) {
  for (OpOperand &operand : op->getOpOperands()) {
    if (isPropagationBoundaryEdge(op, &operand))
      continue;

    Operation *definingOp = operand.get().getDefiningOp();
    if (!definingOp || isQGEMMOperatorBoundary(definingOp))
      continue;
    hasDownstreamReduce.insert(definingOp);
  }
}

/// Enumerate roots in one block using a reverse scan.
static SmallVector<linalg::ReduceOp>
findCandidateRootsInBlock(Block &block) {
  SmallVector<linalg::ReduceOp> roots;
  llvm::DenseSet<Operation *> hasDownstreamReduce;

  for (Operation &op : llvm::reverse(block.getOperations())) {
    if (isQGEMMOperatorBoundary(&op))
      continue;

    if (auto reduce = dyn_cast<linalg::ReduceOp>(&op)) {
      if (!hasSingleAdditiveCombiner(reduce))
        continue;

      if (!hasDownstreamReduce.contains(&op))
        roots.push_back(reduce);

      // Both roots and non-roots propagate the fact upstream. This is what
      // makes the outermost additive reduce the only candidate in a chain.
      propagateUpstream(&op, hasDownstreamReduce);
      continue;
    }

    if (hasDownstreamReduce.contains(&op))
      propagateUpstream(&op, hasDownstreamReduce);
  }

  std::reverse(roots.begin(), roots.end());
  return roots;
}

} // namespace

SmallVector<QGEMMCandidateRoot>
findQGEMMCandidateRoots(func::FuncOp funcOp) {
  SmallVector<linalg::ReduceOp> reductions;
  funcOp.walk([&](Block *block) {
    SmallVector<linalg::ReduceOp> blockRoots =
        findCandidateRootsInBlock(*block);
    reductions.append(blockRoots.begin(), blockRoots.end());
  });

  SmallVector<QGEMMCandidateRoot> candidates;
  candidates.reserve(reductions.size());
  for (auto [index, reduce] : llvm::enumerate(reductions))
    candidates.push_back({reduce, static_cast<unsigned>(index)});
  return candidates;
}

} // namespace buddy
} // namespace mlir
