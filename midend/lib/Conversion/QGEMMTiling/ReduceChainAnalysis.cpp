//===- ReduceChainAnalysis.cpp - Reduce chain root identification -*- C++ -===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the ReduceChainAnalysis utility. It identifies all
// reduce chain roots in a FuncOp using a reverse-scan algorithm with
// subgraph-scoped propagation.
//
// Algorithm overview:
//   For each block, scan ops in reverse order. When an additive reduce is
//   encountered that has NOT been marked as "has downstream reduce", it is
//   a root. All additive reduces propagate the "has downstream reduce" mark
//   upstream through non-boundary edges and non-boundary ops. Non-reduce ops
//   that are marked also propagate upstream.
//
//===----------------------------------------------------------------------===//

#include "ReduceChainAnalysis.h"
#include "SubgraphBoundary.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "qgemm-tiling"

using namespace mlir;

namespace mlir {
namespace buddy {

//===----------------------------------------------------------------------===//
// Helper: propagateUpstream
//===----------------------------------------------------------------------===//

/// Propagate "has downstream reduce" mark to upstream ops through
/// non-boundary edges within the subgraph scope.
///
/// For each operand of `op`, if the edge is not a boundary edge and the
/// defining op is not a boundary op, mark the defining op as having a
/// downstream reduce.
static void propagateUpstream(Operation *op,
                              DenseSet<Operation *> &hasDownstreamReduce) {
  for (OpOperand &operand : op->getOpOperands()) {
    if (isBoundaryEdge(op, &operand))
      continue;
    Operation *defOp = operand.get().getDefiningOp();
    // isBoundaryEdge already checks for null defOp (block argument), but
    // guard defensively.
    if (!defOp)
      continue;
    if (isBoundaryOp(defOp))
      continue;
    hasDownstreamReduce.insert(defOp);
  }
}

//===----------------------------------------------------------------------===//
// findReduceChainRootsInBlock
//===----------------------------------------------------------------------===//

/// Find reduce chain roots within a single block using reverse-scan.
///
/// The algorithm scans ops from last to first. For each additive reduce
/// not already marked as having a downstream reduce, it is identified as
/// a root. All additive reduces (root or not) propagate the mark upstream.
/// Non-reduce, non-boundary ops that are marked also propagate upstream.
///
/// Returns roots in forward IR order (reversed from collection order).
static SmallVector<linalg::ReduceOp> findReduceChainRootsInBlock(Block &block) {
  SmallVector<linalg::ReduceOp> roots;
  DenseSet<Operation *> hasDownstreamReduce;

  // Reverse scan: from last op to first op in the block.
  for (Operation &op : llvm::reverse(block.getOperations())) {
    if (isBoundaryOp(&op))
      continue;

    if (auto reduceOp = dyn_cast<linalg::ReduceOp>(&op)) {
      if (hasAdditiveCombiner(reduceOp)) {
        if (!hasDownstreamReduce.contains(&op)) {
          LLVM_DEBUG(llvm::dbgs() << "  Reduce chain root: " << op << "\n");
          roots.push_back(reduceOp);
        }
        // All additive reduces propagate upstream (root or not).
        propagateUpstream(&op, hasDownstreamReduce);
      }
      continue;
    }

    // Non-reduce, non-boundary op: propagate if marked.
    if (hasDownstreamReduce.contains(&op))
      propagateUpstream(&op, hasDownstreamReduce);
  }

  // Roots were collected in reverse order; flip to forward order.
  std::reverse(roots.begin(), roots.end());
  return roots;
}

//===----------------------------------------------------------------------===//
// findReduceChainRoots
//===----------------------------------------------------------------------===//

SmallVector<linalg::ReduceOp> findReduceChainRoots(func::FuncOp funcOp) {
  SmallVector<linalg::ReduceOp> allRoots;

  // Walk all blocks in all nested regions (depth-first).
  funcOp.walk([&](Block *block) {
    SmallVector<linalg::ReduceOp> blockRoots =
        findReduceChainRootsInBlock(*block);
    allRoots.append(blockRoots.begin(), blockRoots.end());
  });

  LLVM_DEBUG(llvm::dbgs() << "findReduceChainRoots: found " << allRoots.size()
                          << " root(s)\n");
  return allRoots;
}

} // namespace buddy
} // namespace mlir
