//===- SubgraphBoundary.h - Subgraph boundary utilities ---------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared utility functions defining the boundary conditions for contraction
// subgraph traversal and propagation. Used by both ReduceChainAnalysis
// (reverse-scan propagation) and ContractionSubgraph (backward BFS).
//
//===----------------------------------------------------------------------===//

#ifndef QGEMM_TILING_SUBGRAPH_BOUNDARY_H
#define QGEMM_TILING_SUBGRAPH_BOUNDARY_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"

namespace mlir {
namespace buddy {

/// Is this op a subgraph boundary? If true, the op is NOT part of the
/// subgraph and traversal/propagation does not enter it.
/// Boundary ops: tensor.extract_slice, tensor.empty, unrecognized op types.
inline bool isBoundaryOp(Operation *op) {
  if (isa<tensor::ExtractSliceOp, tensor::EmptyOp>(op))
    return true;
  // Recognized subgraph-internal ops:
  if (isa<linalg::GenericOp, linalg::ReduceOp, linalg::TransposeOp,
          linalg::FillOp>(op))
    return false;
  // Everything else is a boundary (unrecognized).
  return true;
}

/// Is this edge a subgraph boundary? An edge is (consumerOp, operand).
/// If true, the defOp of this operand is NOT part of the subgraph
/// (traversal/propagation does not cross this edge).
///
/// An edge is a boundary when:
///   (a) the operand has no defining op (block argument), OR
///   (b) the defining op is in a different block than the consumer, OR
///   (c) the operand's indexing map in the consumer contains a constant
///       expression (broadcast edge — applies to any operand of
///       linalg.generic, both inputs and inits/outputs).
inline bool isBoundaryEdge(Operation *consumerOp, OpOperand *operand) {
  Operation *defOp = operand->get().getDefiningOp();
  if (!defOp)
    return true; // Block argument — boundary
  if (defOp->getBlock() != consumerOp->getBlock())
    return true; // Cross-block edge — boundary
  auto genericOp = dyn_cast<linalg::GenericOp>(consumerOp);
  if (!genericOp)
    return false;
  AffineMap map = genericOp.getMatchingIndexingMap(operand);
  for (AffineExpr expr : map.getResults()) {
    if (isa<AffineConstantExpr>(expr))
      return true;
  }
  return false;
}

/// Return true if `reduceOp` uses an additive combiner (arith.addi or
/// arith.addf in its combiner region).
inline bool hasAdditiveCombiner(linalg::ReduceOp reduceOp) {
  Block &body = reduceOp.getCombiner().front();
  for (Operation &op : body.without_terminator()) {
    if (isa<arith::AddIOp, arith::AddFOp>(op))
      return true;
  }
  return false;
}

} // namespace buddy
} // namespace mlir

#endif // QGEMM_TILING_SUBGRAPH_BOUNDARY_H
