//===- ReduceChainAnalysis.h - Reduce chain root identification -*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the ReduceChainAnalysis utility, which identifies all
// reduce chain roots in a FuncOp. A reduce chain root is a linalg.reduce with
// an additive combiner whose result does not feed another additive reduce
// within the same basic block (possibly through element-wise generics).
//
// This replaces FinalConsumerFinder for v02 of the QGEMM tiling pass.
//
//===----------------------------------------------------------------------===//

#ifndef QGEMM_TILING_REDUCE_CHAIN_ANALYSIS_H
#define QGEMM_TILING_REDUCE_CHAIN_ANALYSIS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace buddy {

/// Find all reduce chain roots in a FuncOp.
///
/// A reduce chain root is a linalg.reduce with an additive combiner
/// (arith.addi or arith.addf) whose result does NOT feed another
/// additive reduce within the same basic block through the subgraph
/// (i.e., through non-boundary edges and non-boundary ops).
///
/// Searches all blocks at all nesting levels (including inside scf.for
/// loop bodies). Returns roots ordered by IR walk order.
SmallVector<linalg::ReduceOp> findReduceChainRoots(func::FuncOp funcOp);

} // namespace buddy
} // namespace mlir

#endif // QGEMM_TILING_REDUCE_CHAIN_ANALYSIS_H
