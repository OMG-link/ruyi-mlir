//===- FlattenVectorTiles.cpp - Map n-D vector tiles onto 1-D registers ---===//
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
// `--flatten-vector-tiles` rewrites n-D vectors (n >= 2) into 1-D vectors in
// row-major order, so that one tile maps onto one register group. LLVM
// lowers an n-D vector as an array of its rows, which turns every
// elementwise op on a 4x8 tile into four ops on 8-element rows; a flat
// vector<32xi8> is one instruction instead.
//
// Two steps, both upstream MLIR patterns:
//
//  1. Contiguous `vector.transfer_read/write` are flattened into 1-D
//     transfers on a collapsed memref. The remaining n-D transfers are split
//     by VectorToSCF's fully unrolled transfer lowering into 1-D transfers of
//     their rows, the same lowering the pipeline would apply later.
//  2. Elementwise ops, constants, shape_casts, shuffle-like ops (including
//     the row extracts and inserts step 1 leaves) and scf.for iter_args are
//     linearized. `vector.contract` and `vector.transpose`, which the
//     linearizer does not cover, keep their n-D types with shape_casts at
//     the boundary, for their own lowerings. `cpu.sched_barrier` holds the
//     flattened values.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Dialect/Cpu/CpuDialect.h"
#include "Dialect/Cpu/CpuOps.h"

using namespace mlir;

namespace {

/// cpu.sched_barrier on n-D vectors holds their flattened forms.
struct LinearizeSchedBarrier
    : OpConversionPattern<::buddy::cpu::SchedBarrierOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::buddy::cpu::SchedBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<::buddy::cpu::SchedBarrierOp>(
        op, adaptor.getInputs(), op.getMemoryAttr());
    return success();
  }
};

class FlattenVectorTilesPass
    : public PassWrapper<FlattenVectorTilesPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FlattenVectorTilesPass)
  StringRef getArgument() const final { return "flatten-vector-tiles"; }
  StringRef getDescription() const final {
    return "Flatten n-D vector tiles and contiguous transfers to 1-D vectors";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    RewritePatternSet transfers(ctx);
    vector::populateFlattenVectorTransferPatterns(transfers);
    if (failed(applyPatternsGreedily(getOperation(), std::move(transfers))))
      return signalPassFailure();

    RewritePatternSet split(ctx);
    populateVectorToSCFConversionPatterns(
        split, VectorTransferToSCFOptions().enableFullUnroll());
    if (failed(applyPatternsGreedily(getOperation(), std::move(split))))
      return signalPassFailure();

    TypeConverter converter;
    ConversionTarget target(*ctx);
    RewritePatternSet linearize(ctx);
    vector::populateForVectorLinearize(converter, target);
    target.addLegalOp<vector::ContractionOp, vector::TransposeOp>();
    target.addDynamicallyLegalOp<::buddy::cpu::SchedBarrierOp>(
        [&](Operation *op) { return converter.isLegal(op); });
    linearize.add<LinearizeSchedBarrier>(converter, ctx);
    vector::populateVectorLinearizeBasePatterns(converter, target, linearize);
    vector::populateVectorLinearizeShuffleLikeOpsPatterns(converter, target,
                                                          linearize);
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, linearize,
                                                         target);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(linearize))))
      return signalPassFailure();

    RewritePatternSet cleanup(ctx);
    vector::ShapeCastOp::getCanonicalizationPatterns(cleanup, ctx);
    (void)applyPatternsGreedily(getOperation(), std::move(cleanup));
  }
};

} // namespace

namespace mlir {
namespace buddy {
void registerFlattenVectorTilesPass() {
  PassRegistration<FlattenVectorTilesPass>();
}
} // namespace buddy
} // namespace mlir
