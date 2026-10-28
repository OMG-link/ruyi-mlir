//===- LowerVectorContractToIMETile.cpp - K1 tile contracts to vmadot -----===//
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
// `--lower-vector-contract-to-ime-tile` maps the integer tile product one
// SpacemiT K1 `vmadot` computes onto `ime.tile.vmadot(u)`:
//
//   C(4x4 i32) + sum_k ext(A[m,k]) * ext(transpose(B)[k,n])
//
// where A and B are 4x8 int8 vectors (B stored as N x K), the matmul maps
// are the standard (m,k) x (k,n) -> (m,n), and both operands are extended
// with the same signedness (extsi: vmadot, extui: vmadotu). The tile op takes
// flat register images, so the operands and the result go through
// `vector.shape_cast`, which `--flatten-vector-tiles` folds away.
//
// An add of the product into a running tile is already the contraction's
// accumulator after upstream canonicalization, which the pipeline runs first,
// so the dot accumulates into that tile as the instruction does.
//
// The pass only applies when the module's target has XSMTVDot at VLEN = 256.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Dialect/IME/IMEDialect.h"
#include "Dialect/IME/IMEOps.h"
#include "Utils/TargetInfo.h"

using namespace mlir;

namespace {

/// The int8 vector an operand was extended from, with its signedness.
Value getExtendedFrom(Value v, bool &isUnsigned) {
  if (auto ext = v.getDefiningOp<arith::ExtSIOp>()) {
    isUnsigned = false;
    return ext.getIn();
  }
  if (auto ext = v.getDefiningOp<arith::ExtUIOp>()) {
    isUnsigned = true;
    return ext.getIn();
  }
  return {};
}

bool hasType(Value v, ArrayRef<int64_t> shape, unsigned width) {
  auto ty = dyn_cast_or_null<VectorType>(v ? v.getType() : Type());
  return ty && !ty.isScalable() && ty.getShape() == shape &&
         ty.getElementType().isInteger(width);
}

struct ContractToTileDot : OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contract,
                                PatternRewriter &rewriter) const override {
    if (contract.getKind() != vector::CombiningKind::ADD)
      return rewriter.notifyMatchFailure(contract, "not an add contraction");
    MLIRContext *ctx = contract.getContext();
    AffineExpr m, n, k;
    bindDims(ctx, m, n, k);
    SmallVector<AffineMap> matmul = {AffineMap::get(3, 0, {m, k}, ctx),
                                     AffineMap::get(3, 0, {k, n}, ctx),
                                     AffineMap::get(3, 0, {m, n}, ctx)};
    if (contract.getIndexingMapsArray() != ArrayRef<AffineMap>(matmul))
      return rewriter.notifyMatchFailure(contract, "not a plain matmul");
    if (!hasType(contract.getAcc(), {4, 4}, 32))
      return rewriter.notifyMatchFailure(contract, "not a 4x4 i32 tile");

    bool lhsUnsigned = false, rhsUnsigned = false;
    Value a = getExtendedFrom(contract.getLhs(), lhsUnsigned);
    Value bt = getExtendedFrom(contract.getRhs(), rhsUnsigned);
    auto transpose = bt ? bt.getDefiningOp<vector::TransposeOp>() : nullptr;
    if (!hasType(a, {4, 8}, 8) || !transpose ||
        transpose.getPermutation() != ArrayRef<int64_t>{1, 0} ||
        lhsUnsigned != rhsUnsigned)
      return rewriter.notifyMatchFailure(
          contract, "operands are not ext(A 4x8 i8), ext(transpose(B 4x8 i8))");
    Value b = transpose.getVector();
    if (!hasType(b, {4, 8}, 8))
      return rewriter.notifyMatchFailure(contract, "B is not 4x8 i8");

    Location loc = contract.getLoc();
    auto flatIn = VectorType::get({32}, rewriter.getI8Type());
    auto flatAcc = VectorType::get({16}, rewriter.getI32Type());
    Value fa = vector::ShapeCastOp::create(rewriter, loc, flatIn, a);
    Value fb = vector::ShapeCastOp::create(rewriter, loc, flatIn, b);
    Value fc =
        vector::ShapeCastOp::create(rewriter, loc, flatAcc, contract.getAcc());
    Value dot =
        lhsUnsigned
            ? ::buddy::ime::TileVmadotuOp::create(rewriter, loc, fc, fa, fb)
                  .getResult()
            : ::buddy::ime::TileVmadotOp::create(rewriter, loc, fc, fa, fb)
                  .getResult();
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
        contract, contract.getResultType(), dot);
    return success();
  }
};

class LowerVectorContractToIMETilePass
    : public PassWrapper<LowerVectorContractToIMETilePass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerVectorContractToIMETilePass)
  StringRef getArgument() const final {
    return "lower-vector-contract-to-ime-tile";
  }
  StringRef getDescription() const final {
    return "Lower K1-tile-sized integer vector.contract to ime.tile.vmadot";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect, ::buddy::ime::IMEDialect>();
  }

  void runOnOperation() override {
    FailureOr<mlir::buddy::TargetInfo> target =
        mlir::buddy::getTargetInfo(getOperation());
    if (failed(target))
      return signalPassFailure();
    if (target->getArch() != mlir::buddy::TargetInfo::Arch::RISCV ||
        !target->hasExtension("xsmtvdot") ||
        target->getVectorRegisterBits() != 256)
      return;
    RewritePatternSet patterns(&getContext());
    patterns.add<ContractToTileDot>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace buddy {
void registerLowerVectorContractToIMETilePass() {
  PassRegistration<LowerVectorContractToIMETilePass>();
}
} // namespace buddy
} // namespace mlir
