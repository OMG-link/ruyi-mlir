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
// `--lower-vector-contract-to-ime-tile` maps a tile product that one
// SpacemiT IME dot computes onto an `ime.tile` op:
//
//   C(M x N) + sum_k ext(A[m,k]) * ext(transpose(B)[k,n])
//
// with the plain matmul maps (m,k) x (k,n) -> (m,n), B stored as N x K, and
// a tile shape of the target (`Dialect/IME/TileShapes.h`). Integer operands
// are int8 extended with the same signedness (extsi: vmadot, extui:
// vmadotu) into an int32 C; fp16 operands extended with extf into an fp32 C
// give vfwmadot, which only the A100 has. The tile ops take flat register
// images, so the operands and the result go through `vector.shape_cast`,
// which `--flatten-vector-tiles` folds away.
//
// An add of the product into a running tile is already the contraction's
// accumulator after upstream canonicalization, which the pipeline runs first,
// so the dot accumulates into that tile as the instruction does.
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
#include "Dialect/IME/TileShapes.h"
#include "Utils/TargetInfo.h"

using namespace mlir;

namespace {

enum class Extension { None, Signed, Unsigned, Float };

/// The narrow vector an operand was extended from.
Value getExtendedFrom(Value v, Extension &kind) {
  if (auto ext = v.getDefiningOp<arith::ExtSIOp>()) {
    kind = Extension::Signed;
    return ext.getIn();
  }
  if (auto ext = v.getDefiningOp<arith::ExtUIOp>()) {
    kind = Extension::Unsigned;
    return ext.getIn();
  }
  if (auto ext = v.getDefiningOp<arith::ExtFOp>()) {
    kind = Extension::Float;
    return ext.getIn();
  }
  kind = Extension::None;
  return {};
}

bool hasType(Value v, ArrayRef<int64_t> shape, Type element) {
  auto ty = dyn_cast_or_null<VectorType>(v ? v.getType() : Type());
  return ty && !ty.isScalable() && ty.getShape() == shape &&
         ty.getElementType() == element;
}

struct ContractToTileDot : OpRewritePattern<vector::ContractionOp> {
  ContractToTileDot(MLIRContext *ctx,
                    SmallVector<::buddy::ime::TileShape, 2> shapes)
      : OpRewritePattern(ctx), shapes(std::move(shapes)) {}

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

    Extension lhsKind, rhsKind;
    Value a = getExtendedFrom(contract.getLhs(), lhsKind);
    Value bt = getExtendedFrom(contract.getRhs(), rhsKind);
    auto transpose = bt ? bt.getDefiningOp<vector::TransposeOp>() : nullptr;
    if (!a || !transpose || lhsKind != rhsKind ||
        transpose.getPermutation() != ArrayRef<int64_t>{1, 0})
      return rewriter.notifyMatchFailure(
          contract, "operands are not ext(A), ext(transpose(B)) with one "
                    "extension");
    Value b = transpose.getVector();
    bool isFloat = lhsKind == Extension::Float;
    Type in = isFloat ? Type(rewriter.getF16Type()) : rewriter.getI8Type();
    Type out = isFloat ? Type(rewriter.getF32Type()) : rewriter.getI32Type();

    for (const ::buddy::ime::TileShape &shape : shapes) {
      if (shape.isFloat != isFloat || !hasType(a, {shape.m, shape.k}, in) ||
          !hasType(b, {shape.n, shape.k}, in) ||
          !hasType(contract.getAcc(), {shape.m, shape.n}, out))
        continue;
      Location loc = contract.getLoc();
      auto flatIn = VectorType::get({shape.m * shape.k}, in);
      auto flatAcc = VectorType::get({shape.m * shape.n}, out);
      Value fa = vector::ShapeCastOp::create(rewriter, loc, flatIn, a);
      Value fb = vector::ShapeCastOp::create(
          rewriter, loc, VectorType::get({shape.n * shape.k}, in), b);
      Value fc = vector::ShapeCastOp::create(rewriter, loc, flatAcc,
                                             contract.getAcc());
      Value dot;
      switch (lhsKind) {
      case Extension::Signed:
        dot = ::buddy::ime::TileVmadotOp::create(rewriter, loc, fc, fa, fb);
        break;
      case Extension::Unsigned:
        dot = ::buddy::ime::TileVmadotuOp::create(rewriter, loc, fc, fa, fb);
        break;
      case Extension::Float:
        dot = ::buddy::ime::TileVfwmadotOp::create(rewriter, loc, fc, fa, fb);
        break;
      case Extension::None:
        llvm_unreachable("operands were checked to be extended");
      }
      rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
          contract, contract.getResultType(), dot);
      return success();
    }
    return rewriter.notifyMatchFailure(contract,
                                       "not an IME tile shape of the target");
  }

  SmallVector<::buddy::ime::TileShape, 2> shapes;
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
    return "Lower IME-tile-sized vector.contract to ime.tile dots";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect, ::buddy::ime::IMEDialect>();
  }

  void runOnOperation() override {
    FailureOr<mlir::buddy::TargetInfo> target =
        mlir::buddy::getTargetInfo(getOperation());
    if (failed(target))
      return signalPassFailure();
    SmallVector<::buddy::ime::TileShape, 2> shapes =
        ::buddy::ime::getTileShapes(*target);
    if (shapes.empty())
      return;
    RewritePatternSet patterns(&getContext());
    patterns.add<ContractToTileDot>(&getContext(), std::move(shapes));
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
