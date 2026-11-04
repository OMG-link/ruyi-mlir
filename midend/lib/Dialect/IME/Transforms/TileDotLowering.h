//===- TileDotLowering.h - ime.tile dots to the dot intrinsics ------------===//
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
// A value-level tile dot. Each fixed 1-D input tile is exactly one register
// and the accumulator one register pair at the target's VLEN, so each is
// placed in the low part of the scalable type the intrinsic takes
// (nxv8i8 / nxv4f16 inputs, nxv4i32 / nxv4f32 accumulator), the intrinsic
// runs, and the accumulator is read back. The insert and extract are
// register-identical at that VLEN and LLVM folds them away.
//
//===----------------------------------------------------------------------===//

#ifndef IME_TRANSFORMS_TILEDOTLOWERING_H
#define IME_TRANSFORMS_TILEDOTLOWERING_H

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace buddy {
namespace ime {

template <typename OpTy, typename IntrOpTy>
struct TileDotLowering : public mlir::ConvertOpToLLVMPattern<OpTy> {
  TileDotLowering(const mlir::LLVMTypeConverter &converter, unsigned vlen)
      : mlir::ConvertOpToLLVMPattern<OpTy>(converter), vlen(vlen) {}

  mlir::LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    using namespace mlir;
    Location loc = op.getLoc();
    auto in = cast<VectorType>(op.getLhs().getType());
    auto acc = cast<VectorType>(op.getAcc().getType());
    if (in.getNumElements() * in.getElementTypeBitWidth() != vlen ||
        acc.getNumElements() * acc.getElementTypeBitWidth() != 2 * vlen)
      return op.emitOpError("operands are not one register tile and one "
                            "register-pair accumulator at VLEN ")
             << vlen;
    // vscale = VLEN / 64.
    auto scalable = [&](VectorType fixed) {
      return VectorType::get({fixed.getNumElements() * 64 / vlen},
                             fixed.getElementType(), /*scalableDims=*/true);
    };
    auto toScalable = [&](Value fixed, VectorType type) -> Value {
      Value undef = LLVM::UndefOp::create(rewriter, loc, type);
      return LLVM::vector_insert::create(rewriter, loc, undef, fixed,
                                         rewriter.getI64IntegerAttr(0));
    };
    VectorType inType = scalable(in), accType = scalable(acc);
    Value c = toScalable(adaptor.getAcc(), accType);
    Value a = toScalable(adaptor.getLhs(), inType);
    Value b = toScalable(adaptor.getRhs(), inType);
    Value dot = IntrOpTy::create(rewriter, loc, accType, c, a, b);
    rewriter.replaceOpWithNewOp<LLVM::vector_extract>(
        op, op.getResult().getType(), dot, rewriter.getI64IntegerAttr(0));
    return success();
  }

  unsigned vlen;
};

} // namespace ime
} // namespace buddy

#endif // IME_TRANSFORMS_TILEDOTLOWERING_H
