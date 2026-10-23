//====- LegalizeForLLVMExport.cpp - Prepare IME for LLVM translation ------===//
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
// Lowering of the memref-level IME ops for the generic target, the SpacemiT
// K1 (XSMTVDot, VLEN = 256).
//
// One vmadot reads two whole vector registers and accumulates into an aligned
// register pair. At VLEN = 256 a register holds a 4x8 int8 tile (A, or B
// stored as N x K) and the pair holds the 4x4 int32 C tile, so the intrinsic
// is nxv4i32 += nxv8i8 * nxv8i8. The sliding-window variants read A from a
// register pair, two 4x8 tiles stacked into 8 rows (nxv16i8).
//
// Each op loads its tiles, issues one intrinsic and stores C back; the tiles
// must be contiguous. The K1 has no floating-point dot and no dynamic slide,
// so vfmadot, vfmadot1/2/3/n and vmadotn* are rejected with a diagnostic.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

#include "Dialect/IME/IMEDialect.h"
#include "Dialect/IME/IMEOps.h"
#include "Dialect/IME/Transform.h"

using namespace mlir;
using namespace buddy::ime;

namespace {

// Elements of one 4x8 int8 tile, of the 8x8 sliding-window A pair, and of
// the 4x4 int32 accumulator.
constexpr int64_t kTileVL = 32;
constexpr int64_t kPairVL = 64;
constexpr int64_t kAccVL = 16;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

static FlatSymbolRefAttr
getOrInsertIntrinsic(ConversionPatternRewriter &rewriter, ModuleOp module,
                     StringRef intrinsicName, LLVM::LLVMFunctionType funcType) {
  auto *ctx = rewriter.getContext();
  if (module.lookupSymbol<LLVM::LLVMFuncOp>(intrinsicName))
    return FlatSymbolRefAttr::get(ctx, intrinsicName);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(module.getBody());
  LLVM::LLVMFuncOp::create(rewriter, module.getLoc(), intrinsicName, funcType,
                           LLVM::Linkage::External, false, LLVM::CConv::C);
  return FlatSymbolRefAttr::get(ctx, intrinsicName);
}

static Value createRVVVectorLoad(ConversionPatternRewriter &rewriter,
                                 Location loc, ModuleOp module, Value pointer,
                                 VectorType vectorType, StringRef suffix,
                                 int64_t vl) {
  auto *ctx = rewriter.getContext();
  auto i64Type = IntegerType::get(ctx, 64);
  auto ptrType = LLVM::LLVMPointerType::get(ctx);
  std::string name = ("llvm.riscv.vle." + suffix + ".p0.i64").str();
  auto funcType = LLVM::LLVMFunctionType::get(
      vectorType, {vectorType, ptrType, i64Type}, false);
  auto funcRef = getOrInsertIntrinsic(rewriter, module, name, funcType);
  Value passthru = LLVM::UndefOp::create(rewriter, loc, vectorType);
  Value vlValue = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                           rewriter.getI64IntegerAttr(vl));
  return LLVM::CallOp::create(rewriter, loc, TypeRange{vectorType}, funcRef,
                              ValueRange{passthru, pointer, vlValue})
      .getResult();
}

static void createRVVVectorStore(ConversionPatternRewriter &rewriter,
                                 Location loc, ModuleOp module, Value vector,
                                 Value pointer, StringRef suffix, int64_t vl) {
  auto *ctx = rewriter.getContext();
  auto i64Type = IntegerType::get(ctx, 64);
  auto ptrType = LLVM::LLVMPointerType::get(ctx);
  std::string name = ("llvm.riscv.vse." + suffix + ".p0.i64").str();
  auto funcType =
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                  {vector.getType(), ptrType, i64Type}, false);
  auto funcRef = getOrInsertIntrinsic(rewriter, module, name, funcType);
  Value vlValue = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                           rewriter.getI64IntegerAttr(vl));
  LLVM::CallOp::create(rewriter, loc, TypeRange{}, funcRef,
                       ValueRange{vector, pointer, vlValue});
}

static Value extractPointerFromMemref(ConversionPatternRewriter &rewriter,
                                      Location loc, Value memref) {
  auto *ctx = rewriter.getContext();
  Value idx =
      memref::ExtractAlignedPointerAsIndexOp::create(rewriter, loc, memref);
  Value i64Val =
      arith::IndexCastOp::create(rewriter, loc, IntegerType::get(ctx, 64), idx);
  return LLVM::IntToPtrOp::create(rewriter, loc,
                                  LLVM::LLVMPointerType::get(ctx), i64Val);
}

static bool hasInt8Inputs(Operation *op) {
  for (Value input : op->getOperands().drop_front().take_front(2)) {
    auto type = dyn_cast<MemRefType>(input.getType());
    if (!type || !type.getElementType().isInteger(8))
      return false;
  }
  return true;
}

/// Lower one K1 int8 dot. `slideA` selects the sliding-window form, whose A
/// operand is the 8x8 register pair instead of one 4x8 register.
static LogicalResult lowerK1Int8Dot(Operation *op, StringRef intrinsicName,
                                    bool slideA,
                                    ConversionPatternRewriter &rewriter) {
  if (!hasInt8Inputs(op))
    return op->emitOpError("the SpacemiT K1 IME supports only int8 inputs");
  auto module = op->getParentOfType<ModuleOp>();
  if (!module)
    return failure();

  Location loc = op->getLoc();
  auto *ctx = rewriter.getContext();
  auto i8 = IntegerType::get(ctx, 8);
  auto tileType = VectorType::get({8}, i8, /*scalableDims=*/true);
  auto pairType = VectorType::get({16}, i8, /*scalableDims=*/true);
  auto accType =
      VectorType::get({4}, IntegerType::get(ctx, 32), /*scalableDims=*/true);
  VectorType aType = slideA ? pairType : tileType;
  StringRef aSuffix = slideA ? "nxv16i8" : "nxv8i8";

  Value vdPtr = extractPointerFromMemref(rewriter, loc, op->getOperand(0));
  Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op->getOperand(1));
  Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op->getOperand(2));

  Value a = createRVVVectorLoad(rewriter, loc, module, vs1Ptr, aType, aSuffix,
                                slideA ? kPairVL : kTileVL);
  Value b = createRVVVectorLoad(rewriter, loc, module, vs2Ptr, tileType,
                                "nxv8i8", kTileVL);
  Value c = createRVVVectorLoad(rewriter, loc, module, vdPtr, accType,
                                "nxv4i32", kAccVL);

  std::string name = (intrinsicName + ".nxv4i32." + aSuffix + ".nxv8i8").str();
  auto funcType =
      LLVM::LLVMFunctionType::get(accType, {accType, aType, tileType}, false);
  auto funcRef = getOrInsertIntrinsic(rewriter, module, name, funcType);
  Value result = LLVM::CallOp::create(rewriter, loc, TypeRange{accType},
                                      funcRef, ValueRange{c, a, b})
                     .getResult();

  createRVVVectorStore(rewriter, loc, module, result, vdPtr, "nxv4i32", kAccVL);
  rewriter.eraseOp(op);
  return success();
}

/// A dot whose intrinsic is `IntrinsicName`; `SlideA` marks the
/// sliding-window forms.
template <typename OpTy, const char *IntrinsicName, bool SlideA>
struct K1DotLowering : public ConvertOpToLLVMPattern<OpTy> {
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return lowerK1Int8Dot(op, IntrinsicName, SlideA, rewriter);
  }
};

/// An op the K1 has no instruction for.
template <typename OpTy>
struct K1UnsupportedLowering : public ConvertOpToLLVMPattern<OpTy> {
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(OpTy op, typename OpTy::Adaptor,
                                ConversionPatternRewriter &) const override {
    return op.emitOpError("has no instruction on the SpacemiT K1 (XSMTVDot)");
  }
};

#define K1_DOT_NAME(VAR, STR) constexpr char VAR[] = "llvm.riscv.ime." STR;
K1_DOT_NAME(kVmadot, "vmadot")
K1_DOT_NAME(kVmadotu, "vmadotu")
K1_DOT_NAME(kVmadotsu, "vmadotsu")
K1_DOT_NAME(kVmadotus, "vmadotus")
K1_DOT_NAME(kVmadot1, "vmadot1")
K1_DOT_NAME(kVmadot1u, "vmadot1u")
K1_DOT_NAME(kVmadot1su, "vmadot1su")
K1_DOT_NAME(kVmadot1us, "vmadot1us")
K1_DOT_NAME(kVmadot2, "vmadot2")
K1_DOT_NAME(kVmadot2u, "vmadot2u")
K1_DOT_NAME(kVmadot2su, "vmadot2su")
K1_DOT_NAME(kVmadot2us, "vmadot2us")
K1_DOT_NAME(kVmadot3, "vmadot3")
K1_DOT_NAME(kVmadot3u, "vmadot3u")
K1_DOT_NAME(kVmadot3su, "vmadot3su")
K1_DOT_NAME(kVmadot3us, "vmadot3us")
#undef K1_DOT_NAME

void addK1Patterns(LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<
      K1DotLowering<VmadotOp, kVmadot, false>,
      K1DotLowering<VmadotuOp, kVmadotu, false>,
      K1DotLowering<VmadotsuOp, kVmadotsu, false>,
      K1DotLowering<VmadotusOp, kVmadotus, false>,
      K1DotLowering<Vmadot1Op, kVmadot1, true>,
      K1DotLowering<Vmadot1uOp, kVmadot1u, true>,
      K1DotLowering<Vmadot1suOp, kVmadot1su, true>,
      K1DotLowering<Vmadot1usOp, kVmadot1us, true>,
      K1DotLowering<Vmadot2Op, kVmadot2, true>,
      K1DotLowering<Vmadot2uOp, kVmadot2u, true>,
      K1DotLowering<Vmadot2suOp, kVmadot2su, true>,
      K1DotLowering<Vmadot2usOp, kVmadot2us, true>,
      K1DotLowering<Vmadot3Op, kVmadot3, true>,
      K1DotLowering<Vmadot3uOp, kVmadot3u, true>,
      K1DotLowering<Vmadot3suOp, kVmadot3su, true>,
      K1DotLowering<Vmadot3usOp, kVmadot3us, true>,
      K1UnsupportedLowering<VfmadotOp>, K1UnsupportedLowering<Vfmadot1Op>,
      K1UnsupportedLowering<Vfmadot2Op>, K1UnsupportedLowering<Vfmadot3Op>,
      K1UnsupportedLowering<VfmadotnOp>, K1UnsupportedLowering<VmadotnOp>,
      K1UnsupportedLowering<VmadotnuOp>, K1UnsupportedLowering<VmadotnsuOp>,
      K1UnsupportedLowering<VmadotnusOp>>(converter);
}

struct LegalizeIMEForLLVMExport
    : public PassWrapper<LegalizeIMEForLLVMExport, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeIMEForLLVMExport)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    MLIRContext &context = getContext();
    LLVMConversionTarget target(context);
    target.addLegalDialect<LLVM::LLVMDialect>();
    configureIMELegalizeForExportTarget(target);

    LLVMTypeConverter typeConverter(&context);
    RewritePatternSet patterns(&context);
    addK1Patterns(typeConverter, patterns);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::populateIMELegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  addK1Patterns(converter, patterns);
}

void mlir::configureIMELegalizeForExportTarget(LLVMConversionTarget &target) {
  target.addLegalDialect<IMEDialect>();
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<memref::MemRefDialect>();
  target.addIllegalOp<VmadotOp, VmadotuOp, VmadotsuOp, VmadotusOp, VfmadotOp,
                      Vmadot1Op, Vmadot1uOp, Vmadot1suOp, Vmadot1usOp,
                      Vmadot2Op, Vmadot2uOp, Vmadot2suOp, Vmadot2usOp,
                      Vmadot3Op, Vmadot3uOp, Vmadot3suOp, Vmadot3usOp,
                      Vfmadot1Op, Vfmadot2Op, Vfmadot3Op, VmadotnOp, VmadotnuOp,
                      VmadotnsuOp, VmadotnusOp, VfmadotnOp>();
}

std::unique_ptr<Pass> buddy::ime::createLegalizeForLLVMExportPass() {
  return std::make_unique<LegalizeIMEForLLVMExport>();
}
