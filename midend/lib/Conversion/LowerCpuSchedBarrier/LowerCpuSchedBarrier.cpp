//===- LowerCpuSchedBarrier.cpp - cpu.sched_barrier to inline asm ---------===//
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
// `--lower-cpu-sched-barrier` turns `cpu.sched_barrier` on 1-D vectors into an
// identity the optimizers cannot see through, so each operand is held in its
// vector register and every use of a result depends on the barrier:
//
//   RISC-V  each operand must fill exactly one register group, LMUL = 1/8
//           .. 8 at the target's VLEN. It is placed in the scalable type of
//           that group (`llvm.intr.vector.insert`) and passed through
//           `llvm.riscv.sched.barrier`, or `llvm.riscv.sched.barrier.mem`
//           when the barrier also orders memory. The intrinsic emits nothing
//           and, unlike inline asm, leaves VL and VTYPE known.
//   x86     each operand must be exactly one XMM, YMM or (with AVX-512) ZMM
//           register and goes through an empty inline asm with side effects,
//           bound with `v` with AVX-512 and `x` otherwise, clobbering memory
//           when the barrier orders memory.
//
// The target comes from the module's `llvm.target` attribute. An operand
// that is not a 1-D vector, or that does not fill one register (group), is
// an error: the barrier cannot hold it in registers.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "Dialect/Cpu/CpuDialect.h"
#include "Dialect/Cpu/CpuOps.h"
#include "Utils/TargetInfo.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"

using namespace mlir;

namespace {

/// The register type an operand is bound as, or null with `reason` set.
Type getRegisterType(VectorType ty, const mlir::buddy::TargetInfo &target,
                     std::string &reason) {
  if (ty.getRank() != 1 || ty.isScalable() ||
      !ty.getElementType().isIntOrFloat()) {
    reason = "operand is not a fixed-length 1-D vector";
    return {};
  }
  int64_t bits = ty.getNumElements() * ty.getElementTypeBitWidth();
  if (!target.canHoldVectorElementType(ty.getElementType())) {
    reason = "element type cannot be represented in vector registers";
    return {};
  }
  switch (target.getArch()) {
  case mlir::buddy::TargetInfo::Arch::RISCV: {
    // A group of LMUL registers holds LMUL * VLEN bits; LMUL is a power of
    // two in [1/8, 8], so the operand's size is VLEN / 8 .. 8 VLEN.
    auto exactVLen = target.getExactVectorRegisterBits();
    if (!exactVLen) {
      reason = "exact RISC-V VLEN is required for scheduling barrier lowering";
      return {};
    }
    int64_t vlen = *exactVLen;
    if (bits * 8 < vlen || bits > 8 * vlen || !llvm::isPowerOf2_64(bits)) {
      reason = ("operand of " + Twine(bits) +
                " bits does not fill one register group at VLEN " + Twine(vlen))
                   .str();
      return {};
    }
    // vscale = VLEN / 64, so the scalable type has 64 / VLEN of the
    // elements as its minimum count.
    int64_t minElems = ty.getNumElements() * 64 / vlen;
    if (minElems == 0 || ty.getNumElements() * 64 % vlen != 0) {
      reason = "operand is smaller than the smallest register group of its "
               "element type";
      return {};
    }
    return VectorType::get({minElems}, ty.getElementType(),
                           /*scalableDims=*/{true});
  }
  case mlir::buddy::TargetInfo::Arch::X86: {
    bool zmm = target.hasExtension("avx512f");
    bool ymm = target.hasExtension("avx");
    if (!(bits == 128 || (bits == 256 && ymm) || (bits == 512 && zmm))) {
      reason = ("operand of " + Twine(bits) +
                " bits is not one vector register of the target")
                   .str();
      return {};
    }
    return ty;
  }
  }
  llvm_unreachable("unknown target architecture");
}

/// LLVM inline-asm register constraints may not accept floating vector types
/// that lack target vector arithmetic support. Preserve their bits through an
/// integer carrier of the same width, then bitcast back after the barrier.
static VectorType getBarrierCarrierType(VectorType ty,
                                        const mlir::buddy::TargetInfo &target) {
  Type elementType = ty.getElementType();
  unsigned width = elementType.getIntOrFloatBitWidth();
  bool unsupportedFloatRegisterType =
      target.getArch() == mlir::buddy::TargetInfo::Arch::RISCV &&
      (elementType.isF32() || elementType.isF64()) &&
      width > target.getISAInfo().getMaxELenFp();
  if (!elementType.isF16() && !unsupportedFloatRegisterType)
    return ty;
  return VectorType::get(ty.getShape(),
                         IntegerType::get(ty.getContext(), width),
                         ty.getScalableDims());
}

class LowerCpuSchedBarrierPass
    : public PassWrapper<LowerCpuSchedBarrierPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerCpuSchedBarrierPass)
  StringRef getArgument() const final { return "lower-cpu-sched-barrier"; }
  StringRef getDescription() const final {
    return "Lower cpu.sched_barrier to an opaque identity on vector "
           "registers";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    SmallVector<::buddy::cpu::SchedBarrierOp> barriers;
    getOperation().walk(
        [&](::buddy::cpu::SchedBarrierOp op) { barriers.push_back(op); });
    if (barriers.empty())
      return;
    FailureOr<mlir::buddy::TargetInfo> target =
        mlir::buddy::getTargetInfo(getOperation());
    if (failed(target))
      return signalPassFailure();
    bool riscv = target->getArch() == mlir::buddy::TargetInfo::Arch::RISCV;
    StringRef reg = riscv ? "^vr" : target->hasExtension("avx512f") ? "v" : "x";

    for (auto op : barriers) {
      OpBuilder b(op);
      Location loc = op.getLoc();
      SmallVector<Value> operands;
      SmallVector<Type> regTypes;
      for (Value v : op.getInputs()) {
        auto ty = dyn_cast<VectorType>(v.getType());
        std::string reason = "operand is not a vector";
        Type regTy = ty ? getRegisterType(ty, *target, reason) : Type();
        if (!regTy) {
          op.emitError("cannot hold operand in registers: ") << reason;
          return signalPassFailure();
        }
        auto carrierTy = getBarrierCarrierType(ty, *target);
        auto regCarrierTy =
            getBarrierCarrierType(cast<VectorType>(regTy), *target);
        if (carrierTy != ty)
          v = vector::BitCastOp::create(b, loc, carrierTy, v);
        if (regCarrierTy != carrierTy) {
          Value undef = LLVM::PoisonOp::create(b, loc, regCarrierTy);
          v = LLVM::vector_insert::create(b, loc, undef, v,
                                          b.getI64IntegerAttr(0));
        }
        operands.push_back(v);
        regTypes.push_back(regCarrierTy);
      }

      SmallVector<Value> held;
      if (riscv) {
        // One `llvm.riscv.sched.barrier(.mem)` per operand: an identity the
        // optimizers cannot see through, with no VL/VTYPE effect.
        StringAttr intrin =
            b.getStringAttr(op.getMemoryAttr() ? "llvm.riscv.sched.barrier.mem"
                                               : "llvm.riscv.sched.barrier");
        for (Value v : operands)
          held.push_back(LLVM::CallIntrinsicOp::create(b, loc, v.getType(),
                                                       intrin, ValueRange{v})
                             .getResult(0));
      } else {
        held = lowerToInlineAsm(b, loc, operands, regTypes, reg,
                                static_cast<bool>(op.getMemoryAttr()));
      }

      for (auto [res, v] : llvm::zip_equal(op.getResults(), held)) {
        auto resultTy = cast<VectorType>(res.getType());
        auto resultCarrierTy = getBarrierCarrierType(resultTy, *target);
        if (v.getType() != resultCarrierTy)
          v = LLVM::vector_extract::create(b, loc, resultCarrierTy, v,
                                           b.getI64IntegerAttr(0));
        if (resultCarrierTy != resultTy)
          v = vector::BitCastOp::create(b, loc, resultTy, v);
        res.replaceAllUsesWith(v);
      }
      op.erase();
    }
  }

private:
  /// An empty side-effecting inline asm that takes and returns every operand
  /// in a register of class `reg`, optionally clobbering memory.
  static SmallVector<Value> lowerToInlineAsm(OpBuilder &b, Location loc,
                                             ArrayRef<Value> operands,
                                             ArrayRef<Type> regTypes,
                                             StringRef reg, bool memory) {
    // "=reg,...,0,1,...": every operand comes back in the
    // register it went in.
    SmallString<64> constraints;
    for (size_t i = 0; i < operands.size(); ++i) {
      if (!constraints.empty())
        constraints += ",";
      constraints += ("=" + reg).str();
    }
    for (size_t i = 0; i < operands.size(); ++i) {
      constraints += (Twine(",") + Twine(i)).str();
    }
    if (memory)
      constraints += ",~{memory}";

    Type resTy =
        regTypes.size() == 1
            ? regTypes.front()
            : LLVM::LLVMStructType::getLiteral(b.getContext(), regTypes);
    auto asmOp = LLVM::InlineAsmOp::create(
        b, loc, resTy, operands, b.getStringAttr(""),
        b.getStringAttr(constraints), b.getUnitAttr(), UnitAttr(),
        LLVM::TailCallKindAttr::get(b.getContext(), LLVM::TailCallKind::None),
        UnitAttr(),
        LLVM::AsmDialectAttr::get(b.getContext(), LLVM::AsmDialect::AD_ATT),
        ArrayAttr());

    SmallVector<Value> results;
    for (size_t i = 0; i < regTypes.size(); ++i)
      results.push_back(
          regTypes.size() == 1
              ? asmOp.getRes()
              : LLVM::ExtractValueOp::create(b, loc, asmOp.getRes(),
                                             ArrayRef<int64_t>{(int64_t)i})
                    .getResult());
    return results;
  }
};

} // namespace

namespace mlir {
namespace buddy {
void registerLowerCpuSchedBarrierPass() {
  PassRegistration<LowerCpuSchedBarrierPass>();
}
} // namespace buddy
} // namespace mlir
