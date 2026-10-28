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
// empty `llvm.inline_asm` with side effects that takes every operand in a
// vector register and returns it there, and clobbers memory:
//
//   RISC-V  each operand must fill exactly one register group, LMUL = 1/8
//           .. 8 at the target's VLEN. It is placed in the scalable type of
//           that group (`llvm.intr.vector.insert`) and bound with `^vr`, the
//           only vector form LLVM accepts in RISC-V inline asm.
//   x86     each operand must be exactly one XMM, YMM or (with AVX-512) ZMM
//           register, bound with `v` with AVX-512 and `x` otherwise.
//
// The target comes from the module's `llvm.target` attribute. An operand
// that is not a 1-D vector, or that does not fill one register (group), is
// an error: the barrier cannot hold it in registers.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
    if (bits * 8 < vlen || bits > 8 * vlen || !llvm::isPowerOf2_64(bits) ||
        !target.supportsVectorElementType(ty.getElementType())) {
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

class LowerCpuSchedBarrierPass
    : public PassWrapper<LowerCpuSchedBarrierPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerCpuSchedBarrierPass)
  StringRef getArgument() const final { return "lower-cpu-sched-barrier"; }
  StringRef getDescription() const final {
    return "Lower cpu.sched_barrier to an empty inline asm on vector "
           "registers";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
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
        if (regTy != ty) {
          Value undef = LLVM::PoisonOp::create(b, loc, regTy);
          v = LLVM::vector_insert::create(b, loc, undef, v,
                                          b.getI64IntegerAttr(0));
        }
        operands.push_back(v);
        regTypes.push_back(regTy);
      }

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
      if (op.getMemoryAttr())
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

      for (auto [i, res] : llvm::enumerate(op.getResults())) {
        Value v =
            regTypes.size() == 1
                ? asmOp.getRes()
                : LLVM::ExtractValueOp::create(b, loc, asmOp.getRes(),
                                               ArrayRef<int64_t>{(int64_t)i})
                      .getResult();
        if (v.getType() != res.getType())
          v = LLVM::vector_extract::create(b, loc, res.getType(), v,
                                           b.getI64IntegerAttr(0));
        res.replaceAllUsesWith(v);
      }
      op.erase();
    }
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
