//===- TargetInfo.cpp - Hardware facts of the compilation target ----------===//
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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "Utils/TargetInfo.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/RISCVTargetParser.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/TargetParser/X86TargetParser.h"

using namespace mlir;
using namespace mlir::buddy;

bool TargetInfo::hasExtension(StringRef ext) const {
  switch (arch) {
  case Arch::RISCV:
    return isa->hasExtension(ext);
  case Arch::X86:
    return x86Features.contains(ext);
  }
  llvm_unreachable("unknown target architecture");
}

bool TargetInfo::hasVector() const {
  switch (arch) {
  case Arch::RISCV:
    return isa->hasExtension("zve32x");
  case Arch::X86:
    return hasExtension("sse2");
  }
  llvm_unreachable("unknown target architecture");
}

unsigned TargetInfo::getMinVectorRegisterBits() const {
  switch (arch) {
  case Arch::RISCV:
    return exactVLen ? exactVLen : isa->getMinVLen();
  case Arch::X86:
    if (hasExtension("avx512f"))
      return 512;
    if (hasExtension("avx"))
      return 256;
    return hasExtension("sse2") ? 128 : 0;
  }
  llvm_unreachable("unknown target architecture");
}

unsigned TargetInfo::getMaxVectorRegisterBits() const {
  if (arch == Arch::RISCV)
    return exactVLen ? exactVLen : isa->getMaxVLen();
  return getMinVectorRegisterBits();
}

std::optional<unsigned> TargetInfo::getExactVectorRegisterBits() const {
  unsigned min = getMinVectorRegisterBits();
  return min && min == getMaxVectorRegisterBits() ? std::optional<unsigned>(min)
                                                  : std::nullopt;
}

bool TargetInfo::canHoldVectorElementType(Type type) const {
  if (!hasVector())
    return false;
  if (auto intType = dyn_cast<IntegerType>(type)) {
    unsigned width = intType.getWidth();
    if (width != 8 && width != 16 && width != 32 && width != 64)
      return false;
    return arch == Arch::X86 || width <= isa->getMaxELen();
  }
  switch (arch) {
  case Arch::RISCV:
    if (type.isF16())
      return true;
    if (type.isF32() || type.isF64())
      return type.getIntOrFloatBitWidth() <= isa->getMaxELen();
    // bf16 has specialized operations outside this pipeline's type policy.
    return false;
  case Arch::X86:
    if (type.isF16())
      return true;
    // SSE2 covers f32 and f64. bf16 has only conversions and dot products.
    return type.isF32() || type.isF64();
  }
  llvm_unreachable("unknown target architecture");
}

/// Element type of a scalar or shaped value.
static Type getElementType(Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type))
    return shaped.getElementType();
  return type;
}

/// Validate data element types while allowing only compare results and select
/// conditions to use the i1 mask representation.
static bool hasSupportedVectorElementTypes(Operation *op,
                                           const TargetInfo &target) {
  for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
    Type type = getElementType(operand.getType());
    if (type.isInteger(1) && mlir::isa<arith::SelectOp>(op) && index == 0)
      continue;
    if (type.isInteger(1) || !target.canHoldVectorElementType(type))
      return false;
  }
  for (Type result : op->getResultTypes()) {
    Type type = getElementType(result);
    if (type.isInteger(1) && mlir::isa<arith::CmpIOp, arith::CmpFOp>(op))
      continue;
    if (type.isInteger(1) || !target.canHoldVectorElementType(type))
      return false;
  }
  return true;
}

static bool containsF16(Operation *op) {
  return llvm::any_of(op->getOperandTypes(),
                      [](Type type) { return getElementType(type).isF16(); }) ||
         llvm::any_of(op->getResultTypes(),
                      [](Type type) { return getElementType(type).isF16(); });
}

static bool isF16F32Conversion(Operation *op) {
  auto isHalfOrFloat = [](Type type) {
    Type elementType = getElementType(type);
    return elementType.isF16() || elementType.isF32();
  };
  return llvm::all_of(op->getOperandTypes(), isHalfOrFloat) &&
         llvm::all_of(op->getResultTypes(), isHalfOrFloat);
}

bool TargetInfo::supportsVectorOperation(Operation *op) const {
  if (!hasVector())
    return false;
  switch (arch) {
  case Arch::RISCV:
    return supportsRISCVVectorOperation(op);
  case Arch::X86:
    return supportsX86VectorOperation(op);
  }
  llvm_unreachable("unknown target architecture");
}

bool TargetInfo::supportsRISCVVectorOperation(Operation *op) const {
  if (!hasSupportedVectorElementTypes(op, *this))
    return false;

  // Select moves values and a mask; it does not need an FP vector extension.
  if (mlir::isa<arith::SelectOp>(op))
    return true;

  // Integer arithmetic, comparisons, bit operations, shifts, and casts.
  if (mlir::isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::AndIOp,
                arith::OrIOp, arith::XOrIOp, arith::ShLIOp, arith::ShRSIOp,
                arith::ShRUIOp, arith::CmpIOp, arith::ExtSIOp, arith::ExtUIOp,
                arith::TruncIOp>(op))
    return true;

  auto hasSupportedFPWidths = [&] {
    auto supported = [&](Type type) {
      type = getElementType(type);
      return (!type.isF32() && !type.isF64()) ||
             type.getIntOrFloatBitWidth() <= isa->getMaxELenFp();
    };
    return llvm::all_of(op->getOperandTypes(), supported) &&
           llvm::all_of(op->getResultTypes(), supported);
  };

  // Zvfhmin permits only f16<->f32 conversions; f16<->f64 and f16<->integer
  // remain outside the conversion-only capability.
  if (mlir::isa<arith::ExtFOp, arith::TruncFOp>(op)) {
    if (!hasSupportedFPWidths())
      return false;
    return !containsF16(op) ||
           (isF16F32Conversion(op) &&
            (isa->hasExtension("zvfhmin") || isa->hasExtension("zvfh")));
  }

  // Integer/floating conversions involving f16 require Zvfh.
  if (mlir::isa<arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp,
                arith::FPToUIOp>(op))
    return hasSupportedFPWidths() &&
           (!containsF16(op) || isa->hasExtension("zvfh"));

  // f16 arithmetic, FMA, and comparison require Zvfh; wider formats use
  // ELEN-FP as checked above.
  if (mlir::isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
                arith::MaximumFOp, arith::MinimumFOp, arith::MaxNumFOp,
                arith::MinNumFOp, arith::NegFOp, arith::CmpFOp, math::FmaOp>(
          op))
    return hasSupportedFPWidths() &&
           (!containsF16(op) || isa->hasExtension("zvfh"));
  return false;
}

bool TargetInfo::supportsX86VectorOperation(Operation *op) const {
  if (!hasSupportedVectorElementTypes(op, *this))
    return false;

  // Select is data movement and does not need AVX512FP16 arithmetic.
  if (mlir::isa<arith::SelectOp>(op))
    return true;

  // Integer arithmetic, comparisons, bit operations, shifts, and casts.
  if (mlir::isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::AndIOp,
                arith::OrIOp, arith::XOrIOp, arith::ShLIOp, arith::ShRSIOp,
                arith::ShRUIOp, arith::CmpIOp, arith::ExtSIOp, arith::ExtUIOp,
                arith::TruncIOp>(op))
    return true;

  // F16C and AVX512F allow f16<->f32 conversion; AVX512FP16 also allows it.
  if (mlir::isa<arith::ExtFOp, arith::TruncFOp>(op))
    return !containsF16(op) ||
           (isF16F32Conversion(op) &&
            (hasExtension("f16c") || hasExtension("avx512f") ||
             hasExtension("avx512fp16")));

  // Integer/floating conversions involving f16 require AVX512FP16.
  if (mlir::isa<arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp,
                arith::FPToUIOp>(op))
    return !containsF16(op) || hasExtension("avx512fp16");

  // f32/f64 arithmetic is admitted; f16 arithmetic/FMA/comparison requires
  // AVX512FP16.
  if (mlir::isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
                arith::MaximumFOp, arith::MinimumFOp, arith::MaxNumFOp,
                arith::MinNumFOp, arith::NegFOp, arith::CmpFOp, math::FmaOp>(
          op))
    return !containsF16(op) || hasExtension("avx512fp16");
  return false;
}

unsigned TargetInfo::getRegisterGroupElements(Type type,
                                              unsigned registers) const {
  if (!canHoldVectorElementType(type) || registers == 0 ||
      registers > kMaxGroupRegisters)
    return 0;
  return getMinVectorRegisterBits() * registers / type.getIntOrFloatBitWidth();
}

std::string TargetInfo::describe() const {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "triple: " << triple << "\n";
  os << "chip: " << chip << "\n";
  switch (arch) {
  case Arch::RISCV:
    os << "xlen: " << isa->getXLen() << "\n";
    os << "vector: " << (hasVector() ? "yes" : "no") << "\n";
    os << "min-vlen: " << getMinVectorRegisterBits() << "\n";
    os << "elen: " << isa->getMaxELen() << "\n";
    os << "elen-fp: " << isa->getMaxELenFp() << "\n";
    os << "isa: " << isa->toString() << "\n";
    break;
  case Arch::X86: {
    os << "vector: " << (hasVector() ? "yes" : "no") << "\n";
    os << "vector-bits: " << getMinVectorRegisterBits() << "\n";
    SmallVector<StringRef> names(x86Features.keys());
    llvm::sort(names);
    os << "features: " << llvm::join(names, ",") << "\n";
    break;
  }
  }
  os << "max-vector-bits: " << getMaxVectorRegisterBits() << "\n";
  os << "exact-vector-bits: ";
  if (auto bits = getExactVectorRegisterBits())
    os << *bits;
  else
    os << "unknown";
  os << "\n";
  return text;
}

/// Splits `llvm.target`'s feature list into validated "+name"/"-name"
/// entries.
static LogicalResult collectFeatures(Operation *op, LLVM::TargetAttr target,
                                     SmallVectorImpl<StringRef> &out) {
  LLVM::TargetFeaturesAttr extra = target.getFeatures();
  if (!extra)
    return success();
  for (StringAttr feature : extra.getFeatures()) {
    StringRef name = feature.getValue();
    if (name.size() < 2 || (name[0] != '+' && name[0] != '-'))
      return op->emitError()
             << "target feature '" << name << "' must start with '+' or '-'";
    out.push_back(name);
  }
  return success();
}

static FailureOr<TargetInfo> getRISCVTargetInfo(Operation *op,
                                                LLVM::TargetAttr target,
                                                const llvm::Triple &triple) {
  unsigned xlen = triple.isRISCV64() ? 64 : 32;

  // The chip's extensions, as LLVM's processor table lists them.
  StringRef chip = target.getChip().getValue();
  std::vector<std::string> features;
  if (!chip.empty() && chip != "generic") {
    if (!llvm::RISCV::parseCPU(chip, xlen == 64))
      return op->emitError() << "unknown RISC-V processor '" << chip << "'";
    StringRef march = llvm::RISCV::getMArchFromMcpu(chip);
    auto chipISA = llvm::RISCVISAInfo::parseArchString(
        march, /*EnableExperimentalExtension=*/true);
    if (!chipISA)
      return op->emitError() << "cannot parse the ISA of '" << chip
                             << "': " << llvm::toString(chipISA.takeError());
    features = (*chipISA)->toFeatures(/*AddAllExtensions=*/false,
                                      /*IgnoreUnknown=*/true);
  }

  // Additions and removals on top of the chip, in order.
  SmallVector<StringRef> extra;
  if (failed(collectFeatures(op, target, extra)))
    return failure();
  for (StringRef name : extra) {
    if (!llvm::RISCVISAInfo::isSupportedExtensionFeature(name.drop_front()))
      return op->emitError()
             << "unknown RISC-V extension '" << name.drop_front() << "'";
    features.push_back(name.str());
  }

  auto isa = llvm::RISCVISAInfo::parseFeatures(xlen, features);
  if (!isa)
    return op->emitError() << "inconsistent target extensions: "
                           << llvm::toString(isa.takeError());
  ModuleOp module = mlir::isa<ModuleOp>(op) ? cast<ModuleOp>(op)
                                            : op->getParentOfType<ModuleOp>();
  unsigned vlen = 0;
  if (Attribute attr = module->getAttr("cpu.vlen")) {
    auto value = dyn_cast<IntegerAttr>(attr);
    if (!value || value.getValue().isNegative() ||
        value.getValue().getActiveBits() > 32)
      return op->emitError("cpu.vlen must be a non-negative 32-bit integer");
    vlen = value.getValue().getZExtValue();
    if (vlen && (vlen < 128 || !llvm::isPowerOf2_64(vlen)))
      return op->emitError("cpu.vlen must be zero or a power of two >= 128");
    if (vlen && (!(*isa)->hasExtension("zve32x") ||
                 vlen < (*isa)->getMinVLen() || vlen > (*isa)->getMaxVLen()))
      return op->emitError("cpu.vlen conflicts with the target VLEN range");
  }
  return TargetInfo(triple.str(), chip.empty() ? "generic" : chip.str(),
                    std::move(*isa), vlen);
}

static FailureOr<TargetInfo> getX86TargetInfo(Operation *op,
                                              LLVM::TargetAttr target,
                                              const llvm::Triple &triple) {
  bool is64 = triple.isArch64Bit();
  StringRef chip = target.getChip().getValue();
  // "generic" means the baseline of the triple, as it does for -mcpu.
  StringRef cpu = (chip.empty() || chip == "generic")
                      ? (is64 ? StringRef("x86-64") : StringRef("i686"))
                      : chip;
  if (llvm::X86::parseArchX86(cpu, /*Only64Bit=*/is64) == llvm::X86::CK_None)
    return op->emitError() << "unknown x86 processor '" << chip << "'";

  llvm::StringMap<bool> enabled;
  SmallVector<StringRef> cpuFeatures;
  llvm::X86::getFeaturesForCPU(cpu, cpuFeatures);
  for (StringRef f : cpuFeatures)
    enabled[f] = true;

  // Additions and removals on top of the chip, in order, with the features
  // they imply or that depend on them.
  SmallVector<StringRef> extra;
  if (failed(collectFeatures(op, target, extra)))
    return failure();
  for (StringRef name : extra) {
    StringRef feature = name.drop_front();
    llvm::StringMap<bool> probe;
    llvm::X86::updateImpliedFeatures(feature, /*Enabled=*/true, probe);
    if (probe.empty())
      return op->emitError() << "unknown x86 feature '" << feature << "'";
    llvm::X86::updateImpliedFeatures(feature, name[0] == '+', enabled);
  }

  llvm::StringSet<> features;
  for (const auto &entry : enabled)
    if (entry.second)
      features.insert(entry.first());
  return TargetInfo(triple.str(), chip.empty() ? "generic" : chip.str(),
                    std::move(features));
}

FailureOr<TargetInfo> mlir::buddy::getTargetInfo(Operation *op) {
  ModuleOp module =
      isa<ModuleOp>(op) ? cast<ModuleOp>(op) : op->getParentOfType<ModuleOp>();
  LLVM::TargetAttr target = module ? module->getAttrOfType<LLVM::TargetAttr>(
                                         LLVM::LLVMDialect::getTargetAttrName())
                                   : LLVM::TargetAttr();
  if (!target)
    return op->emitError() << "no '" << LLVM::LLVMDialect::getTargetAttrName()
                           << "' attribute on the enclosing module; the "
                              "driver must state the target";

  llvm::Triple triple(target.getTriple().getValue());
  if (triple.isRISCV())
    return getRISCVTargetInfo(op, target, triple);
  if (triple.isX86())
    return getX86TargetInfo(op, target, triple);
  return op->emitError() << "target triple '" << triple.str()
                         << "' is neither RISC-V nor x86";
}

//===----------------------------------------------------------------------===//
// --set-llvm-target
//===----------------------------------------------------------------------===//

namespace {
struct SetLLVMTargetPass
    : public PassWrapper<SetLLVMTargetPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SetLLVMTargetPass)

  SetLLVMTargetPass() = default;
  SetLLVMTargetPass(const SetLLVMTargetPass &other) : PassWrapper(other) {}

  StringRef getArgument() const final { return "set-llvm-target"; }
  StringRef getDescription() const final {
    return "Attach #llvm.target to the module and check that it describes a "
           "RISC-V or x86 target LLVM knows";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  Option<std::string> triple{*this, "triple", llvm::cl::desc("Target triple"),
                             llvm::cl::init("riscv64-unknown-linux-gnu")};
  Option<std::string> chip{*this, "chip",
                           llvm::cl::desc("LLVM processor name (-mcpu)"),
                           llvm::cl::init("generic")};
  Option<std::string> features{
      *this, "features",
      llvm::cl::desc("Comma-separated +ext/-ext list (-mattr)"),
      llvm::cl::init("")};

  Option<unsigned> vlen{
      *this, "vlen", llvm::cl::desc("Exact RISC-V VLEN in bits (0: unknown)"),
      llvm::cl::init(0)};

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    SmallVector<StringRef> list;
    StringRef(features).split(list, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    auto featuresAttr = list.empty() ? LLVM::TargetFeaturesAttr()
                                     : LLVM::TargetFeaturesAttr::get(ctx, list);
    auto target =
        LLVM::TargetAttr::get(ctx, StringAttr::get(ctx, triple),
                              StringAttr::get(ctx, chip), featuresAttr);
    getOperation()->setAttr(LLVM::LLVMDialect::getTargetAttrName(), target);
    if (vlen) {
      if (!llvm::Triple(triple).isRISCV()) {
        getOperation().emitError("vlen is only supported for RISC-V targets");
        return signalPassFailure();
      }
      getOperation()->setAttr(
          "cpu.vlen", IntegerAttr::get(IntegerType::get(ctx, 32), vlen));
    }
    if (failed(getTargetInfo(getOperation())))
      signalPassFailure();
  }
};
} // namespace

void mlir::buddy::registerSetLLVMTargetPass() {
  PassRegistration<SetLLVMTargetPass>();
}

//===----------------------------------------------------------------------===//
// --test-target-info
//===----------------------------------------------------------------------===//

namespace {
struct TestTargetInfoPass
    : public PassWrapper<TestTargetInfoPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTargetInfoPass)

  StringRef getArgument() const final { return "test-target-info"; }
  StringRef getDescription() const final {
    return "Print the target the module's llvm.target attribute describes";
  }

  void runOnOperation() override {
    FailureOr<TargetInfo> info = getTargetInfo(getOperation());
    if (failed(info))
      return signalPassFailure();
    llvm::outs() << info->describe();
    MLIRContext *ctx = &getContext();
    Type probes[] = {IntegerType::get(ctx, 8),  IntegerType::get(ctx, 32),
                     IntegerType::get(ctx, 64), Float16Type::get(ctx),
                     Float32Type::get(ctx),     BFloat16Type::get(ctx)};
    for (Type type : probes) {
      llvm::outs() << "elements " << type
                   << ": m1=" << info->getRegisterGroupElements(type, 1)
                   << " m8=" << info->getRegisterGroupElements(type) << "\n";
    }
  }
};
} // namespace

void mlir::buddy::registerTestTargetInfoPass() {
  PassRegistration<TestTargetInfoPass>();
}
