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

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "Utils/TargetInfo.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/RISCVTargetParser.h"
#include "llvm/TargetParser/Triple.h"

using namespace mlir;
using namespace mlir::buddy;

bool TargetInfo::hasVector() const { return isa->hasExtension("zve32x"); }

bool TargetInfo::supportsVectorElementType(Type type) const {
  if (!hasVector())
    return false;
  if (auto intType = dyn_cast<IntegerType>(type)) {
    unsigned width = intType.getWidth();
    return (width == 8 || width == 16 || width == 32 || width == 64) &&
           width <= isa->getMaxELen();
  }
  if (type.isF16())
    return isa->hasExtension("zvfh");
  if (type.isF32() || type.isF64())
    return type.getIntOrFloatBitWidth() <= isa->getMaxELenFp();
  // bf16 has only conversions (Zvfbfmin) and widening FMA (Zvfbfwma), and
  // i1 lives in mask registers; neither is a general vector element type.
  return false;
}

unsigned TargetInfo::getRegisterGroupElements(Type type, unsigned lmul) const {
  if (!supportsVectorElementType(type) || lmul == 0 || lmul > kMaxLMUL)
    return 0;
  return getVLenBits() * lmul / type.getIntOrFloatBitWidth();
}

std::string TargetInfo::describe() const {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "triple: " << triple << "\n";
  os << "chip: " << chip << "\n";
  os << "xlen: " << isa->getXLen() << "\n";
  os << "vector: " << (hasVector() ? "yes" : "no") << "\n";
  os << "vlen: " << getVLenBits() << "\n";
  os << "elen: " << isa->getMaxELen() << "\n";
  os << "elen-fp: " << isa->getMaxELenFp() << "\n";
  os << "isa: " << isa->toString() << "\n";
  return text;
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
  if (!triple.isRISCV())
    return op->emitError() << "target triple '" << triple.str()
                           << "' is not RISC-V";
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
  if (LLVM::TargetFeaturesAttr extra = target.getFeatures()) {
    for (StringAttr feature : extra.getFeatures()) {
      StringRef name = feature.getValue();
      if (name.size() < 2 || (name[0] != '+' && name[0] != '-'))
        return op->emitError()
               << "target feature '" << name << "' must start with '+' or '-'";
      if (!llvm::RISCVISAInfo::isSupportedExtensionFeature(name.drop_front()))
        return op->emitError()
               << "unknown RISC-V extension '" << name.drop_front() << "'";
      features.push_back(name.str());
    }
  }

  auto isa = llvm::RISCVISAInfo::parseFeatures(xlen, features);
  if (!isa)
    return op->emitError() << "inconsistent target extensions: "
                           << llvm::toString(isa.takeError());
  return TargetInfo(triple.str(), chip.empty() ? "generic" : chip.str(),
                    std::move(*isa));
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
           "RISC-V target LLVM knows";
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
