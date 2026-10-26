//===- TargetInfo.h - Hardware facts of the compilation target ------------===//
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
// The hardware a pass compiles for, read from the `llvm.target` attribute of
// the enclosing module:
//
//   module attributes {llvm.target = #llvm.target<
//       triple = "riscv64-unknown-linux-gnu", chip = "spacemit-x60",
//       features = #llvm.target_features<["+xsmtvdot"]>>} { ... }
//
// RISC-V and x86 are supported. `chip` names an LLVM processor (`-mcpu`);
// its features come from LLVM's processor table. `features` adds or removes
// features on top of it (`-mattr`), or describes the target alone when
// `chip` is "generic". The facts come from LLVM's TargetParser (RISCVISAInfo
// for RISC-V, the X86 processor and feature tables for x86); this header only
// assembles them and derives the questions passes ask. The attribute is
// written by the driver (triton-riscv's compiler.py, through
// `--set-llvm-target`) and never inferred here.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_UTILS_TARGETINFO_H
#define BUDDY_UTILS_TARGETINFO_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/TargetParser/RISCVISAInfo.h"

#include <memory>
#include <string>

namespace mlir {
namespace buddy {

class TargetInfo {
public:
  enum class Arch { RISCV, X86 };

  /// A RISC-V target described by its ISA.
  TargetInfo(std::string triple, std::string chip,
             std::unique_ptr<llvm::RISCVISAInfo> isa)
      : arch(Arch::RISCV), triple(std::move(triple)), chip(std::move(chip)),
        isa(std::move(isa)) {}
  /// An x86 target described by its enabled feature set.
  TargetInfo(std::string triple, std::string chip, llvm::StringSet<> features)
      : arch(Arch::X86), triple(std::move(triple)), chip(std::move(chip)),
        x86Features(std::move(features)) {}

  Arch getArch() const { return arch; }
  StringRef getTriple() const { return triple; }
  StringRef getChip() const { return chip; }

  /// The full RISC-V extension set, for queries the helpers below do not
  /// cover. Only valid for RISC-V targets.
  const llvm::RISCVISAInfo &getISAInfo() const {
    assert(arch == Arch::RISCV && "not a RISC-V target");
    return *isa;
  }
  /// A RISC-V extension ("zvfh") or an x86 feature ("avx512f").
  bool hasExtension(StringRef ext) const;

  /// True when the target has a vector unit: V or a Zve* subset on RISC-V,
  /// SSE2 on x86.
  bool hasVector() const;

  /// Width of one vector register in bits. On RISC-V this is VLEN: Zvl*
  /// gives a lower bound, and LLVM's processor tables state the actual VLEN
  /// of the cores this project targets. On x86 it is the widest of XMM, YMM
  /// and ZMM the features enable.
  unsigned getVectorRegisterBits() const;

  /// The most registers one vector value may span: LMUL=8 on RISC-V, and
  /// the same count on x86, where LLVM splits a wide vector across
  /// consecutive registers.
  static constexpr unsigned kMaxGroupRegisters = 8;

  /// True when vector instructions operate on elements of `type`.
  bool supportsVectorElementType(Type type) const;

  /// Elements of `type` that fit in a group of `registers` vector registers,
  /// or 0 when the type is not a supported vector element type.
  unsigned
  getRegisterGroupElements(Type type,
                           unsigned registers = kMaxGroupRegisters) const;

  /// One line per fact, for the test pass and for remarks.
  std::string describe() const;

private:
  Arch arch;
  std::string triple;
  std::string chip;
  std::unique_ptr<llvm::RISCVISAInfo> isa;
  llvm::StringSet<> x86Features;
};

/// Reads the target of the module enclosing `op`. Emits an error at `op` and
/// fails when the attribute is missing or does not describe a RISC-V or x86
/// target LLVM knows.
FailureOr<TargetInfo> getTargetInfo(Operation *op);

/// `--set-llvm-target`: writes the attribute, then validates it.
void registerSetLLVMTargetPass();
void registerTestTargetInfoPass();

} // namespace buddy
} // namespace mlir

#endif // BUDDY_UTILS_TARGETINFO_H
