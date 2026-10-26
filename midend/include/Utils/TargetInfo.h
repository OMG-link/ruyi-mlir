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
// `chip` names an LLVM processor (`-mcpu`); its extensions come from LLVM's
// processor table. `features` adds or removes extensions on top of it
// (`-mattr`), or describes the target alone when `chip` is "generic". The ISA
// facts come from LLVM's RISCVISAInfo; this header only assembles them and
// derives the questions passes ask. The attribute is written by the driver
// (triton-riscv's compiler.py, through `--set-llvm-target`) and never
// inferred here.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_UTILS_TARGETINFO_H
#define BUDDY_UTILS_TARGETINFO_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

#include "llvm/TargetParser/RISCVISAInfo.h"

#include <memory>
#include <string>

namespace mlir {
namespace buddy {

class TargetInfo {
public:
  TargetInfo(std::string triple, std::string chip,
             std::unique_ptr<llvm::RISCVISAInfo> isa)
      : triple(std::move(triple)), chip(std::move(chip)), isa(std::move(isa)) {}

  StringRef getTriple() const { return triple; }
  StringRef getChip() const { return chip; }

  /// The full extension set, for queries the helpers below do not cover.
  const llvm::RISCVISAInfo &getISAInfo() const { return *isa; }
  bool hasExtension(StringRef ext) const { return isa->hasExtension(ext); }

  /// True when the target has a vector unit (V or a Zve* subset).
  bool hasVector() const;

  /// VLEN in bits. Zvl* gives a lower bound; the processor tables of LLVM
  /// state the actual VLEN of the cores this project targets.
  unsigned getVLenBits() const { return isa->getMinVLen(); }

  /// The largest register group a vector instruction can use.
  static constexpr unsigned kMaxLMUL = 8;

  /// True when vector instructions operate on elements of `type`.
  bool supportsVectorElementType(Type type) const;

  /// Elements of `type` that fit in a register group of `lmul` registers,
  /// or 0 when the type is not a supported vector element type.
  unsigned getRegisterGroupElements(Type type, unsigned lmul = kMaxLMUL) const;

  /// One line per fact, for the test pass and for remarks.
  std::string describe() const;

private:
  std::string triple;
  std::string chip;
  std::unique_ptr<llvm::RISCVISAInfo> isa;
};

/// Reads the target of the module enclosing `op`. Emits an error at `op` and
/// fails when the attribute is missing or does not describe a RISC-V target
/// LLVM knows.
FailureOr<TargetInfo> getTargetInfo(Operation *op);

/// `--set-llvm-target`: writes the attribute, then validates it.
void registerSetLLVMTargetPass();
void registerTestTargetInfoPass();

} // namespace buddy
} // namespace mlir

#endif // BUDDY_UTILS_TARGETINFO_H
