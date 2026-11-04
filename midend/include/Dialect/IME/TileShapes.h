//===- TileShapes.h - Register tile shapes of the SpacemiT IME dots -------===//
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
// One IME dot reads A and B from one vector register each and accumulates
// into an aligned register pair, so its tile shape follows from VLEN:
//
//   K1   (XSMTVDot,   VLEN 256)   int8 4x8  * (4x8)^T  -> int32 4x4
//   A100 (XSMTVDotII, VLEN 1024)  int8 8x16 * (8x16)^T -> int32 8x8
//                                 fp16 8x8  * (8x8)^T  -> fp32  8x8
//
// B is stored N x K. The shapes need the exact VLEN; a target whose VLEN is
// only bounded has none.
//
//===----------------------------------------------------------------------===//

#ifndef IME_TILESHAPES_H
#define IME_TILESHAPES_H

#include "Utils/TargetInfo.h"

#include "llvm/ADT/SmallVector.h"

namespace buddy {
namespace ime {

struct TileShape {
  int64_t m, n, k;
  /// fp16 inputs into fp32, rather than int8 into int32.
  bool isFloat;
};

/// The IME dot tile shapes the target executes as one instruction.
inline llvm::SmallVector<TileShape, 2>
getTileShapes(const mlir::buddy::TargetInfo &target) {
  if (target.getArch() != mlir::buddy::TargetInfo::Arch::RISCV)
    return {};
  std::optional<unsigned> vlen = target.getExactVectorRegisterBits();
  if (target.hasExtension("xsmtvdotii") && vlen == 1024u)
    return {{8, 8, 16, false}, {8, 8, 8, true}};
  if (target.hasExtension("xsmtvdot") && vlen == 256u)
    return {{4, 4, 8, false}};
  return {};
}

} // namespace ime
} // namespace buddy

#endif // IME_TILESHAPES_H
