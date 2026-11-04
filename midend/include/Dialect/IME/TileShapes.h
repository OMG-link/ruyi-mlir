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
// One IME dot reads A (M x K) and B (stored N x K) from one vector register
// each and accumulates into C (M x N) in an aligned register pair. The tile
// shape follows from two facts of the target, never from the chip name:
//
//   * the extension gives the input types: XSMTVDot has the int8 dot,
//     XSMTVDotII the int8 and the widening fp16 dot (it has more, not used
//     yet);
//   * the exact VLEN gives M = N: 4 at VLEN 256, 8 at VLEN 1024, and K
//     fills one register, K = VLEN / (M * SEW).
//
// So the K1 (X60, XSMTVDot, VLEN 256) has the int8 4x8 tile and the A100
// (XSMTVDotII, VLEN 1024) the int8 8x16 and fp16 8x8 tiles. A target with
// another VLEN, or whose VLEN is only bounded, has none.
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
  bool dotII = target.hasExtension("xsmtvdotii");
  if (!dotII && !target.hasExtension("xsmtvdot"))
    return {};
  std::optional<unsigned> vlen = target.getExactVectorRegisterBits();
  int64_t mn;
  if (vlen == 256u)
    mn = 4;
  else if (vlen == 1024u)
    mn = 8;
  else
    return {};
  // K fills one register of SEW-bit elements.
  auto shape = [&](unsigned sew, bool isFloat) {
    return TileShape{mn, mn, static_cast<int64_t>(*vlen / (mn * sew)), isFloat};
  };
  llvm::SmallVector<TileShape, 2> shapes = {shape(8, /*isFloat=*/false)};
  if (dotII)
    shapes.push_back(shape(16, /*isFloat=*/true));
  return shapes;
}

} // namespace ime
} // namespace buddy

#endif // IME_TILESHAPES_H
