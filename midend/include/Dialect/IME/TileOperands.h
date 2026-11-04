//===- TileOperands.h - Operands of IME tile products ---------------------===//
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
// Shared analysis of the operands of a matrix product that may become one IME
// tile dot, at the tensor level (`linalg.matmul`) and at the vector level
// (`vector.contract`).
//
// An operand is traced from the product back to the load that produces it:
//
//   load -> zero-preserving casts -> [transpose -> zero-preserving casts]
//        -> product
//
// A zero-preserving cast is an elementwise single-input cast (an op with
// `CastOpInterface`, such as extf, extsi or bitcast), which maps 0 to 0. At
// the tensor level it is an all-parallel identity-map `linalg.generic` whose
// body is that one cast on its block argument. The transpose is the one the
// tile dot form itself has on its B operand; no other shape op is traced.
// Uses are left to the caller: a traced value may feed several products (CSE
// shares a scale operand between them), and a rewrite of the chain has to
// account for every one of them.
//
// The load is `bufferization.to_tensor` of a ranked memref view, or an
// unmasked, in-bounds, minor-identity `vector.transfer_read` of one. From its
// static strides, `canWidenAlongK` decides whether a product whose K is below
// the tile's can read a full-K operand with a mask instead: the operand's
// rows must lie the full K apart, and the K elements it reads must be
// consecutive.
//
//===----------------------------------------------------------------------===//

#ifndef IME_TILEOPERANDS_H
#define IME_TILEOPERANDS_H

#include "Dialect/IME/TileShapes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include <optional>

namespace buddy {
namespace ime {

/// If `v` is the result of a zero-preserving cast, returns that cast's
/// input; otherwise a null value.
mlir::Value getZeroPreservingCastInput(mlir::Value v);

/// Walks up zero-preserving casts from `v` and returns the first value that is
/// not such a cast. `chain`, if given, receives the values from `v` upwards,
/// excluding the returned one.
mlir::Value
traceZeroPreservingCasts(mlir::Value v,
                         llvm::SmallVectorImpl<mlir::Value> *chain = nullptr);

/// A tensor or vector load of a whole static-stride memref view.
struct TileLoad {
  mlir::Operation *op;
  /// The view the load reads.
  mlir::Value memref;
  mlir::MemRefType type;
};

/// `v` as a tile load, if it is one.
std::optional<TileLoad> getTileLoad(mlir::Value v);

/// One operand of a tile product, traced to its load.
struct TileOperand {
  TileLoad load;
  /// Dimensions of the loaded view that carry the operand's row (M for A, N
  /// for B) and its K.
  unsigned rowDim, kDim;
  /// Whether the trace passed the tile dot's own transpose of B.
  bool transposed;
  /// The values between the load result (excluded) and the product operand
  /// (included), from the operand upwards.
  llvm::SmallVector<mlir::Value> chain;
};

/// Traces operand `index` (0: A, 1: B) of a product to its load. A is
/// (M, K); B is (K, N), or the transpose of an (N, K) value.
std::optional<TileOperand> traceTileOperand(mlir::Value operand,
                                            unsigned index);

/// Whether `operand`'s load can be read as a (row, `kFull`) view with strides
/// (`kFull`, 1) at the same address: its rows are `kFull` apart and the K
/// elements it reads, at most `kFull`, are consecutive.
bool canWidenAlongK(const TileOperand &operand, int64_t kFull);

/// The tile shape of `shapes` with this M, N and element kind, whatever its
/// K; `isFloat` selects the fp16 dot.
std::optional<TileShape> findTileShapeMN(llvm::ArrayRef<TileShape> shapes,
                                         int64_t m, int64_t n, bool isFloat);

} // namespace ime
} // namespace buddy

namespace mlir {
namespace buddy {
void registerTestIMETileOperandsPass();
} // namespace buddy
} // namespace mlir

#endif // IME_TILEOPERANDS_H
