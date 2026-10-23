//====- IMEDialect.cpp - MLIR IME dialect implementation ------------------===//
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

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "Dialect/IME/IMEDialect.h"
#include "Dialect/IME/IMEOps.h"

using namespace mlir;
using namespace buddy::ime;

#include "IME/IMEDialect.cpp.inc"

#define GET_OP_CLASSES
#include "IME/IME.cpp.inc"

void IMEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "IME/IME.cpp.inc"
      >();
}

namespace {
LogicalResult verifyDot(Operation *op) {
  Type result = op->getResult(0).getType();
  if (result != op->getOperand(0).getType())
    return op->emitOpError("result and accumulator must have the same type");
  for (Value value : op->getOperands()) {
    auto vector = dyn_cast<VectorType>(value.getType());
    if (!vector || vector.getRank() != 1 || !vector.isScalable())
      return op->emitOpError("requires one-dimensional scalable vector tiles");
  }
  // One dot reads whole registers: the inputs one register each (the
  // sliding-window A operand an aligned pair), the accumulator a pair. The
  // tile shape follows from VLEN (4x8 int8 on the K1, 8x16 on the A100), so
  // the types are the same on every target: int8 inputs are vector<[8]xi8>
  // (vector<[16]xi8> for a window A) into vector<[4]xi32>; the FP16 dot takes
  // vector<[4]xf16> into vector<[4]xf32>.
  auto c = cast<VectorType>(result);
  auto a = cast<VectorType>(op->getOperand(1).getType());
  auto b = cast<VectorType>(op->getOperand(2).getType());
  auto *ctx = op->getContext();
  StringRef name = op->getName().getStringRef();
  bool fp = name == "ime.intr.vfmadot";
  bool window = !fp && name != "ime.intr.vmadot" &&
                name != "ime.intr.vmadotu" && name != "ime.intr.vmadotsu" &&
                name != "ime.intr.vmadotus";
  Type input =
      fp ? Type(Float16Type::get(ctx)) : Type(IntegerType::get(ctx, 8));
  Type acc = fp ? Type(Float32Type::get(ctx)) : Type(IntegerType::get(ctx, 32));
  int64_t lanes = fp ? 4 : 8;
  if (c != VectorType::get({4}, acc, true) ||
      b != VectorType::get({lanes}, input, true) ||
      a != VectorType::get({window ? 2 * lanes : lanes}, input, true))
    return op->emitOpError("invalid dot tile types");
  return success();
}

LogicalResult verifyA100(Operation *op) {
  auto a = cast<VectorType>(op->getOperand(0).getType());
  auto b = cast<VectorType>(op->getOperand(1).getType());
  auto out = cast<VectorType>(op->getResult(0).getType());
  auto isTile = [](VectorType type) {
    return type.getRank() == 1 && type.isScalable() &&
           type.getElementType().isSignlessInteger();
  };
  if (!isTile(a) || !isTile(b) || !isTile(out) || a != b)
    return op->emitOpError(
        "requires matching one-dimensional scalable integer input tiles");
  StringRef name = op->getName().getStringRef();
  unsigned width = a.getElementTypeBitWidth();
  unsigned destWidth = out.getElementTypeBitWidth();
  bool wide = name == "ime.intr.vpack" || name == "ime.intr.vupack";
  bool nibble = name.ends_with("4");
  if (wide) {
    if ((width != 8 && width != 16 && width != 32 && width != 64) ||
        a.getNumElements() * width != 64 || destWidth != width ||
        out.getNumElements() != 2 * a.getNumElements())
      return op->emitOpError("pack/upack requires one-register inputs and a "
                             "same-width register-pair result");
  } else if (nibble) {
    if (width != 8 || a.getNumElements() != 8 || out != a)
      return op->emitOpError(
          "4-bit pack requires vector<[8]xi8> inputs and result");
  } else if ((width != 16 && width != 32 && width != 64) ||
             a.getNumElements() * width != 64 || destWidth * 2 != width ||
             out.getNumElements() != 2 * a.getNumElements()) {
    return op->emitOpError("narrow pack requires one-register inputs and a "
                           "result with twice the lanes and half the width");
  }
  return success();
}
} // namespace

LogicalResult Vmadot_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vmadotu_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vmadotsu_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vmadotus_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vfmadot_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vmadot1_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vmadot1u_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vmadot1su_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vmadot1us_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vmadot2_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vmadot2u_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vmadot2su_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vmadot2us_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vmadot3_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vmadot3u_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vmadot3su_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vmadot3us_IntrOp::verify() { return verifyDot(getOperation()); }

LogicalResult Vpack_IntrOp::verify() { return verifyA100(getOperation()); }

LogicalResult Vupack_IntrOp::verify() { return verifyA100(getOperation()); }

LogicalResult Vnpack_IntrOp::verify() { return verifyA100(getOperation()); }

LogicalResult Vnspack_IntrOp::verify() { return verifyA100(getOperation()); }

LogicalResult Vnpack4_IntrOp::verify() { return verifyA100(getOperation()); }

LogicalResult Vnspack4_IntrOp::verify() { return verifyA100(getOperation()); }
