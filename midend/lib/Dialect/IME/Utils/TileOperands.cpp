//===- TileOperands.cpp - Operands of IME tile products -------------------===//
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

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SetVector.h"

#include "Dialect/Cpu/CpuOps.h"
#include "Dialect/IME/TileOperands.h"

using namespace mlir;

namespace buddy {
namespace ime {

/// An elementwise op with `CastOpInterface`: one input, one result, and 0 in
/// gives 0 out.
static bool isZeroPreservingCast(Operation *op) {
  return op && op->getNumOperands() == 1 && op->getNumResults() == 1 &&
         op->hasTrait<OpTrait::Elementwise>() && isa<CastOpInterface>(op);
}

Value getZeroPreservingCastInput(Value v) {
  Operation *def = v.getDefiningOp();
  if (!def)
    return {};
  // Vector level: the cast op itself, on a vector.
  if (isa<VectorType>(v.getType()))
    return isZeroPreservingCast(def) ? def->getOperand(0) : Value();
  // Tensor level: a generic applying one such cast to its only input.
  auto generic = dyn_cast<linalg::GenericOp>(def);
  if (!generic || generic.getNumDpsInputs() != 1 ||
      generic.getNumResults() != 1 || generic.getNumReductionLoops() != 0 ||
      !llvm::all_of(generic.getIndexingMapsArray(),
                    [](AffineMap map) { return map.isIdentity(); }))
    return {};
  Block *body = generic.getBody();
  if (!llvm::hasSingleElement(body->without_terminator()))
    return {};
  Operation &cast = body->front();
  if (!isZeroPreservingCast(&cast) ||
      cast.getOperand(0) != body->getArgument(0) ||
      body->getTerminator()->getOperand(0) != cast.getResult(0))
    return {};
  return generic.getDpsInputs()[0];
}

/// The operand a `cpu.sched_barrier` result holds, or a null value.
static Value getBarrierInput(Value v) {
  auto result = dyn_cast<OpResult>(v);
  if (!result)
    return {};
  auto barrier = dyn_cast<::buddy::cpu::SchedBarrierOp>(result.getOwner());
  return barrier ? barrier.getInputs()[result.getResultNumber()] : Value();
}

/// The init of `arg` if it is an `scf.for` iter_arg the loop yields back
/// unchanged, through barriers only, and whose loop result is unused.
/// `yielded` receives the values between the yield operand (included) and
/// `arg` (excluded).
static Value getLoopCarriedInit(BlockArgument arg,
                                SmallVectorImpl<Value> *yielded) {
  auto forOp = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
  if (!forOp)
    return {};
  OpOperand *init = forOp.getTiedLoopInit(arg);
  if (!init || !forOp.getTiedLoopResult(arg).use_empty())
    return {};
  SmallVector<Value> path;
  Value back = forOp.getTiedLoopYieldedValue(arg)->get();
  while (back != arg) {
    path.push_back(back);
    back = getBarrierInput(back);
    if (!back)
      return {};
  }
  if (yielded)
    yielded->append(path);
  return init->get();
}

static Value getValueIdentityInput(Value v, SmallVectorImpl<Value> *yielded) {
  if (Value input = getBarrierInput(v))
    return input;
  if (auto arg = dyn_cast<BlockArgument>(v))
    return getLoopCarriedInit(arg, yielded);
  return {};
}

Value getTransparentInput(Value v) {
  if (Value input = getZeroPreservingCastInput(v))
    return input;
  return getValueIdentityInput(v, nullptr);
}

Value getForwardedValue(OpOperand &use) {
  Operation *user = use.getOwner();
  if (auto barrier = dyn_cast<::buddy::cpu::SchedBarrierOp>(user))
    return barrier.getResults()[use.getOperandNumber()];
  if (auto forOp = dyn_cast<scf::ForOp>(user))
    return forOp.getTiedLoopRegionIterArg(&use);
  if (auto yield = dyn_cast<scf::YieldOp>(user)) {
    auto forOp = dyn_cast<scf::ForOp>(yield->getParentOp());
    return forOp ? forOp.getRegionIterArgs()[use.getOperandNumber()] : Value();
  }
  if (auto transpose = dyn_cast<vector::TransposeOp>(user))
    return transpose.getPermutation() == ArrayRef<int64_t>{1, 0}
               ? transpose.getResult()
               : Value();
  if (auto transpose = dyn_cast<linalg::TransposeOp>(user))
    return use.get() == transpose.getInput() &&
                   transpose.getPermutation() == ArrayRef<int64_t>{1, 0}
               ? transpose->getResult(0)
               : Value();
  if (user->getNumResults() == 1 &&
      getZeroPreservingCastInput(user->getResult(0)) == use.get())
    return user->getResult(0);
  return {};
}

Value traceTransparentSteps(Value v, SmallVectorImpl<Value> *chain) {
  SmallPtrSet<Value, 8> seen;
  while (seen.insert(v).second) {
    SmallVector<Value> yielded;
    Value input = getZeroPreservingCastInput(v);
    if (!input)
      input = getValueIdentityInput(v, &yielded);
    if (!input)
      return v;
    if (chain) {
      chain->push_back(v);
      chain->append(yielded);
    }
    v = input;
  }
  return v;
}

std::optional<TileLoad> getTileLoad(Value v) {
  Value memref;
  Operation *op = v.getDefiningOp();
  if (auto load = dyn_cast_or_null<bufferization::ToTensorOp>(op)) {
    memref = load.getBuffer();
  } else if (auto read = dyn_cast_or_null<vector::TransferReadOp>(op)) {
    if (read.getMask() || read.hasOutOfBoundsDim() ||
        !read.getPermutationMap().isMinorIdentity() ||
        llvm::any_of(read.getIndices(), [](Value index) {
          return !matchPattern(index, m_Zero());
        }))
      return std::nullopt;
    memref = read.getBase();
  } else {
    return std::nullopt;
  }
  auto type = dyn_cast<MemRefType>(memref.getType());
  if (!type || !type.hasStaticShape() ||
      type.getShape() != cast<ShapedType>(v.getType()).getShape())
    return std::nullopt;
  return TileLoad{op, memref, type};
}

std::optional<TileOperand> traceTileOperand(Value operand, unsigned index) {
  assert(index < 2 && "a product has two operands");
  TileOperand result;
  Value v = traceTransparentSteps(operand, &result.chain);
  result.transposed = false;
  if (index == 1) {
    Value input;
    if (auto transpose = v.getDefiningOp<linalg::TransposeOp>()) {
      if (transpose.getPermutation() == ArrayRef<int64_t>{1, 0})
        input = transpose.getInput();
    } else if (auto transpose = v.getDefiningOp<vector::TransposeOp>()) {
      if (transpose.getPermutation() == ArrayRef<int64_t>{1, 0})
        input = transpose.getVector();
    }
    if (input) {
      result.chain.push_back(v);
      v = traceTransparentSteps(input, &result.chain);
      result.transposed = true;
    }
  }
  std::optional<TileLoad> load = getTileLoad(v);
  if (!load || load->type.getRank() != 2)
    return std::nullopt;
  // A yield path may run along the operand's own path.
  llvm::SetVector<Value> unique(result.chain.begin(), result.chain.end());
  result.chain.assign(unique.begin(), unique.end());
  result.load = *load;
  // A is (M, K). B is (K, N) as given, or (N, K) under its transpose.
  bool rowFirst = index == 0 || result.transposed;
  result.rowDim = rowFirst ? 0 : 1;
  result.kDim = rowFirst ? 1 : 0;
  return result;
}

bool canWidenAlongK(const TileOperand &operand, int64_t kFull) {
  SmallVector<int64_t> strides;
  int64_t offset;
  MemRefType type = operand.load.type;
  if (failed(type.getStridesAndOffset(strides, offset)))
    return false;
  // The widened operand is a (row, kFull) view at the same address with
  // strides (kFull, 1): the rows the load reads, each read further along.
  // So rows must be kFull apart, and the K elements already read must be
  // consecutive; a K extent of one has no stride to check.
  int64_t k = type.getDimSize(operand.kDim);
  return strides[operand.rowDim] == kFull && k <= kFull &&
         (k == 1 || strides[operand.kDim] == 1);
}

std::optional<TileShape> findTileShapeMN(ArrayRef<TileShape> shapes, int64_t m,
                                         int64_t n, bool isFloat) {
  for (const TileShape &shape : shapes)
    if (shape.m == m && shape.n == n && shape.isFloat == isFloat)
      return shape;
  return std::nullopt;
}

} // namespace ime
} // namespace buddy

//===----------------------------------------------------------------------===//
// --test-ime-tile-operands
//===----------------------------------------------------------------------===//

namespace {
/// Prints, prefixed by the enclosing symbol, for every 2-D `linalg.matmul` and
/// `vector.contract` in program order, the target tile shape with its M and N,
/// and for each operand the trace result and whether it can be widened to that
/// tile's K.
struct TestIMETileOperandsPass
    : public PassWrapper<TestIMETileOperandsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestIMETileOperandsPass)
  StringRef getArgument() const final { return "test-ime-tile-operands"; }
  StringRef getDescription() const final {
    return "Print the IME tile operand analysis of every matrix product";
  }

  void runOnOperation() override {
    FailureOr<mlir::buddy::TargetInfo> target =
        mlir::buddy::getTargetInfo(getOperation());
    if (failed(target))
      return signalPassFailure();
    SmallVector<::buddy::ime::TileShape, 2> shapes =
        ::buddy::ime::getTileShapes(*target);
    getOperation().walk([&](Operation *op) {
      Value lhs, rhs, acc;
      if (auto matmul = dyn_cast<linalg::MatmulOp>(op)) {
        lhs = matmul.getDpsInputs()[0];
        rhs = matmul.getDpsInputs()[1];
        acc = matmul.getDpsInits()[0];
      } else if (auto contract = dyn_cast<vector::ContractionOp>(op)) {
        lhs = contract.getLhs();
        rhs = contract.getRhs();
        acc = contract.getAcc();
      } else {
        return;
      }
      auto accType = dyn_cast<ShapedType>(acc.getType());
      auto lhsType = cast<ShapedType>(lhs.getType());
      if (!accType || accType.getRank() != 2 || lhsType.getRank() != 2)
        return;
      int64_t m = accType.getDimSize(0), n = accType.getDimSize(1);
      int64_t k = lhsType.getDimSize(1);
      bool isFloat = isa<FloatType>(accType.getElementType());
      if (auto symbol = op->getParentOfType<SymbolOpInterface>())
        llvm::outs() << "@" << symbol.getName() << " ";
      llvm::outs() << op->getName() << " " << m << "x" << n << "x" << k
                   << (isFloat ? " float" : " int") << ": ";
      std::optional<::buddy::ime::TileShape> shape =
          ::buddy::ime::findTileShapeMN(shapes, m, n, isFloat);
      if (shape)
        llvm::outs() << "tile k=" << shape->k << "\n";
      else
        llvm::outs() << "no tile\n";
      const char *names[] = {"A", "B"};
      Value operands[] = {lhs, rhs};
      for (unsigned i = 0; i < 2; ++i) {
        llvm::outs() << "  " << names[i] << ": ";
        std::optional<::buddy::ime::TileOperand> operand =
            ::buddy::ime::traceTileOperand(operands[i], i);
        if (!operand) {
          llvm::outs() << "untraced\n";
          continue;
        }
        llvm::outs() << operand->load.op->getName() << " " << operand->load.type
                     << " row=" << operand->rowDim << " k=" << operand->kDim
                     << " transposed=" << operand->transposed
                     << " chain=" << operand->chain.size();
        if (shape)
          llvm::outs() << " widen="
                       << ::buddy::ime::canWidenAlongK(*operand, shape->k);
        llvm::outs() << "\n";
      }
    });
  }
};
} // namespace

void mlir::buddy::registerTestIMETileOperandsPass() {
  PassRegistration<TestIMETileOperandsPass>();
}
