//===- PromoteIMETileK.cpp - Widen K-short IME tile products --------------===//
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
// `--promote-ime-tile-k` widens a tile product whose K is below the IME
// tile's to the tile's full K, so that `--lower-vector-contract-to-ime-tile`
// maps it onto one dot. The product is the form that lowering accepts except
// for K:
//
//   C(M x N) + sum_k ext(A[m,k]) * ext(B'[k,n]),  k < K
//
// where ext(A) is the extension of a value that traces
// (`Dialect/IME/TileOperands.h`) to an unmasked read of an (M, k) view, and
// ext(B') is either the extension of a transpose of a value that traces to
// a read of an (N, k) view, or the extension of a read of a (k, N) view
// itself. Each read becomes a masked read of the same address as an (M, K)
// or (N, K) view with strides (K, 1), the mask covering the original k
// columns and padding 0, which `canWidenAlongK` decides is the same data; a
// (k, N) read is transposed right after. Every value on the traced path,
// casts, `cpu.sched_barrier` results and `scf.for` iter_args included, is
// then retyped in place to the full K, and the contraction runs over K. The
// extra K columns are zero on both sides and every step on the path keeps
// zero, so their products add nothing.
//
// A read is widened once for every product that reads it, and only when
// every reader, through the path, is such a product, at the same operand,
// with the same tile K, B orientation and extension; otherwise none of them
// is. The products are only rewritten when the result is the tile form the
// IME lowering takes: plain matmul maps, an add, the same extension on both
// operands (extsi or extui from i8 into an i32 C, extf from f16 into an f32
// C) right at the product, B's transpose right below it, and a tile shape of
// the target.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "Dialect/Cpu/CpuOps.h"
#include "Dialect/IME/TileOperands.h"
#include "Dialect/IME/TileShapes.h"
#include "Utils/TargetInfo.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

namespace {

/// One operand of a K-short tile product.
struct KShortOperand {
  ::buddy::ime::TileOperand trace;
  /// The extension right at the product.
  Operation *ext;
};

/// A K-short tile product the IME lowering takes once K is widened.
struct KShortProduct {
  vector::ContractionOp contract;
  KShortOperand operands[2];
  ::buddy::ime::TileShape shape;
};

/// The extension that defines `v`, if one the tile dots take.
Operation *getTileExtension(Value v) {
  Operation *def = v.getDefiningOp();
  return isa_and_nonnull<arith::ExtSIOp, arith::ExtUIOp, arith::ExtFOp>(def)
             ? def
             : nullptr;
}

std::optional<KShortProduct>
matchKShortProduct(vector::ContractionOp contract,
                   ArrayRef<::buddy::ime::TileShape> shapes) {
  if (contract.getKind() != vector::CombiningKind::ADD)
    return std::nullopt;
  MLIRContext *ctx = contract.getContext();
  AffineExpr m, n, k;
  bindDims(ctx, m, n, k);
  SmallVector<AffineMap> matmul = {AffineMap::get(3, 0, {m, k}, ctx),
                                   AffineMap::get(3, 0, {k, n}, ctx),
                                   AffineMap::get(3, 0, {m, n}, ctx)};
  if (contract.getIndexingMapsArray() != ArrayRef<AffineMap>(matmul))
    return std::nullopt;
  auto accTy = dyn_cast<VectorType>(contract.getAcc().getType());
  auto lhsTy = cast<VectorType>(contract.getLhs().getType());
  if (!accTy || accTy.isScalable() || lhsTy.isScalable())
    return std::nullopt;

  KShortProduct product;
  product.contract = contract;
  Value operands[] = {contract.getLhs(), contract.getRhs()};
  for (unsigned i = 0; i < 2; ++i) {
    Operation *ext = getTileExtension(operands[i]);
    if (!ext)
      return std::nullopt;
    std::optional<::buddy::ime::TileOperand> trace =
        ::buddy::ime::traceTileOperand(operands[i], i);
    if (!trace || !isa<vector::TransferReadOp>(trace->load.op))
      return std::nullopt;
    if (i == 1) {
      // The lowering reads B as ext(transpose(B)). A transposed path must
      // end in that transpose; a direct one is the read itself, which the
      // widening transposes.
      Value in = ext->getOperand(0);
      auto transpose = in.getDefiningOp<vector::TransposeOp>();
      if (trace->transposed ? !transpose : in != trace->load.op->getResult(0))
        return std::nullopt;
    }
    product.operands[i] = {*trace, ext};
  }
  Operation *extA = product.operands[0].ext;
  Operation *extB = product.operands[1].ext;
  if (extA->getName() != extB->getName())
    return std::nullopt;
  bool isFloat = isa<arith::ExtFOp>(extA);
  Type in = isFloat ? Type(Float16Type::get(ctx)) : IntegerType::get(ctx, 8);
  Type out = isFloat ? Type(Float32Type::get(ctx)) : IntegerType::get(ctx, 32);
  if (accTy.getElementType() != out ||
      getElementTypeOrSelf(extA->getOperand(0)) != in ||
      getElementTypeOrSelf(extB->getOperand(0)) != in ||
      getElementTypeOrSelf(extA->getResult(0)) != out ||
      getElementTypeOrSelf(extB->getResult(0)) != out)
    return std::nullopt;

  std::optional<::buddy::ime::TileShape> shape = ::buddy::ime::findTileShapeMN(
      shapes, accTy.getDimSize(0), accTy.getDimSize(1), isFloat);
  if (!shape || lhsTy.getDimSize(1) >= shape->k)
    return std::nullopt;
  for (const KShortOperand &operand : product.operands)
    if (!::buddy::ime::canWidenAlongK(operand.trace, shape->k))
      return std::nullopt;
  product.shape = *shape;
  return product;
}

/// How a product reads one value of an operand chain. Two products may
/// share a value only when they read it the same way, so that one widened
/// chain serves both.
struct Claim {
  unsigned index;
  int64_t kTile;
  bool transposed;
  OperationName ext;
  bool operator==(const Claim &other) const {
    return index == other.index && kTile == other.kTile &&
           transposed == other.transposed && ext == other.ext;
  }
  bool operator!=(const Claim &other) const { return !(*this == other); }
};

Claim getClaim(const KShortProduct &p, unsigned i) {
  const KShortOperand &operand = p.operands[i];
  return {i, p.shape.k, operand.trace.transposed, operand.ext->getName()};
}

/// The read and the chain values of operand `i`.
SmallVector<Value> getChainValues(const KShortProduct &p, unsigned i) {
  SmallVector<Value> values{p.operands[i].trace.load.op->getResult(0)};
  llvm::append_range(values, p.operands[i].trace.chain);
  return values;
}

class PromoteIMETileKPass
    : public PassWrapper<PromoteIMETileKPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PromoteIMETileKPass)
  StringRef getArgument() const final { return "promote-ime-tile-k"; }
  StringRef getDescription() const final {
    return "Widen K-short IME tile contractions to the tile's K with masked "
           "reads";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override {
    FailureOr<mlir::buddy::TargetInfo> target =
        mlir::buddy::getTargetInfo(getOperation());
    if (failed(target))
      return signalPassFailure();
    SmallVector<::buddy::ime::TileShape, 2> shapes =
        ::buddy::ime::getTileShapes(*target);
    if (shapes.empty())
      return;

    SmallVector<KShortProduct, 4> products;
    getOperation().walk([&](vector::ContractionOp contract) {
      if (std::optional<KShortProduct> p = matchKShortProduct(contract, shapes))
        products.push_back(*p);
    });

    // Drop products until every value they widen is read only by kept
    // products, all reading it the same way.
    for (bool changed = true; changed;) {
      DenseMap<Value, std::optional<Claim>> claims;
      llvm::SmallPtrSet<Operation *, 8> kept;
      for (const KShortProduct &p : products) {
        kept.insert(p.contract);
        for (unsigned i = 0; i < 2; ++i)
          for (Value v : getChainValues(p, i)) {
            auto [it, inserted] = claims.try_emplace(v, getClaim(p, i));
            if (!inserted && it->second != getClaim(p, i))
              it->second = std::nullopt; // Read two ways.
          }
      }
      auto isClean = [&](const KShortProduct &p) {
        for (unsigned i = 0; i < 2; ++i) {
          Claim claim = getClaim(p, i);
          for (Value v : getChainValues(p, i)) {
            if (claims.lookup(v) != claim)
              return false;
            for (OpOperand &use : v.getUses()) {
              Operation *user = use.getOwner();
              if (auto contract = dyn_cast<vector::ContractionOp>(user)) {
                if (!kept.contains(contract) || use.getOperandNumber() != i)
                  return false;
              } else {
                Value forwarded = ::buddy::ime::getForwardedValue(use);
                if (!forwarded || claims.lookup(forwarded) != claim)
                  return false;
              }
            }
          }
        }
        return true;
      };
      size_t before = products.size();
      llvm::erase_if(products,
                     [&](const KShortProduct &p) { return !isClean(p); });
      changed = products.size() != before;
    }

    // Widen each read once, retype every path value in place, then rebuild
    // each product over the full K.
    llvm::SetVector<Value> pending;
    llvm::SmallPtrSet<Value, 16> ready;
    llvm::SmallPtrSet<Operation *, 8> widenedReads;
    OpBuilder b(&getContext());
    for (const KShortProduct &p : products)
      for (unsigned i = 0; i < 2; ++i) {
        const KShortOperand &operand = p.operands[i];
        pending.insert(operand.trace.chain.begin(), operand.trace.chain.end());
        Operation *read = operand.trace.load.op;
        if (!widenedReads.insert(read).second)
          continue;
        Value w = widen(b, operand, i, p.shape);
        read->getResult(0).replaceAllUsesWith(w);
        read->erase();
        ready.insert(w);
      }
    retype(b, pending, ready);
    for (KShortProduct &p : products) {
      b.setInsertionPoint(p.contract);
      auto res = vector::ContractionOp::create(
          b, p.contract.getLoc(), p.contract.getLhs(), p.contract.getRhs(),
          p.contract.getAcc(), p.contract.getIndexingMaps(),
          p.contract.getIteratorTypes(), p.contract.getKind());
      p.contract.replaceAllUsesWith(res.getResult());
      p.contract.erase();
    }
  }

private:
  /// What operand `index`'s read becomes at full K: a masked (rows, K) read
  /// of the same address, transposed for a direct (k, N) B.
  Value widen(OpBuilder &b, const KShortOperand &operand, unsigned index,
              const ::buddy::ime::TileShape &shape) {
    auto read = cast<vector::TransferReadOp>(operand.trace.load.op);
    b.setInsertionPoint(read);
    Location loc = read.getLoc();
    MemRefType type = operand.trace.load.type;
    int64_t rows = type.getDimSize(operand.trace.rowDim);
    int64_t k = type.getDimSize(operand.trace.kDim);
    Type element = type.getElementType();

    auto meta =
        memref::ExtractStridedMetadataOp::create(b, loc, read.getBase());
    auto viewTy = MemRefType::get({rows, shape.k}, element,
                                  StridedLayoutAttr::get(b.getContext(),
                                                         ShapedType::kDynamic,
                                                         {shape.k, 1}));
    Value view = memref::ReinterpretCastOp::create(
        b, loc, viewTy, meta.getBaseBuffer(),
        getAsOpFoldResult(meta.getOffset()),
        {b.getIndexAttr(rows), b.getIndexAttr(shape.k)},
        {b.getIndexAttr(shape.k), b.getIndexAttr(1)});
    Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
    Value mask = vector::CreateMaskOp::create(
        b, loc, VectorType::get({rows, shape.k}, b.getI1Type()),
        ValueRange{arith::ConstantIndexOp::create(b, loc, rows),
                   arith::ConstantIndexOp::create(b, loc, k)});
    Value padding = arith::ConstantOp::create(
        b, loc, cast<TypedAttr>(b.getZeroAttr(element)));
    auto wideRead = vector::TransferReadOp::create(
        b, loc, VectorType::get({rows, shape.k}, element), view,
        ValueRange{c0, c0}, padding, ArrayRef<bool>{true, true});
    wideRead.getMaskMutable().assign(mask);
    Value v = wideRead.getResult();
    // A (k, N) read is now its (N, K) storage; the path expects (K, N).
    if (index == 1 && !operand.trace.transposed)
      v = vector::TransposeOp::create(b, loc, v, ArrayRef<int64_t>{1, 0});
    return v;
  }

  /// Retypes the path values `pending` once their input is `ready`: a cast
  /// or transpose is rebuilt on its widened input, a barrier result and an
  /// iter_arg (with its unused loop result) take their input's type.
  void retype(OpBuilder &b, llvm::SetVector<Value> &pending,
              llvm::SmallPtrSet<Value, 16> &ready) {
    while (!pending.empty()) {
      Value next, input;
      for (Value v : pending) {
        Value in = ::buddy::ime::getTransparentInput(v);
        if (!in)
          if (auto transpose = v.getDefiningOp<vector::TransposeOp>())
            in = transpose.getVector();
        assert(in && "a path value has no input");
        if (ready.contains(in)) {
          next = v;
          input = in;
          break;
        }
      }
      if (!next)
        llvm_unreachable("a path value's input is never widened");
      pending.remove(next);
      auto inputTy = cast<VectorType>(input.getType());
      if (auto arg = dyn_cast<BlockArgument>(next)) {
        auto forOp = cast<scf::ForOp>(arg.getOwner()->getParentOp());
        arg.setType(inputTy);
        forOp.getTiedLoopResult(arg).setType(inputTy);
        ready.insert(next);
        continue;
      }
      Operation *def = next.getDefiningOp();
      if (isa<::buddy::cpu::SchedBarrierOp>(def)) {
        next.setType(inputTy);
        ready.insert(next);
        continue;
      }
      // A cast keeps its element type; a transpose swaps the shape.
      b.setInsertionPoint(def);
      Value rebuilt;
      if (auto transpose = dyn_cast<vector::TransposeOp>(def)) {
        rebuilt = vector::TransposeOp::create(b, def->getLoc(), input,
                                              transpose.getPermutation());
      } else {
        Type type = VectorType::get(inputTy.getShape(),
                                    getElementTypeOrSelf(next.getType()));
        OperationState state(def->getLoc(), def->getName(), ValueRange{input},
                             TypeRange{type}, def->getAttrs());
        rebuilt = b.create(state)->getResult(0);
      }
      next.replaceAllUsesWith(rebuilt);
      def->erase();
      ready.insert(rebuilt);
    }
  }
};

} // namespace

namespace mlir {
namespace buddy {
void registerPromoteIMETileKPass() { PassRegistration<PromoteIMETileKPass>(); }
} // namespace buddy
} // namespace mlir
