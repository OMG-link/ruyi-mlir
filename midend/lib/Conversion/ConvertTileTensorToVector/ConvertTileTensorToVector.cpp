//===- ConvertTileTensorToVector.cpp - Keep register-sized tiles in vectors
//===//
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
// `--convert-tile-tensor-to-vector` rewrites tile-sized tensors into vectors
// of the same shape, so that they live in vector registers instead of being
// bufferized into a heap buffer and copied element by element. It only
// changes types: tensor<d0 x ... x T> becomes vector<d0 x ... x T>, and each
// op becomes the vector op with the same meaning. How a vector moves between
// memory and registers, and how an n-D vector maps onto 1-D registers, is
// left to the vector lowering (`--flatten-vector-tiles`, VectorToSCF,
// VectorToLLVM).
//
// Tensor values are grouped into connected components by union-find: a
// convertible op joins all the tensor values it reads (that matter) and
// produces. `scf.for` joins, per iter_arg, the init, region argument, yield
// operand and result; `scf.if` joins each result with the matching yield from
// both branches. A component converts only when every value in it is legal
// (static shape, a vector element type of the target, and at most one
// 8-register group) and no op touching it is a blocker. Converting a component
// never introduces a vector<->tensor conversion, so a component is converted
// entirely or not at all.
//
//   bufferization.to_tensor             vector.transfer_read
//   bufferization.materialize_in_dest.  vector.transfer_write
//   linalg.fill                         vector.broadcast
//   elementwise linalg.generic          its body on vectors
//   tensor.expand/collapse_shape        vector.shape_cast
//   scf.for iter_args                   vector iter_args
//   scf.if tensor results               vector results and yields
//   cpu.sched_barrier                   cpu.sched_barrier on vectors
//   cpu.shuffle on tensors              vector.shuffle
//
// A `cpu.shuffle` joins both operands and its result. One that stays in tensors
// lowers to a `linalg.generic` that reads each result element with
// `tensor.extract`, so a shuffle is correct whether or not its tiles fit in
// registers.
//
// A `cpu.sched_barrier` operand joins its result. The barrier exists to hold
// values in registers, so an operand whose component stays in tensors is an
// error.
//
// With `allow-ime-tile-contracts`, `linalg.matmul(A, linalg.transpose(B))` of
// an IME tile shape of the target (`Dialect/IME/TileShapes.h`: the K1's int8
// 4x8 tiles, the A100's int8 8x16 and fp16 8x8 tiles) also converts, to
// `vector.transpose` and a `vector.contract` on extended operands. A product
// whose K is below the tile's converts too when its operands are loads that
// a full-K tile can be read from under a mask (see `matchTileDotForm`); its
// `vector.contract` keeps the short K until `--promote-ime-tile-k` widens it.
// Other matmuls and transposes block, because only those have a
// register-level lowering (`--lower-vector-contract-to-ime-tile`) in the
// pipelines that set the option.
//
// Values that carry no data dependence — `tensor.empty`, dense
// `arith.constant`, the `outs` of `linalg.fill` and the unused `outs` of a
// `linalg.generic` — do not join components; a converted user materializes
// its own vector constant instead.
//
// The target comes from the module's `llvm.target` attribute (TargetInfo).
// Decisions are reported as optimization remarks in category
// `tile-to-vector`, shown only with `--remarks-filter*`.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Remarks.h"
#include "mlir/Pass/Pass.h"

#include "Dialect/Cpu/CpuDialect.h"
#include "Dialect/Cpu/CpuOps.h"
#include "Dialect/IME/TileOperands.h"
#include "Dialect/IME/TileShapes.h"
#include "Utils/TargetInfo.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

constexpr StringLiteral kCategory = "tile-to-vector";

std::string describeVectorOperation(Operation *op) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << op->getName() << " operands<";
  llvm::interleaveComma(op->getOperandTypes(), os,
                        [&](Type type) { os << type; });
  os << "> results<";
  llvm::interleaveComma(op->getResultTypes(), os,
                        [&](Type type) { os << type; });
  os << ">";
  return text;
}

bool isTensor(Value v) { return isa<TensorType>(v.getType()); }

/// Ops whose tensor result carries no data dependence on other tensors.
bool isSource(Operation *op) {
  if (isa<tensor::EmptyOp>(op))
    return true;
  if (auto cst = dyn_cast<arith::ConstantOp>(op))
    return isa<RankedTensorType>(cst.getType()) &&
           isa<DenseElementsAttr>(cst.getValue());
  return false;
}

bool isSourceValue(Value v) {
  Operation *def = v.getDefiningOp();
  return def && isSource(def);
}

/// The static memref a store target or load source really is, looking
/// through `memref.cast`s that only erase static information.
Value stripCasts(Value memref) {
  while (auto cast = memref.getDefiningOp<memref::CastOp>()) {
    auto srcTy = cast.getSource().getType();
    if (!isa<MemRefType>(srcTy))
      break;
    memref = cast.getSource();
  }
  return memref;
}

enum class Access { Contiguous, Rows, None };

/// How a tile can be moved between `memref` and a flat vector.
Access classifyMemref(Value memref, ShapedType tensorTy) {
  auto ty = dyn_cast<MemRefType>(memref.getType());
  if (!ty || !ty.hasStaticShape() || ty.getShape() != tensorTy.getShape() ||
      ty.getElementType() != tensorTy.getElementType())
    return Access::None;
  if (memref::isStaticShapeAndContiguousRowMajor(ty))
    return Access::Contiguous;
  if (ty.getRank() < 2)
    return Access::None;
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(ty.getStridesAndOffset(strides, offset)) || strides.back() != 1)
    return Access::None;
  return Access::Rows;
}

/// True when `generic` is an all-parallel, identity-map elementwise op whose
/// body ops all have a vector form. Sets `reason` otherwise.
bool isElementwiseGeneric(linalg::GenericOp generic, std::string &reason) {
  if (!generic.hasPureTensorSemantics()) {
    reason = "linalg.generic does not have pure tensor semantics";
    return false;
  }
  if (generic.getNumReductionLoops() != 0) {
    reason = "linalg.generic has a reduction loop";
    return false;
  }
  for (AffineMap map : generic.getIndexingMapsArray())
    if (!map.isIdentity()) {
      reason = "linalg.generic has a non-identity indexing map";
      return false;
    }
  for (Operation &op : generic.getBody()->without_terminator()) {
    if (op.getNumRegions() != 0 || !op.hasTrait<OpTrait::Vectorizable>()) {
      reason =
          ("body op '" + op.getName().getStringRef() + "' has no vector form")
              .str();
      return false;
    }
    for (Type t : op.getResultTypes())
      if (!t.isIntOrIndexOrFloat()) {
        reason = "body op produces a non-scalar value";
        return false;
      }
  }
  return true;
}

/// `linalg.matmul(A, linalg.transpose(B))` of an IME tile shape of the
/// target (B stored as N x K). The int8 form multiplies int8 tiles into an
/// int32 tile with the matmul's cast; the fp16 form is how Triton writes a
/// widening fp16 dot: both operands are extended to fp32 first, B after its
/// transpose.
///
/// A product whose K is below the tile's also matches when it can be read as
/// a full-K tile under a mask (`--promote-ime-tile-k` widens it later): each
/// operand traces (`Dialect/IME/TileOperands.h`) from the form's extension
/// and transpose to a `bufferization.to_tensor` load whose view
/// `canWidenAlongK` accepts. B may then also be a direct K x N load, read by
/// the form's extension itself (fp16) or by the product (int8), since the
/// widening transposes it right after the read. Every reader of those loads,
/// through transparent steps, must be such a product, at the same operand,
/// with the same tile K and B orientation, so widening the load changes no
/// other value.
struct TileDot {
  Value lhs, rhs, acc;
  bool isUnsigned;
  bool isFloat;
};

bool isTileType(Value v, ArrayRef<int64_t> shape, Type element) {
  auto ty = dyn_cast<RankedTensorType>(v.getType());
  return ty && ty.getShape() == shape && ty.getElementType() == element;
}

/// The f16 tensor an all-parallel `arith.extf` generic extends to f32.
Value getExtendedF16(Value v) {
  Value in = ::buddy::ime::getZeroPreservingCastInput(v);
  if (!in || !isa<RankedTensorType>(v.getType()))
    return {};
  auto generic = v.getDefiningOp<linalg::GenericOp>();
  if (!isa<arith::ExtFOp>(generic.getBody()->front()) ||
      !getElementTypeOrSelf(in).isF16() || !getElementTypeOrSelf(v).isF32())
    return {};
  return in;
}

/// The structure of a tile dot, without the reader check on a K-short one.
/// `kTile` receives the K of the matched tile shape.
std::optional<TileDot>
matchTileDotForm(linalg::MatmulOp matmul,
                 ArrayRef<::buddy::ime::TileShape> shapes, int64_t *kTile,
                 Value *loads, bool *bTransposed = nullptr) {
  if (!matmul.hasPureTensorSemantics() || matmul.hasUserDefinedMaps())
    return std::nullopt;
  MLIRContext *ctx = matmul.getContext();
  Value lhs = matmul.getDpsInputs()[0];
  Value rhs = matmul.getDpsInputs()[1];
  Value acc = matmul.getDpsInits()[0];
  auto accTy = dyn_cast<RankedTensorType>(acc.getType());
  auto lhsTy = dyn_cast<RankedTensorType>(lhs.getType());
  if (!accTy || !lhsTy || accTy.getRank() != 2 || lhsTy.getRank() != 2 ||
      !lhsTy.hasStaticShape())
    return std::nullopt;
  int64_t k = lhsTy.getDimSize(1);
  for (bool isFloat : {false, true}) {
    std::optional<::buddy::ime::TileShape> shape =
        ::buddy::ime::findTileShapeMN(shapes, accTy.getDimSize(0),
                                      accTy.getDimSize(1), isFloat);
    if (!shape || k > shape->k)
      continue;
    bool full = k == shape->k;
    Value a = lhs, bt = rhs;
    if (isFloat) {
      a = getExtendedF16(lhs);
      bt = getExtendedF16(rhs);
      if (!a || !bt)
        continue;
    }
    Type in =
        isFloat ? Type(Float16Type::get(ctx)) : Type(IntegerType::get(ctx, 8));
    Type out =
        isFloat ? Type(Float32Type::get(ctx)) : Type(IntegerType::get(ctx, 32));
    // B as stored: N x K under the form's transpose, or (K-short only) a
    // direct K x N load that the widening reads transposed.
    Value b;
    auto transpose = bt.getDefiningOp<linalg::TransposeOp>();
    if (transpose && transpose.hasPureTensorSemantics() &&
        transpose.getPermutation() == ArrayRef<int64_t>{1, 0}) {
      b = transpose.getInput();
      if (!isTileType(b, {shape->n, k}, in))
        continue;
    } else if (!full && isTileType(bt, {k, shape->n}, in)) {
      b = bt;
    } else {
      continue;
    }
    if (!isTileType(a, {shape->m, k}, in) ||
        !isTileType(acc, {shape->m, shape->n}, out))
      continue;
    if (!full) {
      // Both operands are loads of views a full-K tile can be read from.
      std::optional<::buddy::ime::TileOperand> ta =
          ::buddy::ime::traceTileOperand(lhs, 0);
      std::optional<::buddy::ime::TileOperand> tb =
          ::buddy::ime::traceTileOperand(rhs, 1);
      bool direct = b == bt;
      if (!ta || !tb || tb->transposed == direct ||
          (direct && tb->load.op->getResult(0) != b) ||
          !isa<bufferization::ToTensorOp>(ta->load.op) ||
          !isa<bufferization::ToTensorOp>(tb->load.op) ||
          !::buddy::ime::canWidenAlongK(*ta, shape->k) ||
          !::buddy::ime::canWidenAlongK(*tb, shape->k))
        continue;
      if (loads) {
        loads[0] = ta->load.op->getResult(0);
        loads[1] = tb->load.op->getResult(0);
      }
      if (bTransposed)
        *bTransposed = tb->transposed;
    }
    if (kTile)
      *kTile = shape->k;
    return TileDot{lhs, rhs, acc,
                   matmul.getCast() == linalg::TypeFn::cast_unsigned, isFloat};
  }
  return std::nullopt;
}

/// True when every reader of `load`, through transparent steps and B's
/// transpose, is operand `index` of a K-short tile dot of tile K `kTile` that
/// reads it as its load, with B transposed iff `bTransposed`.
bool readOnlyByShortTileDots(Value load, unsigned index, int64_t kTile,
                             bool bTransposed,
                             ArrayRef<::buddy::ime::TileShape> shapes) {
  SmallVector<Value> worklist{load};
  llvm::SmallPtrSet<Value, 8> seen;
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!seen.insert(v).second)
      continue;
    for (OpOperand &use : v.getUses()) {
      if (auto matmul = dyn_cast<linalg::MatmulOp>(use.getOwner())) {
        int64_t k = 0;
        Value loads[2];
        bool transposed = false;
        if (use.getOperandNumber() != index ||
            !matchTileDotForm(matmul, shapes, &k, loads, &transposed) ||
            k != kTile || loads[index] != load ||
            (index == 1 && transposed != bTransposed))
          return false;
        continue;
      }
      Value forwarded = ::buddy::ime::getForwardedValue(use);
      if (!forwarded)
        return false;
      worklist.push_back(forwarded);
    }
  }
  return true;
}

/// `shortLoads`, if given, receives the operand loads of a K-short match.
std::optional<TileDot>
matchTileDot(linalg::MatmulOp matmul, ArrayRef<::buddy::ime::TileShape> shapes,
             SmallVectorImpl<Value> *shortLoads = nullptr) {
  int64_t kTile = 0;
  Value loads[2];
  bool bTransposed = false;
  std::optional<TileDot> dot =
      matchTileDotForm(matmul, shapes, &kTile, loads, &bTransposed);
  if (!dot || !loads[0])
    return dot;
  for (unsigned i = 0; i < 2; ++i)
    if (!readOnlyByShortTileDots(loads[i], i, kTile, bTransposed, shapes))
      return std::nullopt;
  if (shortLoads)
    shortLoads->append({loads[0], loads[1]});
  return dot;
}

/// True when every reader of `v` ends in the B operand of tile products:
/// the matmul itself, or (fp16) the extension generic that feeds it.
bool feedsOnlyTileDotRhs(Value v, ArrayRef<::buddy::ime::TileShape> shapes) {
  if (v.use_empty())
    return false;
  for (OpOperand &use : v.getUses()) {
    Operation *user = use.getOwner();
    if (auto matmul = dyn_cast<linalg::MatmulOp>(user)) {
      if (use.getOperandNumber() != 1 || !matchTileDot(matmul, shapes))
        return false;
      continue;
    }
    auto generic = dyn_cast<linalg::GenericOp>(user);
    if (!generic || getExtendedF16(generic.getResult(0)) != v ||
        !feedsOnlyTileDotRhs(generic.getResult(0), shapes))
      return false;
  }
  return true;
}

/// A transpose read only (through an extension) as the B operand of tile
/// products converts with them; the IME lowering reads B in N x K order and
/// drops it.
bool isAbsorbedTranspose(linalg::TransposeOp transpose,
                         ArrayRef<::buddy::ime::TileShape> shapes) {
  return !shapes.empty() &&
         feedsOnlyTileDotRhs(transpose->getResult(0), shapes);
}

/// Lowers a `cpu.shuffle` on tensors to a `linalg.generic` that computes each
/// result element with `tensor.extract`: element (i, j...) reads mask[i] from
/// a constant, then row mask[i] of v1 or row mask[i] - v1.shape[0] of v2. Both
/// reads use an in-range index and a select picks one, so a poison index (-1)
/// reads row 0 of v1. This keeps a shuffle whose tiles stay in memory correct.
void lowerTensorShuffle(::buddy::cpu::ShuffleOp shuffle) {
  OpBuilder b(shuffle);
  Location loc = shuffle.getLoc();
  auto v1Ty = cast<RankedTensorType>(shuffle.getV1().getType());
  auto resTy = cast<RankedTensorType>(shuffle.getResult().getType());
  int64_t n1 = v1Ty.getDimSize(0);
  unsigned rank = resTy.getRank();
  ArrayRef<int64_t> mask = shuffle.getMask();
  auto maskTy = RankedTensorType::get({static_cast<int64_t>(mask.size())},
                                      b.getIndexType());
  SmallVector<APInt> maskValues;
  for (int64_t m : mask)
    maskValues.push_back(APInt(64, m, /*isSigned=*/true));
  Value maskCst = arith::ConstantOp::create(
      b, loc, DenseElementsAttr::get(maskTy, maskValues));
  Value init =
      tensor::EmptyOp::create(b, loc, resTy.getShape(), resTy.getElementType());
  SmallVector<AffineMap> maps{b.getMultiDimIdentityMap(rank)};
  SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
  auto generic = linalg::GenericOp::create(
      b, loc, TypeRange{resTy}, ValueRange{}, ValueRange{init}, maps, iters,
      [&](OpBuilder &nb, Location nloc, ValueRange) {
        SmallVector<Value> idx;
        for (unsigned d = 0; d < rank; ++d)
          idx.push_back(linalg::IndexOp::create(nb, nloc, d));
        Value m = tensor::ExtractOp::create(nb, nloc, maskCst, idx[0]);
        Value c0 = arith::ConstantIndexOp::create(nb, nloc, 0);
        Value cn1 = arith::ConstantIndexOp::create(nb, nloc, n1);
        Value nonNeg = arith::MaxSIOp::create(nb, nloc, m, c0);
        Value inV1 = arith::CmpIOp::create(nb, nloc, arith::CmpIPredicate::slt,
                                           nonNeg, cn1);
        Value row1 = arith::SelectOp::create(nb, nloc, inV1, nonNeg, c0);
        Value off = arith::SubIOp::create(nb, nloc, nonNeg, cn1);
        Value row2 = arith::SelectOp::create(nb, nloc, inV1, c0, off);
        SmallVector<Value> i1(idx), i2(idx);
        i1[0] = row1;
        i2[0] = row2;
        Value a = tensor::ExtractOp::create(nb, nloc, shuffle.getV1(), i1);
        Value c = tensor::ExtractOp::create(nb, nloc, shuffle.getV2(), i2);
        Value r = arith::SelectOp::create(nb, nloc, inV1, a, c);
        linalg::YieldOp::create(nb, nloc, r);
      });
  shuffle.getResult().replaceAllUsesWith(generic.getResult(0));
  shuffle.erase();
}

void lowerTensorShuffles(func::FuncOp func) {
  SmallVector<::buddy::cpu::ShuffleOp> shuffles;
  func.walk([&](::buddy::cpu::ShuffleOp op) {
    if (isTensor(op.getResult()))
      shuffles.push_back(op);
  });
  for (::buddy::cpu::ShuffleOp op : shuffles)
    lowerTensorShuffle(op);
}

class ConvertTileTensorToVectorPass
    : public PassWrapper<ConvertTileTensorToVectorPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTileTensorToVectorPass)

  ConvertTileTensorToVectorPass() = default;
  ConvertTileTensorToVectorPass(const ConvertTileTensorToVectorPass &other)
      : PassWrapper(other) {}

  Option<bool> allowIMETileContracts{
      *this, "allow-ime-tile-contracts",
      llvm::cl::desc("Convert matmul(A, transpose(B)) of the target's IME "
                     "tile shapes into vector.contract; needs "
                     "-lower-vector-contract-to-ime-tile later"),
      llvm::cl::init(false)};
  StringRef getArgument() const final {
    return "convert-tile-tensor-to-vector";
  }
  StringRef getDescription() const final {
    return "Rewrite register-sized tensor components into vectors";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, vector::VectorDialect,
                    tensor::TensorDialect, ::buddy::cpu::CpuDialect>();
  }

  void runOnOperation() override;

private:
  // ---- Union-find over tensor values -------------------------------------
  DenseMap<Value, Value> parent;
  /// Values in the order they joined, for deterministic remarks.
  SmallVector<Value> order;
  /// First blocking reason per component root.
  DenseMap<Value, std::pair<Location, std::string>> blocked;

  Value find(Value v) {
    auto it = parent.find(v);
    if (it == parent.end()) {
      parent[v] = v;
      order.push_back(v);
      return v;
    }
    if (it->second == v)
      return v;
    Value root = find(it->second);
    parent[v] = root;
    return root;
  }
  void block(Value v, Location loc, StringRef reason) {
    Value r = find(v);
    blocked.try_emplace(r, loc, reason.str());
  }
  void unite(Value a, Value b) {
    Value ra = find(a), rb = find(b);
    if (ra == rb)
      return;
    parent[rb] = ra;
    auto it = blocked.find(rb);
    if (it != blocked.end()) {
      blocked.try_emplace(ra, it->second);
      blocked.erase(rb);
    }
  }
  /// Joins `vs` into one component, skipping data-free sources.
  void uniteAll(ArrayRef<Value> vs) {
    Value first;
    for (Value v : vs) {
      if (isSourceValue(v))
        continue;
      if (!first) {
        first = v;
        find(v);
      } else {
        unite(first, v);
      }
    }
  }
  bool isConverted(Value v) {
    return parent.count(v) && !blocked.count(find(v));
  }

  // ---- Analysis ----------------------------------------------------------
  void analyze(func::FuncOp func, const mlir::buddy::TargetInfo &target);
  bool isLegalType(Type t, const mlir::buddy::TargetInfo &target,
                   std::string &reason);

  // ---- Rewrite -----------------------------------------------------------
  IRMapping mapping;
  SmallVector<Operation *> toErase;

  /// The vector a tensor value becomes: same shape, same element type.
  VectorType vectorType(Type tensorTy) {
    auto t = cast<RankedTensorType>(tensorTy);
    return VectorType::get(t.getShape(), t.getElementType());
  }
  Value lookup(OpBuilder &b, Value v);
  void rewrite(func::FuncOp func);
  void rewriteOp(OpBuilder &b, Operation *op);
  Value loadTile(OpBuilder &b, Location loc, Value memref, VectorType vecTy);
  void storeTile(OpBuilder &b, Location loc, Value vec, Value memref);
};

bool ConvertTileTensorToVectorPass::isLegalType(
    Type t, const mlir::buddy::TargetInfo &target, std::string &reason) {
  auto ty = dyn_cast<RankedTensorType>(t);
  if (!ty || !ty.hasStaticShape()) {
    reason = "tensor shape is not static";
    return false;
  }
  if (!target.canHoldVectorElementType(ty.getElementType())) {
    reason = "element type cannot be represented in vector registers";
    return false;
  }
  unsigned limit = target.getRegisterGroupElements(ty.getElementType());
  if (ty.getNumElements() > static_cast<int64_t>(limit)) {
    reason = ("tile has " + Twine(ty.getNumElements()) +
              " elements, more than one LMUL=8 register group (" +
              Twine(limit) + ")")
                 .str();
    return false;
  }
  return true;
}

void ConvertTileTensorToVectorPass::analyze(
    func::FuncOp func, const mlir::buddy::TargetInfo &target) {
  auto blockAll = [&](Operation *op, StringRef reason) {
    for (Value v : op->getOperands())
      if (isTensor(v) && !isSourceValue(v))
        block(v, op->getLoc(), reason);
    for (Value v : op->getResults())
      if (isTensor(v))
        block(v, op->getLoc(), reason);
  };

  // Loads of K-short tile dots are read whatever their strides: the K
  // widening replaces each by a masked full-K read before any lowering.
  llvm::SmallDenseSet<Value> shortTileLoads;
  if (allowIMETileContracts) {
    SmallVector<::buddy::ime::TileShape, 2> shapes =
        ::buddy::ime::getTileShapes(target);
    func.walk([&](linalg::MatmulOp matmul) {
      SmallVector<Value, 2> loads;
      if (matchTileDot(matmul, shapes, &loads))
        shortTileLoads.insert(loads.begin(), loads.end());
    });
  }

  func.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isSource(op))
      return;
    if (auto fill = dyn_cast<linalg::FillOp>(op)) {
      if (fill.hasPureTensorSemantics()) {
        uniteAll({fill.getResult(0)});
        return;
      }
    } else if (auto load = dyn_cast<bufferization::ToTensorOp>(op)) {
      auto tensorTy = cast<ShapedType>(load.getType());
      uniteAll({load.getResult()});
      if (!shortTileLoads.contains(load.getResult()) &&
          classifyMemref(stripCasts(load.getBuffer()), tensorTy) ==
              Access::None)
        block(load.getResult(), op->getLoc(),
              "load source is not row-major with unit innermost stride");
      return;
    } else if (auto store =
                   dyn_cast<bufferization::MaterializeInDestinationOp>(op)) {
      Value src = store.getSource();
      if (isSourceValue(src))
        return; // Nothing to unify; converted only if never blocked.
      uniteAll({src});
      if (!isa<MemRefType>(store.getDest().getType()) ||
          classifyMemref(stripCasts(store.getDest()),
                         cast<ShapedType>(src.getType())) == Access::None)
        block(src, op->getLoc(),
              "store target is not row-major with unit innermost stride");
      return;
    } else if (auto matmul = dyn_cast<linalg::MatmulOp>(op)) {
      if (allowIMETileContracts) {
        if (std::optional<TileDot> dot =
                matchTileDot(matmul, ::buddy::ime::getTileShapes(target))) {
          uniteAll({dot->lhs, dot->rhs, dot->acc, matmul.getResult(0)});
          return;
        }
      }
    } else if (auto transpose = dyn_cast<linalg::TransposeOp>(op)) {
      if (allowIMETileContracts &&
          isAbsorbedTranspose(transpose, ::buddy::ime::getTileShapes(target))) {
        uniteAll({transpose.getInput(), transpose.getResult()[0]});
        return;
      }
    } else if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
      std::string reason;
      if (isElementwiseGeneric(generic, reason)) {
        for (Operation &bodyOp : generic.getBody()->without_terminator()) {
          if (target.supportsVectorOperation(&bodyOp))
            continue;
          blockAll(op, (Twine("generic body operation ") +
                        describeVectorOperation(&bodyOp) +
                        " is unsupported for vector execution")
                           .str());
          return;
        }
        SmallVector<Value> vs(generic.getDpsInputs());
        for (OpOperand &out : generic.getDpsInitsMutable())
          if (!generic.getMatchingBlockArgument(&out).use_empty())
            vs.push_back(out.get());
        llvm::append_range(vs, generic.getResults());
        uniteAll(vs);
        // Intermediate body values become vectors too.
        ArrayRef<int64_t> shape =
            cast<ShapedType>(generic.getResult(0).getType()).getShape();
        for (Operation &bodyOp : generic.getBody()->without_terminator())
          for (Type t : bodyOp.getResultTypes()) {
            if (t.isInteger(1))
              continue;
            std::string why;
            if (!isLegalType(RankedTensorType::get(shape, t), target, why))
              block(generic.getResult(0), bodyOp.getLoc(),
                    "intermediate value: " + why);
          }
        return;
      }
      blockAll(op, reason);
      return;
    } else if (isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(op)) {
      uniteAll({op->getOperand(0), op->getResult(0)});
      return;
    } else if (auto shuffle = dyn_cast<::buddy::cpu::ShuffleOp>(op)) {
      if (isTensor(shuffle.getResult())) {
        uniteAll({shuffle.getV1(), shuffle.getV2(), shuffle.getResult()});
        return;
      }
    } else if (auto barrier = dyn_cast<::buddy::cpu::SchedBarrierOp>(op)) {
      for (auto [in, out] :
           llvm::zip(barrier.getInputs(), barrier.getResults()))
        if (isTensor(in))
          uniteAll({in, out});
      return;
    } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      for (auto [i, init] : llvm::enumerate(forOp.getInitArgs())) {
        if (!isTensor(init))
          continue;
        uniteAll({forOp.getRegionIterArgs()[i], forOp.getResult(i), init,
                  yield.getOperand(i)});
      }
      return;
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      // Each result is selected from the corresponding yield in either
      // branch. Join all of those values so the result can only convert when
      // both branches (and everything feeding them) are legal.
      for (auto [i, result] : llvm::enumerate(ifOp.getResults())) {
        if (!isTensor(result))
          continue;
        SmallVector<Value> values{result};
        for (Region &region : ifOp->getRegions()) {
          auto yield = cast<scf::YieldOp>(region.front().getTerminator());
          values.push_back(yield.getOperand(i));
        }
        uniteAll(values);
      }
      return;
    } else if (auto yield = dyn_cast<scf::YieldOp>(op)) {
      if (isa<scf::ForOp, scf::IfOp>(op->getParentOp()))
        return;
    } else if (op->hasTrait<OpTrait::Elementwise>() &&
               op->getNumRegions() == 0 && op->getNumResults() > 0 &&
               llvm::all_of(op->getResults(), isTensor)) {
      if (!target.supportsVectorOperation(op)) {
        blockAll(op, (Twine("vector operation ") + describeVectorOperation(op) +
                      " is unsupported for vector execution")
                         .str());
        return;
      }
      SmallVector<Value> vs;
      for (Value v : op->getOperands())
        if (isTensor(v))
          vs.push_back(v);
      llvm::append_range(vs, op->getResults());
      uniteAll(vs);
      return;
    }
    blockAll(op,
             ("'" + op->getName().getStringRef() + "' has no vector form here")
                 .str());
  });

  // Type legality of every value in a component.
  for (Value v : SmallVector<Value>(order)) {
    std::string reason;
    if (!isLegalType(v.getType(), target, reason)) {
      Location loc = v.getDefiningOp() ? v.getDefiningOp()->getLoc()
                                       : v.getParentRegion()->getLoc();
      block(v, loc, reason);
    }
  }
}

Value ConvertTileTensorToVectorPass::lookup(OpBuilder &b, Value v) {
  if (Value m = mapping.lookupOrNull(v))
    return m;
  Operation *def = v.getDefiningOp();
  assert(def && isSource(def) && "unconverted value inside a component");
  VectorType vecTy = vectorType(v.getType());
  Location loc = def->getLoc();
  // Same shape, so the dense attribute only changes its type.
  if (auto cst = dyn_cast<arith::ConstantOp>(def)) {
    auto attr = cast<DenseElementsAttr>(cst.getValue()).reshape(vecTy);
    return arith::ConstantOp::create(b, loc, vecTy, attr);
  }
  return arith::ConstantOp::create(b, loc, vecTy,
                                   cast<TypedAttr>(b.getZeroAttr(vecTy)));
}

Value ConvertTileTensorToVectorPass::loadTile(OpBuilder &b, Location loc,
                                              Value memref, VectorType vecTy) {
  memref = stripCasts(memref);
  auto ty = cast<MemRefType>(memref.getType());
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  SmallVector<Value> indices(ty.getRank(), c0);
  Value padding = arith::ConstantOp::create(
      b, loc, cast<TypedAttr>(b.getZeroAttr(ty.getElementType())));
  SmallVector<bool> inBounds(vecTy.getRank(), true);
  return vector::TransferReadOp::create(b, loc, vecTy, memref, indices, padding,
                                        inBounds);
}

void ConvertTileTensorToVectorPass::storeTile(OpBuilder &b, Location loc,
                                              Value vec, Value memref) {
  memref = stripCasts(memref);
  auto ty = cast<MemRefType>(memref.getType());
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  SmallVector<Value> indices(ty.getRank(), c0);
  SmallVector<bool> inBounds(cast<VectorType>(vec.getType()).getRank(), true);
  vector::TransferWriteOp::create(b, loc, vec, memref, indices, inBounds);
}

void ConvertTileTensorToVectorPass::rewriteOp(OpBuilder &b, Operation *op) {
  Location loc = op->getLoc();
  b.setInsertionPoint(op);

  if (auto fill = dyn_cast<linalg::FillOp>(op)) {
    if (!fill.hasPureTensorSemantics())
      return;
    Value res = fill.getResult(0);
    if (!isConverted(res))
      return;
    Value v = vector::BroadcastOp::create(b, loc, vectorType(res.getType()),
                                          fill.getInputs()[0]);
    mapping.map(res, v);
    toErase.push_back(op);
    return;
  }
  if (auto load = dyn_cast<bufferization::ToTensorOp>(op)) {
    if (!isConverted(load.getResult()))
      return;
    mapping.map(load.getResult(),
                loadTile(b, loc, load.getBuffer(), vectorType(load.getType())));
    toErase.push_back(op);
    return;
  }
  if (auto store = dyn_cast<bufferization::MaterializeInDestinationOp>(op)) {
    // The source may be an scf.for result already retyped to a vector, so
    // read it as a plain operand.
    Value src = op->getOperand(0);
    bool converted;
    if (isSourceValue(src)) {
      auto srcTy = cast<ShapedType>(src.getType());
      converted =
          isa<MemRefType>(store.getDest().getType()) &&
          srcTy.hasStaticShape() &&
          classifyMemref(stripCasts(store.getDest()), srcTy) != Access::None;
    } else {
      converted = isConverted(src);
    }
    if (!converted)
      return;
    storeTile(b, loc, lookup(b, src), store.getDest());
    toErase.push_back(op);
    return;
  }
  if (auto matmul = dyn_cast<linalg::MatmulOp>(op)) {
    if (!matmul.hasPureTensorSemantics() || !isConverted(matmul.getResult(0)))
      return;
    // C + sum_k ext(A[m,k]) * ext(B[k,n]): integer operands are extended
    // with linalg.matmul's cast; fp32 operands are already the extensions.
    Type i32 = b.getI32Type();
    bool isUnsigned = matmul.getCast() == linalg::TypeFn::cast_unsigned;
    auto extend = [&](Value v) -> Value {
      auto ty = cast<VectorType>(v.getType());
      if (!ty.getElementType().isInteger())
        return v;
      auto wide = VectorType::get(ty.getShape(), i32);
      return isUnsigned ? arith::ExtUIOp::create(b, loc, wide, v).getResult()
                        : arith::ExtSIOp::create(b, loc, wide, v).getResult();
    };
    Value lhs = extend(lookup(b, matmul.getDpsInputs()[0]));
    Value rhs = extend(lookup(b, matmul.getDpsInputs()[1]));
    Value acc = lookup(b, matmul.getDpsInits()[0]);
    MLIRContext *ctx = b.getContext();
    AffineExpr m, n, k;
    bindDims(ctx, m, n, k);
    auto maps = b.getAffineMapArrayAttr({AffineMap::get(3, 0, {m, k}, ctx),
                                         AffineMap::get(3, 0, {k, n}, ctx),
                                         AffineMap::get(3, 0, {m, n}, ctx)});
    auto par =
        vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel);
    auto red =
        vector::IteratorTypeAttr::get(ctx, vector::IteratorType::reduction);
    Value res = vector::ContractionOp::create(b, loc, lhs, rhs, acc, maps,
                                              b.getArrayAttr({par, par, red}));
    mapping.map(matmul.getResult(0), res);
    toErase.push_back(op);
    return;
  }
  if (auto transpose = dyn_cast<linalg::TransposeOp>(op)) {
    if (!transpose.hasPureTensorSemantics() ||
        !isConverted(transpose.getResult()[0]))
      return;
    Value res = vector::TransposeOp::create(
        b, loc, lookup(b, transpose.getInput()), transpose.getPermutation());
    mapping.map(transpose.getResult()[0], res);
    toErase.push_back(op);
    return;
  }
  if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
    if (generic.getNumResults() == 0 || !isConverted(generic.getResult(0)))
      return;
    ArrayRef<int64_t> shape =
        cast<ShapedType>(generic.getResult(0).getType()).getShape();
    IRMapping body;
    for (OpOperand &opnd : generic->getOpOperands()) {
      BlockArgument arg = generic.getMatchingBlockArgument(&opnd);
      if (!arg.use_empty())
        body.map(arg, lookup(b, opnd.get()));
    }
    // Scalars captured from outside the body are broadcast.
    auto mapOperand = [&](Value v) -> Value {
      if (Value m = body.lookupOrNull(v))
        return m;
      auto vecTy = VectorType::get(shape, v.getType());
      Value bc = vector::BroadcastOp::create(b, loc, vecTy, v);
      body.map(v, bc);
      return bc;
    };
    for (Operation &bodyOp : generic.getBody()->without_terminator()) {
      SmallVector<Value> operands;
      for (Value v : bodyOp.getOperands())
        operands.push_back(mapOperand(v));
      SmallVector<Type> types;
      for (Type t : bodyOp.getResultTypes())
        types.push_back(VectorType::get(shape, t));
      OperationState state(bodyOp.getLoc(), bodyOp.getName(), operands, types,
                           bodyOp.getAttrs());
      Operation *clone = b.create(state);
      body.map(bodyOp.getResults(), clone->getResults());
    }
    auto yield = cast<linalg::YieldOp>(generic.getBody()->getTerminator());
    for (auto [res, y] : llvm::zip(generic.getResults(), yield.getOperands()))
      mapping.map(res, mapOperand(y));
    toErase.push_back(op);
    return;
  }
  if (isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(op)) {
    if (!isConverted(op->getResult(0)))
      return;
    mapping.map(op->getResult(0),
                vector::ShapeCastOp::create(
                    b, loc, vectorType(op->getResult(0).getType()),
                    lookup(b, op->getOperand(0))));
    toErase.push_back(op);
    return;
  }
  if (auto shuffle = dyn_cast<::buddy::cpu::ShuffleOp>(op)) {
    if (!isTensor(shuffle.getResult()) || !isConverted(shuffle.getResult()))
      return;
    Value res = vector::ShuffleOp::create(b, loc, lookup(b, shuffle.getV1()),
                                          lookup(b, shuffle.getV2()),
                                          shuffle.getMask());
    mapping.map(shuffle.getResult(), res);
    toErase.push_back(op);
    return;
  }
  if (auto barrier = dyn_cast<::buddy::cpu::SchedBarrierOp>(op)) {
    SmallVector<Value> inputs;
    for (Value v : barrier.getInputs())
      inputs.push_back(isTensor(v) ? lookup(b, v) : v);
    auto vec = ::buddy::cpu::SchedBarrierOp::create(b, loc, inputs,
                                                    barrier.getMemoryAttr());
    for (auto [res, nres] : llvm::zip(barrier.getResults(), vec.getResults())) {
      if (isTensor(res))
        mapping.map(res, nres);
      else
        res.replaceAllUsesWith(nres);
    }
    toErase.push_back(op);
    return;
  }
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    for (auto [i, init] : llvm::enumerate(forOp.getInitArgs())) {
      BlockArgument arg = forOp.getRegionIterArgs()[i];
      if (!isTensor(arg) || !isConverted(arg))
        continue;
      // An earlier scf.if may already have retagged this init to a vector.
      Value vInit = isTensor(init) ? lookup(b, init) : init;
      forOp.getInitArgsMutable()[i].set(vInit);
      VectorType vecTy = vectorType(arg.getType());
      arg.setType(vecTy);
      forOp.getResult(i).setType(vecTy);
      mapping.map(arg, arg);
      mapping.map(forOp.getResult(i), forOp.getResult(i));
    }
    return;
  }
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    for (auto [i, result] : llvm::enumerate(ifOp.getResults())) {
      if (!isTensor(result) || !isConverted(result))
        continue;
      result.setType(vectorType(result.getType()));
      mapping.map(result, result);
    }
    return;
  }
  if (auto yield = dyn_cast<scf::YieldOp>(op)) {
    if (auto forOp = dyn_cast<scf::ForOp>(op->getParentOp())) {
      for (auto [i, arg] : llvm::enumerate(forOp.getRegionIterArgs()))
        if (isa<VectorType>(arg.getType()) && isTensor(yield.getOperand(i)))
          yield->setOperand(i, lookup(b, yield.getOperand(i)));
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op->getParentOp())) {
      for (auto [i, result] : llvm::enumerate(ifOp.getResults()))
        if (isa<VectorType>(result.getType()) && isTensor(yield.getOperand(i)))
          yield->setOperand(i, lookup(b, yield.getOperand(i)));
    }
    return;
  }
  if (op->hasTrait<OpTrait::Elementwise>() && op->getNumResults() > 0 &&
      llvm::all_of(op->getResults(), isTensor) &&
      isConverted(op->getResult(0))) {
    SmallVector<Value> operands;
    for (Value v : op->getOperands())
      operands.push_back(isTensor(v) ? lookup(b, v) : v);
    SmallVector<Type> types;
    for (Type t : op->getResultTypes())
      types.push_back(vectorType(t));
    OperationState state(loc, op->getName(), operands, types, op->getAttrs());
    Operation *clone = b.create(state);
    mapping.map(op->getResults(), clone->getResults());
    toErase.push_back(op);
  }
}

void ConvertTileTensorToVectorPass::rewrite(func::FuncOp func) {
  SmallVector<Operation *> ops;
  func.walk<WalkOrder::PreOrder>([&](Operation *op) {
    // Bodies of rewritten generics are cloned, not visited.
    if (isa<linalg::GenericOp>(op)) {
      ops.push_back(op);
      return WalkResult::skip();
    }
    ops.push_back(op);
    return WalkResult::advance();
  });
  OpBuilder b(func.getContext());
  for (Operation *op : ops)
    rewriteOp(b, op);

  // Erase users before producers. A remaining use is a data-free position
  // (fill or unused generic outs) of an unconverted op; give it an empty.
  for (Operation *op : llvm::reverse(toErase)) {
    for (Value res : op->getResults()) {
      for (OpOperand &use : llvm::make_early_inc_range(res.getUses())) {
        b.setInsertionPoint(use.getOwner());
        auto ty = cast<RankedTensorType>(res.getType());
        use.set(tensor::EmptyOp::create(b, op->getLoc(), ty.getShape(),
                                        ty.getElementType()));
      }
    }
    op->erase();
  }
  func.walk([](Operation *op) {
    if (isSource(op) && op->use_empty())
      op->erase();
  });
}

void ConvertTileTensorToVectorPass::runOnOperation() {
  func::FuncOp func = getOperation();
  FailureOr<mlir::buddy::TargetInfo> target = mlir::buddy::getTargetInfo(func);
  if (failed(target))
    return signalPassFailure();
  if (!target->hasVector()) {
    remark::missed(func.getLoc(),
                   remark::RemarkOpts::name("no-vector").category(kCategory))
        << remark::add("function '{0}'", func.getSymName())
        << remark::reason("the target has no vector unit");
    lowerTensorShuffles(func);
    return;
  }

  parent.clear();
  order.clear();
  blocked.clear();
  mapping.clear();
  toErase.clear();
  analyze(func, *target);

  // One remark per component, at the op defining its first value.
  llvm::MapVector<Value, unsigned> sizes;
  for (Value v : order)
    ++sizes[find(v)];
  for (auto [root, size] : sizes) {
    auto it = blocked.find(root);
    if (it != blocked.end()) {
      remark::missed(it->second.first,
                     remark::RemarkOpts::name("component").category(kCategory))
          << remark::add("tensor component of {0} values stays in memory", size)
          << remark::reason("{0}", it->second.second);
    } else {
      Location loc = root.getDefiningOp() ? root.getDefiningOp()->getLoc()
                                          : root.getParentRegion()->getLoc();
      remark::passed(loc,
                     remark::RemarkOpts::name("component").category(kCategory))
          << remark::add("tensor component of {0} values converted to vectors",
                         size);
    }
  }

  // A barrier whose tensor operand stays in memory cannot do its job.
  bool barrierBlocked = false;
  func.walk([&](::buddy::cpu::SchedBarrierOp barrier) {
    for (Value v : barrier.getInputs()) {
      if (!isTensor(v) || isConverted(v))
        continue;
      auto it = blocked.find(find(v));
      InFlightDiagnostic diag = barrier.emitError(
          "sched_barrier operand cannot be held in vector registers");
      if (it != blocked.end())
        diag << ": " << it->second.second;
      barrierBlocked = true;
    }
  });
  if (barrierBlocked)
    return signalPassFailure();

  rewrite(func);
  lowerTensorShuffles(func);
}

} // namespace

namespace mlir {
namespace buddy {
void registerConvertTileTensorToVectorPass() {
  PassRegistration<ConvertTileTensorToVectorPass>();
}
} // namespace buddy
} // namespace mlir
