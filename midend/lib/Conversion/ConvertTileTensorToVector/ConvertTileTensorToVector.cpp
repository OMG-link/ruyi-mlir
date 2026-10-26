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
// `--convert-tile-tensor-to-vector` rewrites tile-sized tensors into flat
// 1-D vectors so that they live in a vector register group instead of being
// bufferized into a heap buffer and copied element by element.
//
// A tensor<d0 x ... x T> becomes vector<N x T>, N = d0 * ... , in row-major
// order. Tensor values are grouped into connected components by union-find:
// a convertible op joins all the tensor values it reads (that matter) and
// produces, and `scf.for` joins, per iter_arg, the init, region argument,
// yield operand and result. A component converts only when every value in it
// is legal (static shape, a vector element type of the target, and at most
// one LMUL=8 register group) and no op touching it is a blocker. Converting a
// component never introduces a vector<->tensor conversion, so a component is
// converted entirely or not at all.
//
// With `ime-tile-dot`, on the SpacemiT K1 (XSMTVDot, VLEN = 256), a
// `linalg.matmul(A, linalg.transpose(B))` on 4x8 int8 tiles into a 4x4 int32
// tile is one vmadot: it becomes `ime.tile.vmadot(u)` on the flat vectors,
// the transpose is absorbed when only such dots read it, and a dot from a
// zero accumulator that is then added to a tile accumulates into that tile.
// The option is set by pipelines that run `-lower-ime`.
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
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Remarks.h"
#include "mlir/Pass/Pass.h"

#include "Dialect/IME/IMEDialect.h"
#include "Dialect/IME/IMEOps.h"
#include "Utils/TargetInfo.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

constexpr StringLiteral kCategory = "tile-to-vector";

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

/// `linalg.matmul(A, linalg.transpose(B))` on the register tiles of one K1
/// IME dot: A and B are 4x8 int8 (B stored as N x K), C is 4x4 int32.
struct TileDot {
  Value lhs, rhs, acc;
  bool isUnsigned;
};

/// True when the target executes one tile dot as a single `vmadot`: the K1's
/// XSMTVDot at VLEN = 256, where a 4x8 int8 tile is one register.
bool hasK1TileDot(const mlir::buddy::TargetInfo &target) {
  return target.getArch() == mlir::buddy::TargetInfo::Arch::RISCV &&
         target.hasExtension("xsmtvdot") &&
         target.getVectorRegisterBits() == 256;
}

bool isTileType(Value v, ArrayRef<int64_t> shape, unsigned width) {
  auto ty = dyn_cast<RankedTensorType>(v.getType());
  return ty && ty.getShape() == shape && ty.getElementType().isInteger(width);
}

std::optional<TileDot> matchTileDot(linalg::MatmulOp matmul) {
  if (!matmul.hasPureTensorSemantics() || matmul.hasUserDefinedMaps())
    return std::nullopt;
  Value lhs = matmul.getDpsInputs()[0];
  auto transpose =
      matmul.getDpsInputs()[1].getDefiningOp<linalg::TransposeOp>();
  Value acc = matmul.getDpsInits()[0];
  if (!transpose || !transpose.hasPureTensorSemantics() ||
      transpose.getPermutation() != ArrayRef<int64_t>{1, 0})
    return std::nullopt;
  Value rhs = transpose.getInput();
  if (!isTileType(lhs, {4, 8}, 8) || !isTileType(rhs, {4, 8}, 8) ||
      !isTileType(acc, {4, 4}, 32))
    return std::nullopt;
  return TileDot{lhs, rhs, acc,
                 matmul.getCast() == linalg::TypeFn::cast_unsigned};
}

/// A transpose consumed only as the B operand of tile dots needs no op of its
/// own: the dot reads B in N x K order.
bool isAbsorbedTranspose(linalg::TransposeOp transpose,
                         const mlir::buddy::TargetInfo &target) {
  if (!hasK1TileDot(target) || transpose->use_empty())
    return false;
  for (OpOperand &use : transpose->getResult(0).getUses()) {
    auto matmul = dyn_cast<linalg::MatmulOp>(use.getOwner());
    if (!matmul || use.getOperandNumber() != 1 || !matchTileDot(matmul))
      return false;
  }
  return true;
}

class ConvertTileTensorToVectorPass
    : public PassWrapper<ConvertTileTensorToVectorPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTileTensorToVectorPass)

  ConvertTileTensorToVectorPass() = default;
  ConvertTileTensorToVectorPass(const ConvertTileTensorToVectorPass &other)
      : PassWrapper(other) {}

  Option<bool> imeTileDot{
      *this, "ime-tile-dot",
      llvm::cl::desc("Rewrite K1-sized matmul(A, transpose(B)) into "
                     "ime.tile dots; needs -lower-ime later in the pipeline"),
      llvm::cl::init(false)};
  StringRef getArgument() const final {
    return "convert-tile-tensor-to-vector";
  }
  StringRef getDescription() const final {
    return "Rewrite register-sized tensor components into flat vectors";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect,
                    vector::VectorDialect, tensor::TensorDialect,
                    ::buddy::ime::IMEDialect>();
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

  VectorType flatType(Type tensorTy) {
    auto t = cast<RankedTensorType>(tensorTy);
    return VectorType::get({t.getNumElements()}, t.getElementType());
  }
  Value lookup(OpBuilder &b, Value v);
  void rewrite(func::FuncOp func);
  void foldDotAccumulation(func::FuncOp func);
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
  if (!target.supportsVectorElementType(ty.getElementType())) {
    reason = "element type has no vector form on the target";
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
      if (classifyMemref(stripCasts(load.getBuffer()), tensorTy) ==
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
      if (imeTileDot && hasK1TileDot(target)) {
        if (std::optional<TileDot> dot = matchTileDot(matmul)) {
          uniteAll({dot->lhs, dot->rhs, dot->acc, matmul.getResult(0)});
          return;
        }
      }
    } else if (auto transpose = dyn_cast<linalg::TransposeOp>(op)) {
      if (imeTileDot && isAbsorbedTranspose(transpose, target))
        return;
    } else if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
      std::string reason;
      if (isElementwiseGeneric(generic, reason)) {
        SmallVector<Value> vs(generic.getDpsInputs());
        for (OpOperand &out : generic.getDpsInitsMutable())
          if (!generic.getMatchingBlockArgument(&out).use_empty())
            vs.push_back(out.get());
        llvm::append_range(vs, generic.getResults());
        uniteAll(vs);
        // Intermediate body values become vectors too.
        int64_t n =
            cast<ShapedType>(generic.getResult(0).getType()).getNumElements();
        for (Operation &bodyOp : generic.getBody()->without_terminator())
          for (Type t : bodyOp.getResultTypes()) {
            if (t.isInteger(1))
              continue;
            std::string why;
            if (!isLegalType(RankedTensorType::get({n}, t), target, why))
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
    } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      for (auto [i, init] : llvm::enumerate(forOp.getInitArgs())) {
        if (!isTensor(init))
          continue;
        uniteAll({forOp.getRegionIterArgs()[i], forOp.getResult(i), init,
                  yield.getOperand(i)});
      }
      return;
    } else if (auto yield = dyn_cast<scf::YieldOp>(op)) {
      if (isa<scf::ForOp>(op->getParentOp()))
        return;
    } else if (op->hasTrait<OpTrait::Elementwise>() &&
               op->getNumRegions() == 0 && op->getNumResults() > 0 &&
               llvm::all_of(op->getResults(), isTensor)) {
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
  VectorType vecTy = flatType(v.getType());
  Location loc = def->getLoc();
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
  if (classifyMemref(memref, ty) == Access::Contiguous) {
    if (ty.getRank() != 1) {
      SmallVector<ReassociationIndices> reassoc(1);
      for (int64_t i = 0; i < ty.getRank(); ++i)
        reassoc[0].push_back(i);
      memref = memref::CollapseShapeOp::create(b, loc, memref, reassoc);
    }
    return vector::LoadOp::create(b, loc, vecTy, memref, ValueRange{c0});
  }
  // One load per innermost row.
  ArrayRef<int64_t> shape = ty.getShape();
  int64_t rowLen = shape.back();
  auto rowTy = VectorType::get({rowLen}, ty.getElementType());
  Value acc = arith::ConstantOp::create(b, loc, vecTy,
                                        cast<TypedAttr>(b.getZeroAttr(vecTy)));
  int64_t rows = vecTy.getNumElements() / rowLen;
  for (int64_t r = 0; r < rows; ++r) {
    SmallVector<Value> idx;
    int64_t rem = r;
    SmallVector<int64_t> coords(shape.size() - 1);
    for (int64_t d = shape.size() - 2; d >= 0; --d) {
      coords[d] = rem % shape[d];
      rem /= shape[d];
    }
    for (int64_t c : coords)
      idx.push_back(arith::ConstantIndexOp::create(b, loc, c));
    idx.push_back(c0);
    Value row = vector::LoadOp::create(b, loc, rowTy, memref, idx);
    acc = vector::InsertStridedSliceOp::create(
        b, loc, row, acc, ArrayRef<int64_t>{r * rowLen}, ArrayRef<int64_t>{1});
  }
  return acc;
}

void ConvertTileTensorToVectorPass::storeTile(OpBuilder &b, Location loc,
                                              Value vec, Value memref) {
  memref = stripCasts(memref);
  auto ty = cast<MemRefType>(memref.getType());
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  if (classifyMemref(memref, ty) == Access::Contiguous) {
    if (ty.getRank() != 1) {
      SmallVector<ReassociationIndices> reassoc(1);
      for (int64_t i = 0; i < ty.getRank(); ++i)
        reassoc[0].push_back(i);
      memref = memref::CollapseShapeOp::create(b, loc, memref, reassoc);
    }
    vector::StoreOp::create(b, loc, vec, memref, ValueRange{c0});
    return;
  }
  ArrayRef<int64_t> shape = ty.getShape();
  int64_t rowLen = shape.back();
  int64_t rows = cast<VectorType>(vec.getType()).getNumElements() / rowLen;
  for (int64_t r = 0; r < rows; ++r) {
    SmallVector<Value> idx;
    int64_t rem = r;
    SmallVector<int64_t> coords(shape.size() - 1);
    for (int64_t d = shape.size() - 2; d >= 0; --d) {
      coords[d] = rem % shape[d];
      rem /= shape[d];
    }
    for (int64_t c : coords)
      idx.push_back(arith::ConstantIndexOp::create(b, loc, c));
    idx.push_back(c0);
    Value row = vector::ExtractStridedSliceOp::create(
        b, loc, vec, ArrayRef<int64_t>{r * rowLen}, ArrayRef<int64_t>{rowLen},
        ArrayRef<int64_t>{1});
    vector::StoreOp::create(b, loc, row, memref, idx);
  }
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
    Value v = vector::BroadcastOp::create(b, loc, flatType(res.getType()),
                                          fill.getInputs()[0]);
    mapping.map(res, v);
    toErase.push_back(op);
    return;
  }
  if (auto load = dyn_cast<bufferization::ToTensorOp>(op)) {
    if (!isConverted(load.getResult()))
      return;
    mapping.map(load.getResult(),
                loadTile(b, loc, load.getBuffer(), flatType(load.getType())));
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
    // Only matched tile dots join a component, so this matches.
    TileDot dot = *matchTileDot(matmul);
    Value acc = lookup(b, dot.acc), lhs = lookup(b, dot.lhs),
          rhs = lookup(b, dot.rhs);
    Value res = dot.isUnsigned
                    ? ::buddy::ime::TileVmadotuOp::create(b, loc, acc, lhs, rhs)
                          .getResult()
                    : ::buddy::ime::TileVmadotOp::create(b, loc, acc, lhs, rhs)
                          .getResult();
    mapping.map(matmul.getResult(0), res);
    toErase.push_back(op);
    return;
  }
  if (auto transpose = dyn_cast<linalg::TransposeOp>(op)) {
    // An absorbed transpose dies with the dots that read it; it is listed
    // before them, so it is erased after them.
    auto users = transpose->getResult(0).getUsers();
    if (transpose.hasPureTensorSemantics() && !transpose->use_empty() &&
        llvm::all_of(users, [&](Operation *user) {
          auto matmul = dyn_cast<linalg::MatmulOp>(user);
          return matmul && isConverted(matmul.getResult(0));
        }))
      toErase.push_back(op);
    return;
  }
  if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
    if (generic.getNumResults() == 0 || !isConverted(generic.getResult(0)))
      return;
    int64_t n =
        cast<ShapedType>(generic.getResult(0).getType()).getNumElements();
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
      auto vecTy = VectorType::get({n}, v.getType());
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
        types.push_back(VectorType::get({n}, t));
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
    mapping.map(op->getResult(0), lookup(b, op->getOperand(0)));
    toErase.push_back(op);
    return;
  }
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    for (auto [i, init] : llvm::enumerate(forOp.getInitArgs())) {
      BlockArgument arg = forOp.getRegionIterArgs()[i];
      if (!isTensor(init) || !isConverted(arg))
        continue;
      Value vInit = lookup(b, init);
      forOp.getInitArgsMutable()[i].set(vInit);
      VectorType vecTy = flatType(arg.getType());
      arg.setType(vecTy);
      forOp.getResult(i).setType(vecTy);
      mapping.map(arg, arg);
      mapping.map(forOp.getResult(i), forOp.getResult(i));
    }
    return;
  }
  if (auto yield = dyn_cast<scf::YieldOp>(op)) {
    auto forOp = dyn_cast<scf::ForOp>(op->getParentOp());
    if (!forOp)
      return;
    for (auto [i, arg] : llvm::enumerate(forOp.getRegionIterArgs()))
      if (isa<VectorType>(arg.getType()) && isTensor(yield.getOperand(i)))
        yield->setOperand(i, lookup(b, yield.getOperand(i)));
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
      types.push_back(flatType(t));
    OperationState state(loc, op->getName(), operands, types, op->getAttrs());
    Operation *clone = b.create(state);
    mapping.map(op->getResults(), clone->getResults());
    toErase.push_back(op);
  }
}

/// True when `v` is an all-zero vector: a zero splat constant or a broadcast
/// of a zero scalar.
bool isZeroVector(Value v) {
  if (matchPattern(v, m_Zero()))
    return true;
  auto bc = v.getDefiningOp<vector::BroadcastOp>();
  return bc && matchPattern(bc.getSource(), m_Zero());
}

/// `x + dot(0, a, b)` is `dot(x, a, b)`: the dot accumulates into x directly,
/// as the instruction does, instead of into a zeroed register then added.
template <typename DotOp> void foldOne(DotOp dot) {
  if (!isZeroVector(dot.getAcc()) || !dot->hasOneUse())
    return;
  auto add = dyn_cast<arith::AddIOp>(*dot->user_begin());
  if (!add)
    return;
  Value other = add.getLhs() == dot.getResult() ? add.getRhs() : add.getLhs();
  OpBuilder b(add);
  auto fused =
      DotOp::create(b, dot.getLoc(), other, dot.getLhs(), dot.getRhs());
  add.replaceAllUsesWith(fused.getResult());
  add.erase();
  dot.erase();
}

void ConvertTileTensorToVectorPass::foldDotAccumulation(func::FuncOp func) {
  SmallVector<Operation *> dots;
  func.walk([&](Operation *op) {
    if (isa<::buddy::ime::TileVmadotOp, ::buddy::ime::TileVmadotuOp>(op))
      dots.push_back(op);
  });
  for (Operation *op : dots) {
    if (auto dot = dyn_cast<::buddy::ime::TileVmadotOp>(op))
      foldOne(dot);
    else
      foldOne(cast<::buddy::ime::TileVmadotuOp>(op));
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
  foldDotAccumulation(func);
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

  rewrite(func);
}

} // namespace

namespace mlir {
namespace buddy {
void registerConvertTileTensorToVectorPass() {
  PassRegistration<ConvertTileTensorToVectorPass>();
}
} // namespace buddy
} // namespace mlir
