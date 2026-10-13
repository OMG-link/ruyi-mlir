//===- MemrefView.h - Shared memref view predicate ------------*- C++ -*-===//
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
// The one definition of a memref view op, its view source, the origin a
// memref value traces to, and the may-alias relation between two origins.
//
// A memref view is a memref-dialect op with a memref result implementing
// ViewLikeOpInterface (memref.reinterpret_cast, memref.cast, memref.subview,
// memref.expand_shape, memref.collapse_shape, ...). Every component that
// needs to know whether an op is a view calls isMemrefView, so their lists
// cannot drift.
//
// Header-only and free of any QGEMM dependency, so that context-free
// rewrites such as ForwardReadOnlyCopyPattern can use it.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_UTILS_MEMREFVIEW_H
#define BUDDY_UTILS_MEMREFVIEW_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace buddy {

/// A memref-dialect op with a memref result implementing
/// ViewLikeOpInterface.
inline bool isMemrefView(Operation *op) {
  if (!op || !isa<memref::MemRefDialect>(op->getDialect()) ||
      op->getNumResults() != 1 ||
      !isa<BaseMemRefType>(op->getResult(0).getType()))
    return false;
  return isa<ViewLikeOpInterface>(op);
}

/// The aliased source of a memref view.
inline Value getMemrefViewSource(Operation *op) {
  assert(isMemrefView(op) && "not a memref view");
  return cast<ViewLikeOpInterface>(op).getViewSource();
}

/// Follow memref view sources from `memref` until a value no memref view
/// produces.
inline Value traceMemrefOrigin(Value memref) {
  while (isMemrefView(memref.getDefiningOp()))
    memref = getMemrefViewSource(memref.getDefiningOp());
  return memref;
}

/// A classified Memref_Origin.
struct MemrefOrigin {
  enum Kind {
    /// An entry-block argument of the enclosing function.
    Argument,
    /// The result of a memref.alloc or memref.alloca.
    Local,
    /// The result of a memref.get_global of a `constant` global.
    ConstantGlobal,
    /// Anything else, including a non-constant global.
    Unknown
  };
  Kind kind;
  /// The origin value itself.
  Value value;
  /// The global's symbol, for ConstantGlobal.
  StringAttr global;
};

/// Trace `memref` to its origin and classify it relative to `func`.
inline MemrefOrigin classifyMemrefOrigin(Value memref,
                                         FunctionOpInterface func) {
  Value origin = traceMemrefOrigin(memref);
  if (auto argument = dyn_cast<BlockArgument>(origin)) {
    if (func && argument.getOwner() == &func.getFunctionBody().front())
      return {MemrefOrigin::Argument, origin, nullptr};
    return {MemrefOrigin::Unknown, origin, nullptr};
  }
  Operation *producer = origin.getDefiningOp();
  if (isa<memref::AllocOp, memref::AllocaOp>(producer))
    return {MemrefOrigin::Local, origin, nullptr};
  if (auto getGlobal = dyn_cast<memref::GetGlobalOp>(producer)) {
    auto global = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
        getGlobal, getGlobal.getNameAttr());
    if (global && global.getConstant())
      return {MemrefOrigin::ConstantGlobal, origin,
              getGlobal.getNameAttr().getAttr()};
  }
  return {MemrefOrigin::Unknown, origin, nullptr};
}

/// Whether the origin is one Load_Copy_Forwarding supports.
inline bool isSupportedMemrefOrigin(const MemrefOrigin &origin) {
  return origin.kind != MemrefOrigin::Unknown;
}

/// May_Alias: two function arguments may alias unless one carries
/// `llvm.noalias`; a local alloc aliases only itself; a constant global
/// aliases only itself; origins of different supported kinds never alias;
/// an unknown origin may alias anything.
inline bool memrefOriginsMayAlias(const MemrefOrigin &a, const MemrefOrigin &b,
                                  FunctionOpInterface func) {
  if (a.kind == MemrefOrigin::Unknown || b.kind == MemrefOrigin::Unknown)
    return true;
  if (a.kind != b.kind)
    return false;
  switch (a.kind) {
  case MemrefOrigin::Argument: {
    if (a.value == b.value)
      return true;
    auto hasNoAlias = [&](Value value) {
      return static_cast<bool>(func.getArgAttr(
          cast<BlockArgument>(value).getArgNumber(), "llvm.noalias"));
    };
    return !hasNoAlias(a.value) && !hasNoAlias(b.value);
  }
  case MemrefOrigin::Local:
    return a.value == b.value;
  case MemrefOrigin::ConstantGlobal:
    return a.global == b.global;
  case MemrefOrigin::Unknown:
    return true;
  }
  llvm_unreachable("unhandled memref origin kind");
}

} // namespace buddy
} // namespace mlir

#endif // BUDDY_UTILS_MEMREFVIEW_H
