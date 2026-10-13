//===- ForwardReadOnlyCopy.cpp - Forward read-only memref copies ----------===//
//
// Licensed under the Apache License v2.0 (the "License");
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
// Shape matching and legality of ForwardReadOnlyCopyPattern.
//
// The copy takes a snapshot of the source at the copy point. After the
// rewrite the tensor reads the source itself, and after bufferization that
// read may happen at any later point of the function. The memcpy is
// removable when three conditions hold:
//
//  (1) The source is not modified anywhere in the function, before or after
//      the copy (S3 and W1-W5 below).
//  (2) The destination is read-only: nothing but the to_tensor reads or
//      writes the alloc (S1), and the replacement to_tensor is never
//      `writable`. One-Shot Bufferize treats a non-writable to_tensor buffer
//      as read-only, so any in-place write to the forwarded tensor -- a DPS
//      init, an scf iter_arg/yield, a function return -- gets a new buffer
//      instead, copied from the source when the old value is read. The
//      source is therefore never written through the forwarded tensor, and
//      no restriction on the forwarded tensor's uses is needed.
//  (3) The `restrict` claim holds (R below). One-Shot Bufferize accepts only
//      `restrict` to_tensor ops, and `restrict` asserts that the tensor is
//      the only tensor-IR access to its memref and every alias of it. The
//      forwarded to_tensor keeps `restrict`, so no other to_tensor or
//      materialize_in_destination may reach memory aliasing the source.
//
// The conditions checked are:
//
//  S1  The to_tensor reads a memref.alloc whose only users are the to_tensor
//      and exactly one memref.copy targeting it. Requiring the copy in the
//      same block before the to_tensor is a shape restriction, not a safety
//      condition.
//  S2  The copy source is any memref value with the alloc's shape and
//      element type. This is a shape restriction as well.
//  S3  The source traces through memref views (MemrefView.h) to a supported
//      origin `base`: an entry-block argument of the enclosing function, a
//      local memref.alloc/alloca, or a memref.get_global of a `constant`
//      global. Any other origin, including a non-constant global, rejects.
//  W   Every op of the function is scanned (nested regions included), and
//      each one must be provably unable to write or free memory that may
//      alias `base`:
//      W1  An op without MemoryEffectOpInterface and without the recursive
//          memory effects trait (func.call, unknown ops) rejects.
//      W2  A Write or Free effect not tied to a value rejects.
//      W3  A Write or Free effect on a value is traced to its origin. An
//          unknown origin (an unknown memref-producing op, a nested block
//          argument such as a loop-carried memref, a non-constant global)
//          rejects.
//      W4  A Write or Free effect on a value whose origin may alias `base`
//          rejects. Two arguments may alias unless one carries
//          `llvm.noalias`; a local alloc aliases only itself, so a write or
//          free tracing to the same alloc as a local `base` rejects; a
//          constant global aliases only itself; origins of different kinds
//          never alias.
//      W5  A `bufferization.to_tensor ... writable` whose buffer may alias
//          `base` rejects, because bufferization may write through it.
//      Ops with the recursive memory effects trait are not judged themselves;
//      their nested ops are, by the same scan.
//  R   Every other `bufferization.to_tensor` buffer and every memref
//      destination of a `bufferization.materialize_in_destination` in the
//      function is traced to its origin; an origin that may alias `base` by
//      the W4 rule, or an unknown origin, rejects. This is evaluated on the
//      current IR, so a to_tensor forwarded earlier and an upstream
//      `restrict` view (e.g. a lookup table) count as accesses.
//
// Residual gap: memory reachable from `base` by means other than SSA values
// (e.g. a pointer reconstructed from an integer, or a callee holding a
// global alias) is not visible here; W1 covers calls, but an op that claims
// no effects through its interface while writing memory would be trusted.
// Condition (2) relies on One-Shot Bufferize honouring the read-only status
// of a non-writable to_tensor; a bufferization that wrote through such a
// buffer in place would break it.
//
//===----------------------------------------------------------------------===//

#include "ForwardReadOnlyCopy.h"
#include "Utils/MemrefView.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace buddy {

namespace {

/// W3/W4: whether a write or free of `value` may modify `base`.
bool writeMayHitBase(Value value, const MemrefOrigin &base,
                     FunctionOpInterface func) {
  if (!isa<BaseMemRefType>(value.getType()))
    return true;
  return memrefOriginsMayAlias(classifyMemrefOrigin(value, func), base, func);
}

/// Condition W: true when some op of `func` may write or free memory that
/// may alias `base`.
bool functionMayModify(FunctionOpInterface func, const MemrefOrigin &base) {
  WalkResult result = func.getFunctionBody().walk([&](Operation *op) {
    // W5: a writable to_tensor on aliasing memory lets bufferization write.
    if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(op)) {
      if (toTensor.getWritable() &&
          writeMayHitBase(toTensor.getBuffer(), base, func))
        return WalkResult::interrupt();
      return WalkResult::advance();
    }
    // Nested ops are visited by this walk and judged on their own.
    if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
      return WalkResult::advance();
    auto effectsOp = dyn_cast<MemoryEffectOpInterface>(op);
    // W1: effects unknown.
    if (!effectsOp)
      return WalkResult::interrupt();
    SmallVector<MemoryEffects::EffectInstance> effects;
    effectsOp.getEffects(effects);
    for (const MemoryEffects::EffectInstance &effect : effects) {
      if (!isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect()))
        continue;
      Value target = effect.getValue();
      // W2: a write to an unnamed resource may hit anything.
      if (!target || writeMayHitBase(target, base, func))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

/// Condition R: true when some tensor-IR access of `func` other than
/// `self` -- a `bufferization.to_tensor` buffer or a memref destination of
/// `bufferization.materialize_in_destination` -- may alias `base`. Evaluated
/// on the current IR, so to_tensors forwarded earlier count.
bool otherTensorAccessMayAlias(FunctionOpInterface func,
                               const MemrefOrigin &base,
                               bufferization::ToTensorOp self) {
  auto mayHitBase = [&](Value memref) {
    return memrefOriginsMayAlias(classifyMemrefOrigin(memref, func), base,
                                 func);
  };
  WalkResult result = func.getFunctionBody().walk([&](Operation *op) {
    if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(op)) {
      if (toTensor != self && mayHitBase(toTensor.getBuffer()))
        return WalkResult::interrupt();
      return WalkResult::advance();
    }
    if (auto materialize =
            dyn_cast<bufferization::MaterializeInDestinationOp>(op)) {
      Value dest = materialize.getDest();
      // A tensor destination is not a memref access of tensor IR.
      if (isa<BaseMemRefType>(dest.getType()) && mayHitBase(dest))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

} // namespace

LogicalResult
ForwardReadOnlyCopyPattern::matchAndRewrite(bufferization::ToTensorOp toTensor,
                                            PatternRewriter &rewriter) const {
  // S1: to_tensor(alloc) with exactly one preceding copy into the alloc.
  auto alloc = toTensor.getBuffer().getDefiningOp<memref::AllocOp>();
  if (!alloc)
    return rewriter.notifyMatchFailure(toTensor,
                                       "buffer is not a memref.alloc");
  memref::CopyOp copy;
  for (Operation *user : alloc.getResult().getUsers()) {
    if (user == toTensor.getOperation())
      continue;
    auto candidate = dyn_cast<memref::CopyOp>(user);
    if (!candidate || candidate.getTarget() != alloc.getResult() || copy)
      return rewriter.notifyMatchFailure(toTensor,
                                         "alloc has users besides one copy");
    copy = candidate;
  }
  if (!copy || copy->getBlock() != toTensor->getBlock() ||
      !copy->isBeforeInBlock(toTensor))
    return rewriter.notifyMatchFailure(toTensor,
                                       "no copy into the alloc before use");

  // S2: the copy reads a memref of matching shape and element type.
  Value view = copy.getSource();
  auto sourceType = dyn_cast<MemRefType>(view.getType());
  if (!sourceType || sourceType.getShape() != alloc.getType().getShape() ||
      sourceType.getElementType() != alloc.getType().getElementType())
    return rewriter.notifyMatchFailure(toTensor, "copy source type mismatch");

  // S3: the source traces to a supported origin.
  auto func = toTensor->getParentOfType<FunctionOpInterface>();
  if (!func)
    return rewriter.notifyMatchFailure(toTensor, "not inside a function");
  MemrefOrigin origin = classifyMemrefOrigin(view, func);
  if (!isSupportedMemrefOrigin(origin))
    return rewriter.notifyMatchFailure(toTensor,
                                       "source origin is not supported");

  // W: nothing in the function may modify the source memory.
  if (functionMayModify(func, origin))
    return rewriter.notifyMatchFailure(toTensor,
                                       "source memory may be modified");

  // R: the forwarded `restrict` to_tensor must be the only tensor-IR access
  // to the source memory.
  if (otherTensorAccessMayAlias(func, origin, toTensor))
    return rewriter.notifyMatchFailure(
        toTensor, "another tensor access may alias the source");

  auto forwarded = rewriter.create<bufferization::ToTensorOp>(
      toTensor.getLoc(), toTensor.getResult().getType(), view,
      toTensor.getRestrict(), /*writable=*/false);
  forwarded->setDiscardableAttrs(toTensor->getDiscardableAttrDictionary());
  rewriter.replaceOp(toTensor, forwarded.getResult());
  rewriter.eraseOp(copy);
  rewriter.eraseOp(alloc);
  return success();
}

void populateForwardReadOnlyCopyPatterns(RewritePatternSet &patterns) {
  patterns.add<ForwardReadOnlyCopyPattern>(patterns.getContext());
}

} // namespace buddy
} // namespace mlir
