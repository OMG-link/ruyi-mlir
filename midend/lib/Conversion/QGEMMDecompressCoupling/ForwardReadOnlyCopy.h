//===- ForwardReadOnlyCopy.h - Forward read-only memref copies -*- C++ -*-===//
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
// Load_Copy_Forwarding (Req 14.7, 14.8, 14.11, 14.12) as a context-free
// rewrite pattern. It rewrites
//
//   %v = memref.reinterpret_cast %arg ...
//   %a = memref.alloc()
//   memref.copy %v, %a
//   %t = bufferization.to_tensor %a restrict writable
//
// into
//
//   %v = memref.reinterpret_cast %arg ...
//   %t = bufferization.to_tensor %v restrict
//
// when the source memory provably cannot change for the whole function, so
// that reading it at the use is equivalent to reading the snapshot the copy
// took. The source may be any memref whose origin is a function argument, a
// local alloc/alloca or a constant global (see Utils/MemrefView.h); the
// view above is the common case. The pattern reads no QGEMM annotation; it only
// carries the original to_tensor's discardable attributes over to the
// replacement. The legality conditions are listed in ForwardReadOnlyCopy.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_QGEMM_DECOMPRESS_COUPLING_FORWARD_READ_ONLY_COPY_H
#define BUDDY_QGEMM_DECOMPRESS_COUPLING_FORWARD_READ_ONLY_COPY_H

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace buddy {

/// Rewrites to_tensor(alloc) filled by copy(source -> alloc) into
/// to_tensor(source) when no op of the enclosing function can modify the
/// memory the source reads.
struct ForwardReadOnlyCopyPattern
    : public OpRewritePattern<bufferization::ToTensorOp> {
  using OpRewritePattern<bufferization::ToTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(bufferization::ToTensorOp toTensor,
                                PatternRewriter &rewriter) const override;
};

/// Adds ForwardReadOnlyCopyPattern to `patterns`.
void populateForwardReadOnlyCopyPatterns(RewritePatternSet &patterns);

} // namespace buddy
} // namespace mlir

#endif // BUDDY_QGEMM_DECOMPRESS_COUPLING_FORWARD_READ_ONLY_COPY_H
