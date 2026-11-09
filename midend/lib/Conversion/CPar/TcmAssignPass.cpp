//===-- TcmAssignPass.cpp - Place cpu.alloc buffers in the TCM ------------===//
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
// --cpu-tcm-assign{limit=<bytes>} (memref level, after
// --cpar-forall-to-parallel, before --finalize-memref-to-llvm).
//
// triton-riscv lowers triton.language.extra.cpu.alloc to
//
//   %buf = memref.alloc() {alignment = A, cpu.tcm} : memref<BYTESxi8>
//
// This pass gives every such buffer a static offset in the TCM of the thread
// that executes the allocation (buffers are laid out one after the other in
// program order, each aligned to its alignment) and replaces it by
//
//   %base = llvm.call @triton_cpu_tcm_base(%total) : (i64) -> !llvm.ptr
//   %ptr  = llvm.getelementptr %base[OFFSET] : (!llvm.ptr) -> !llvm.ptr, i8
//   %buf  = builtin.unrealized_conversion_cast <descriptor of %ptr>
//             : !llvm.struct<...> to memref<BYTESxi8>
//
// where %total is the size of all buffers of the module, so that the runtime
// can check that they fit.  The call is made where the allocation was, by the
// thread executing it: an allocation inside a cpar.parallel region gets the
// TCM of each thread of the region, one outside of it the TCM of the thread
// entering the region.  The call has side effects, so no later pass hoists it
// out of a region.
//
// A buffer allocated outside of a cpar.parallel region and used inside it is
// shared by the threads of the region; its base is obtained with
// @triton_cpu_tcm_base_shared instead, so that the runtime can check that the
// threads of the region are close to that TCM.
//
// The descriptor cast is folded by --finalize-memref-to-llvm and
// --reconcile-unrealized-casts.  The total size of the buffers of the module
// must not exceed `limit`.
//
//===----------------------------------------------------------------------===//

#include "CPar/CParDialect.h"
#include "CPar/CParOps.h"
#include "CPar/CParTransforms.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/MathExtras.h"

#include <limits>

using namespace mlir;
using ::buddy::cpar::ParallelOp;

namespace {

constexpr llvm::StringLiteral kTcmAttr = "cpu.tcm";
constexpr llvm::StringLiteral kBaseFn = "triton_cpu_tcm_base";
constexpr llvm::StringLiteral kSharedBaseFn = "triton_cpu_tcm_base_shared";

// True if a value derived from `alloc` (through ops returning memrefs, e.g.
// views) is used inside a cpar.parallel region that does not contain `alloc`.
bool usedInNestedRegion(memref::AllocOp alloc) {
  SmallVector<Value> work{alloc.getResult()};
  llvm::SmallPtrSet<Value, 8> seen;
  while (!work.empty()) {
    Value value = work.pop_back_val();
    if (!seen.insert(value).second)
      continue;
    for (Operation *user : value.getUsers()) {
      if (auto region = user->getParentOfType<ParallelOp>())
        if (!region->isProperAncestor(alloc))
          return true;
      for (Value result : user->getResults())
        if (isa<BaseMemRefType>(result.getType()))
          work.push_back(result);
    }
  }
  return false;
}

LLVM::LLVMFuncOp getOrCreateBaseFn(ModuleOp module, StringRef name) {
  if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return fn;
  MLIRContext *ctx = module.getContext();
  OpBuilder b(ctx);
  b.setInsertionPointToStart(module.getBody());
  auto type = LLVM::LLVMFunctionType::get(LLVM::LLVMPointerType::get(ctx),
                                          {IntegerType::get(ctx, 64)});
  return b.create<LLVM::LLVMFuncOp>(module.getLoc(), name, type);
}

struct CpuTcmAssignPass
    : public PassWrapper<CpuTcmAssignPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CpuTcmAssignPass)
  StringRef getArgument() const final { return "cpu-tcm-assign"; }
  StringRef getDescription() const final {
    return "Place memref.alloc ops marked 'cpu.tcm' at static offsets of the "
           "TCM of the executing thread (triton_cpu_tcm_base())";
  }
  CpuTcmAssignPass() = default;
  CpuTcmAssignPass(const CpuTcmAssignPass &other) : PassWrapper(other) {}

  Option<int64_t> limit{*this, "limit",
                        llvm::cl::desc("TCM capacity of one thread in bytes"),
                        llvm::cl::init(393216)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<memref::AllocOp> allocs;
    module.walk([&](memref::AllocOp alloc) {
      if (alloc->hasAttr(kTcmAttr))
        allocs.push_back(alloc);
    });
    if (allocs.empty())
      return;

    // Layout in program order.
    SmallVector<int64_t> offsets;
    int64_t total = 0;
    for (memref::AllocOp alloc : allocs) {
      MemRefType type = alloc.getType();
      if (type.getRank() != 1 || !type.hasStaticShape() ||
          !type.getElementType().isInteger(8) ||
          !type.getLayout().isIdentity() || type.getMemorySpace()) {
        alloc.emitError("cpu-tcm-assign: expected a static 1-d i8 buffer, got ")
            << type;
        return signalPassFailure();
      }
      const uint64_t align = alloc.getAlignment().value_or(1);
      const int64_t offset =
          static_cast<int64_t>(llvm::alignTo(static_cast<uint64_t>(total), align));
      offsets.push_back(offset);
      total = offset + type.getNumElements();
    }
    if (total > limit || total > std::numeric_limits<int32_t>::max()) {
      InFlightDiagnostic diag =
          module.emitError("cpu.alloc: the buffers of this kernel need ")
          << total << " bytes of TCM, more than the " << limit.getValue()
          << " bytes available to a thread";
      for (auto [alloc, offset] : llvm::zip(allocs, offsets))
        diag.attachNote(alloc.getLoc())
            << alloc.getType().getNumElements() << " bytes at offset "
            << offset;
      return signalPassFailure();
    }

    MLIRContext *ctx = &getContext();
    LLVMTypeConverter typeConverter(ctx);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);
    for (auto [alloc, offset] : llvm::zip(allocs, offsets)) {
      LLVM::LLVMFuncOp fn = getOrCreateBaseFn(
          module, usedInNestedRegion(alloc) ? kSharedBaseFn : kBaseFn);
      OpBuilder b(alloc);
      Location loc = alloc.getLoc();
      Value bytes = b.create<LLVM::ConstantOp>(loc, b.getI64Type(),
                                               b.getI64IntegerAttr(total));
      Value ptr = b.create<LLVM::CallOp>(loc, fn, ValueRange{bytes}).getResult();
      if (offset != 0)
        // offset <= limit, a TCM block (a few 100 KB) fits a GEP constant.
        ptr = b.create<LLVM::GEPOp>(
            loc, ptrType, b.getI8Type(), ptr,
            ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(offset)});
      MemRefType type = alloc.getType();
      Value descriptor = MemRefDescriptor::fromStaticShape(b, loc, typeConverter,
                                                           type, ptr);
      Value buffer =
          b.create<UnrealizedConversionCastOp>(loc, type, descriptor)
              .getResult(0);
      alloc.replaceAllUsesWith(buffer);
      alloc.erase();
    }
  }
};

} // namespace

namespace mlir {
namespace buddy {
void registerCpuTcmAssignPass() { PassRegistration<CpuTcmAssignPass>(); }
} // namespace buddy
} // namespace mlir
