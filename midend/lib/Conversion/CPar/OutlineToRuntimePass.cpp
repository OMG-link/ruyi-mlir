//===-- OutlineToRuntimePass.cpp - Outline cpar.parallel regions ----------===//
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
// --cpar-outline-to-runtime: runs after the lowering to the LLVM dialect (all
// ops inside cpar.parallel regions are LLVM ops).  Every region
//
//   llvm.func @kernel(...) {
//     ...
//     cpar.parallel num_threads(N) {
//     ^bb0(%tid: i64, %nt: i64):
//       ... uses %a, %b defined above ...
//       cpar.yield
//     }
//   }
//
// becomes
//
//   llvm.func internal @kernel__cpar0(%ctx: !llvm.ptr, %tid: i64, %nt: i64) {
//     %a = llvm.load (gep %ctx[0, 0]) ; %b = llvm.load (gep %ctx[0, 1])
//     llvm.br ^body(%tid, %nt)
//   ^body(...): ... llvm.return
//   }
//   llvm.func @kernel(...) {
//     %ctx = llvm.alloca 1 x !llvm.struct<(typeof a, typeof b)>  // scope entry
//     ...
//     llvm.store %a, (gep %ctx[0, 0]) ; llvm.store %b, (gep %ctx[0, 1])
//     llvm.call @__cpar_parallel(@kernel__cpar0, %ctx, N)
//   }
//
// Captured values are passed by value through the context struct.  Memref
// descriptors are captured as descriptors, i.e. all threads address the same
// buffers.  Constants are cloned into the outlined function instead.  The
// context is allocated in the closest automatic allocation scope, so nested
// regions get one context per thread of the enclosing region.  Nested regions
// are outlined first.
//
//===----------------------------------------------------------------------===//

#include "CPar/CParDialect.h"
#include "CPar/CParTransforms.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using ::buddy::cpar::ParallelOp;
using ::buddy::cpar::YieldOp;

namespace {

// Ops that are cheaper to rematerialize than to pass through the context.
static bool isRematerializable(Operation *op) {
  if (op->getNumOperands() != 0 || op->getNumRegions() != 0)
    return false;
  return op->hasTrait<OpTrait::ConstantLike>() ||
         isa<LLVM::ZeroOp, LLVM::UndefOp, LLVM::PoisonOp>(op);
}

static LLVM::LLVMFuncOp getOrCreateRuntimeFunction(ModuleOp module,
                                                   OpBuilder &builder) {
  if (auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(
          ::buddy::cpar::kRuntimeFunctionName))
    return func;
  MLIRContext *ctx = module.getContext();
  auto ptr = LLVM::LLVMPointerType::get(ctx);
  auto type = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(ctx), {ptr, ptr, IntegerType::get(ctx, 64)});
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  return builder.create<LLVM::LLVMFuncOp>(
      module.getLoc(), ::buddy::cpar::kRuntimeFunctionName, type);
}

static std::string uniqueName(ModuleOp module, StringRef base) {
  for (unsigned i = 0;; ++i) {
    std::string name = (base + "__cpar" + Twine(i)).str();
    if (!SymbolTable::lookupSymbolIn(module, name))
      return name;
  }
}

static LogicalResult outline(ParallelOp op, ModuleOp module,
                             OpBuilder &builder) {
  auto parentFunc = op->getParentOfType<LLVM::LLVMFuncOp>();
  if (!parentFunc)
    return op.emitOpError("must be nested in an llvm.func; run "
                          "--cpar-outline-to-runtime after the lowering to "
                          "the LLVM dialect");
  Operation *scope =
      op->getParentWithTrait<OpTrait::AutomaticAllocationScope>();
  if (!scope || scope->getNumRegions() == 0 || scope->getRegion(0).empty())
    return op.emitOpError("has no enclosing automatic allocation scope");

  MLIRContext *ctx = op.getContext();
  Location loc = op.getLoc();
  Region &body = op.getBody();

  // Values used inside the region but defined above it.
  llvm::SetVector<Value> used;
  getUsedValuesDefinedAbove(body, used);
  SmallVector<Value> captures;
  SmallVector<Operation *> rematerialized;
  for (Value value : used) {
    Operation *def = value.getDefiningOp();
    if (def && isRematerializable(def)) {
      rematerialized.push_back(def);
      continue;
    }
    if (!LLVM::isCompatibleType(value.getType()))
      return op.emitOpError("captures a value of non-LLVM type ")
             << value.getType()
             << "; run --cpar-outline-to-runtime after the lowering to the "
                "LLVM dialect";
    captures.push_back(value);
  }

  auto ptrType = LLVM::LLVMPointerType::get(ctx);
  auto i64 = IntegerType::get(ctx, 64);
  SmallVector<Type> fieldTypes;
  for (Value value : captures)
    fieldTypes.push_back(value.getType());
  auto ctxType = LLVM::LLVMStructType::getLiteral(ctx, fieldTypes);

  // Context: one alloca in the entry block of the enclosing scope, filled in
  // right before the call.
  Value context;
  if (captures.empty()) {
    builder.setInsertionPoint(op);
    context = builder.create<LLVM::ZeroOp>(loc, ptrType);
  } else {
    Block &scopeEntry = scope->getRegion(0).front();
    builder.setInsertionPointToStart(&scopeEntry);
    Value one = builder.create<LLVM::ConstantOp>(
        loc, i64, builder.getI64IntegerAttr(1));
    context = builder.create<LLVM::AllocaOp>(loc, ptrType, ctxType, one);
    builder.setInsertionPoint(op);
    for (auto [index, value] : llvm::enumerate(captures)) {
      Value field = builder.create<LLVM::GEPOp>(
          loc, ptrType, ctxType, context,
          ArrayRef<LLVM::GEPArg>{0, static_cast<int32_t>(index)});
      builder.create<LLVM::StoreOp>(loc, value, field);
    }
  }

  // Outlined function: (ctx, tid, nthreads) -> void.
  builder.setInsertionPoint(parentFunc);
  auto fnType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                            {ptrType, i64, i64});
  auto outlined = builder.create<LLVM::LLVMFuncOp>(
      loc, uniqueName(module, parentFunc.getSymName()), fnType,
      LLVM::Linkage::Internal);
  Region &fnBody = outlined.getBody();
  fnBody.takeBody(body);
  Block *regionEntry = &fnBody.front();
  Block *entry = new Block();
  fnBody.push_front(entry);
  Value ctxArg = entry->addArgument(ptrType, loc);
  Value tidArg = entry->addArgument(i64, loc);
  Value ntArg = entry->addArgument(i64, loc);

  builder.setInsertionPointToStart(entry);
  IRMapping mapping;
  for (Operation *def : rematerialized) {
    Operation *clone = builder.clone(*def);
    for (auto [from, to] : llvm::zip(def->getResults(), clone->getResults()))
      mapping.map(from, to);
  }
  for (auto [index, value] : llvm::enumerate(captures)) {
    Value field = builder.create<LLVM::GEPOp>(
        loc, ptrType, ctxType, ctxArg,
        ArrayRef<LLVM::GEPArg>{0, static_cast<int32_t>(index)});
    mapping.map(value, builder.create<LLVM::LoadOp>(loc, value.getType(),
                                                    field));
  }
  builder.create<LLVM::BrOp>(loc, ValueRange{tidArg, ntArg}, regionEntry);

  for (auto [from, to] : mapping.getValueMap())
    replaceAllUsesInRegionWith(from, to, fnBody);
  SmallVector<YieldOp> yields;
  fnBody.walk([&](YieldOp yield) { yields.push_back(yield); });
  for (YieldOp yield : yields) {
    builder.setInsertionPoint(yield);
    builder.create<LLVM::ReturnOp>(yield.getLoc(), ValueRange{});
    yield.erase();
  }

  // Call site.
  LLVM::LLVMFuncOp runtime = getOrCreateRuntimeFunction(module, builder);
  builder.setInsertionPoint(op);
  Value fn = builder.create<LLVM::AddressOfOp>(loc, outlined);
  Value nthreads = builder.create<LLVM::ConstantOp>(
      loc, i64, builder.getI64IntegerAttr(op.getNumThreads()));
  builder.create<LLVM::CallOp>(loc, runtime,
                               ValueRange{fn, context, nthreads});
  op.erase();
  return success();
}

struct CParOutlineToRuntimePass
    : public PassWrapper<CParOutlineToRuntimePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CParOutlineToRuntimePass)
  StringRef getArgument() const final { return "cpar-outline-to-runtime"; }
  StringRef getDescription() const final {
    return "Outline cpar.parallel regions (LLVM dialect) into functions "
           "executed through the __cpar_parallel runtime call";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    // Post order: nested regions are outlined before the enclosing ones.
    SmallVector<ParallelOp> regions;
    module.walk([&](ParallelOp op) { regions.push_back(op); });
    OpBuilder builder(&getContext());
    for (ParallelOp op : regions)
      if (failed(outline(op, module, builder)))
        return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace buddy {
void registerCParOutlineToRuntimePass() {
  PassRegistration<CParOutlineToRuntimePass>();
}
} // namespace buddy
} // namespace mlir
