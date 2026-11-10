//===-- CParDialect.cpp - CPU Parallel Region Dialect ---------------------===//
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

#include "CPar/CParDialect.h"
#include "CPar/CParOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace buddy::cpar;

#include "CPar/CParDialect.cpp.inc"

void CParDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "CPar/CPar.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ParallelOp
//===----------------------------------------------------------------------===//

void ParallelOp::build(
    OpBuilder &builder, OperationState &state, int64_t numThreads,
    llvm::function_ref<void(OpBuilder &, Location, Value, Value)> bodyBuilder) {
  state.addAttribute(getNumThreadsAttrName(state.name),
                     builder.getI64IntegerAttr(numThreads));
  // createBlock moves the insertion point into the new region; the op itself
  // must still be inserted where the caller's builder points.
  OpBuilder::InsertionGuard guard(builder);
  Region *body = state.addRegion();
  Type i64 = builder.getI64Type();
  Block *entry = builder.createBlock(body, body->end(), {i64, i64},
                                     {state.location, state.location});
  if (bodyBuilder)
    bodyBuilder(builder, state.location, entry->getArgument(0),
                entry->getArgument(1));
  builder.setInsertionPointToEnd(entry);
  YieldOp::create(builder, state.location);
}

LogicalResult ParallelOp::verify() {
  Region &body = getBody();
  if (body.empty())
    return emitOpError("expects a non-empty body");
  Block &entry = body.front();
  if (entry.getNumArguments() != 2 ||
      !entry.getArgument(0).getType().isInteger(64) ||
      !entry.getArgument(1).getType().isInteger(64))
    return emitOpError(
        "expects the entry block to have two i64 arguments (tid, nthreads)");
  for (Block &block : body) {
    if (block.empty() || !block.back().hasTrait<OpTrait::IsTerminator>())
      return emitOpError("expects every block of the body to end with a "
                         "terminator");
    Operation &terminator = block.back();
    if (terminator.getNumSuccessors() == 0 && !isa<YieldOp>(terminator))
      return terminator.emitOpError("leaves a cpar.parallel region; only "
                                    "cpar.yield may do so");
  }
  return success();
}

#define GET_OP_CLASSES
#include "CPar/CPar.cpp.inc"
