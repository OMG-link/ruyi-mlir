//===- LoadChain.cpp - QGEMM operand load chain walk -----------*- C++ -*-===//
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
// This file implements the shared load-chain walk.  It was extracted from
// QGEMMDimTransform, whose coarse scope was its first consumer.
//
//===----------------------------------------------------------------------===//

#include "LoadChain.h"

#include "Boundary.h"
#include "Utils/MemrefView.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

namespace mlir {
namespace buddy {

void collectQGEMMLoadChain(Value value, Operation *producer,
                           llvm::function_ref<bool(Operation *)> isAdmitted,
                           llvm::function_ref<void(Operation *)> record,
                           QGEMMLoadChainState &state) {
  if (!value || !state.visitedValues.insert(value).second)
    return;

  if (!producer)
    producer = value.getDefiningOp();
  if (!producer || !isAdmitted(producer))
    return;
  if (!state.visitedOperations.insert(producer).second)
    return;

  // A boundary which is already admitted by the shared walk is not a source
  // chain to recover. This keeps the source tracing tied to Boundary's
  // classification rather than introducing a second admission predicate.
  if (!isQGEMMOperatorBoundary(producer))
    return;

  if (isa<bufferization::ToTensorOp>(producer)) {
    record(producer);
    collectQGEMMLoadChain(producer->getOperand(0), nullptr, isAdmitted, record,
                          state);
    return;
  }

  if (isa<memref::AllocOp>(producer)) {
    record(producer);
    Value allocation = producer->getResult(0);
    for (Operation *user : allocation.getUsers()) {
      auto copy = dyn_cast<memref::CopyOp>(user);
      if (!copy || copy.getTarget() != allocation)
        continue;
      if (isAdmitted(copy.getOperation()))
        record(copy.getOperation());
      collectQGEMMLoadChain(copy.getSource(), nullptr, isAdmitted, record,
                            state);
    }
    return;
  }

  // Memref views (Utils/MemrefView.h) are followed through their view
  // source down to the origin, without admitting unrelated memref
  // computations.
  if (isMemrefView(producer)) {
    record(producer);
    collectQGEMMLoadChain(getMemrefViewSource(producer), nullptr, isAdmitted,
                          record, state);
  }
}

} // namespace buddy
} // namespace mlir
