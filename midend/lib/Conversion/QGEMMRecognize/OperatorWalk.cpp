//===- OperatorWalk.cpp - QGEMM backwards boundary traversal ---*- C++ -*-===//
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
// This file implements the backwards traversal from a candidate QGEMM
// reduction to its upstream boundaries.
//
//===----------------------------------------------------------------------===//

#include "OperatorWalk.h"

#include "Boundary.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "qgemm-recognize"

using namespace mlir;

namespace mlir {
namespace buddy {
namespace {

/// Return true for tensor-like values that can carry operator data. Scalar and
/// index plumbing is intentionally not traversed.
static bool isTraceableValue(Value value) {
  return isa<TensorType, BaseMemRefType>(value.getType());
}

static QGEMMBoundaryKind classifyBoundary(Value value, Operation *producer,
                                          Block *rootBlock) {
  if (!producer)
    return QGEMMBoundaryKind::BlockArgument;
  if (producer->getBlock() != rootBlock)
    return QGEMMBoundaryKind::CrossBlockValue;
  if (isa<bufferization::ToTensorOp>(producer))
    return QGEMMBoundaryKind::BufferizationToTensor;
  if (isa<BaseMemRefType>(value.getType()))
    return QGEMMBoundaryKind::MemrefResult;
  return QGEMMBoundaryKind::NotAdmitted;
}

class OperatorWalker {
public:
  explicit OperatorWalker(QGEMMOperatorWalk &result)
      : result(result), rootBlock(result.candidate.reduce->getBlock()) {}

  void run() { visitOperation(result.candidate.reduce); }

private:
  void recordBoundary(Value value, Operation *producer,
                      QGEMMBoundaryKind kind) {
    if (!seenBoundaries.insert(value).second)
      return;

    result.boundaries.push_back({value, producer, kind});
  }

  void visitValue(Value value) {
    if (!isTraceableValue(value) || !seenValues.insert(value).second)
      return;

    Operation *producer = value.getDefiningOp();
    QGEMMBoundaryKind kind = classifyBoundary(value, producer, rootBlock);
    if (kind != QGEMMBoundaryKind::NotAdmitted || !producer ||
        isQGEMMOperatorBoundary(producer)) {
      recordBoundary(value, producer, kind);
      return;
    }

    visitOperation(producer);
  }

  void visitOperation(Operation *operation) {
    if (!result.operations.insert(operation))
      return;

    // Traverse every tensor-like operand. In particular, do not apply the
    // candidate-root broadcast-edge cutoff here: broadcast leaves are part of
    // the operator and diamonds are naturally deduplicated by SetVector.
    for (OpOperand &operand : operation->getOpOperands())
      visitValue(operand.get());

    // Tensor operands captured by a region operation are not exposed through
    // the admitted operation's own operand list. Walk nested operations to
    // include gather tables and other captured tensor values in the same
    // backwards boundary traversal.
    for (Region &region : operation->getRegions())
      region.walk([&](Operation *nested) {
        for (OpOperand &operand : nested->getOpOperands())
          visitValue(operand.get());
      });
  }

  QGEMMOperatorWalk &result;
  Block *rootBlock;
  llvm::DenseSet<Value> seenValues;
  llvm::DenseSet<Value> seenBoundaries;
};

} // namespace

QGEMMOperatorWalk walkQGEMMOperator(QGEMMCandidateRoot candidate) {
  QGEMMOperatorWalk result;
  result.candidate = candidate;

  OperatorWalker walker(result);
  walker.run();

  LLVM_DEBUG(llvm::dbgs() << "[qgemm-recognize] op " << candidate.index
                          << " walk: ops=" << result.operations.size()
                          << " boundaries=" << result.boundaries.size()
                          << "\n");
  return result;
}

} // namespace buddy
} // namespace mlir
