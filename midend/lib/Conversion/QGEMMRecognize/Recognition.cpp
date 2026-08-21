//===- Recognition.cpp - QGEMM recognition model ----------------*- C++ -*-===//
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
// This file reconstructs the scope, stage partition and broadcast dimensions
// needed by the QGEMM recognizer.  It intentionally does not assign algebraic
// roles to the element-wise operations between reductions.
//
//===----------------------------------------------------------------------===//

#include "Recognition.h"

#include "Conversion/QGEMMRecognize/QGEMMAnnotation.h"

#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>

using namespace mlir;

namespace mlir {
namespace buddy {
namespace {

using DimMapping = SmallVector<std::optional<unsigned>>;

struct BroadcastEdge {
  bool isBroadcast = false;
  SmallVector<unsigned> outputDimensions;
};

struct RecognitionAnalysis {
  DenseSet<Operation *> dataPrepOps;
  DenseMap<Operation *, StringRef> stages;
  DenseMap<unsigned, DenseSet<OpOperand *>> coreBroadcastEdgesByOutputDim;
  Operation *stageConflict = nullptr;
};

static bool isTraceableValue(Value value) {
  return isa<TensorType, BaseMemRefType>(value.getType());
}

static std::optional<unsigned> getRank(Value value) {
  auto shapedType = dyn_cast<ShapedType>(value.getType());
  if (!shapedType || !shapedType.hasRank())
    return std::nullopt;
  return shapedType.getRank();
}

static DimMapping makeIdentityMapping(unsigned rank) {
  DimMapping mapping;
  mapping.reserve(rank);
  for (unsigned dimension = 0; dimension < rank; ++dimension)
    mapping.push_back(dimension);
  return mapping;
}

static DimMapping makeNullMapping(unsigned rank) {
  return DimMapping(rank, std::nullopt);
}

static DimMapping resizeMapping(const DimMapping &mapping, unsigned rank) {
  DimMapping resized = makeNullMapping(rank);
  for (unsigned index = 0, end = std::min<unsigned>(mapping.size(), rank);
       index < end; ++index)
    resized[index] = mapping[index];
  return resized;
}

/// Map the result dimensions of an affine operand map through the consumer's
/// iteration-space mapping.  Only dimension expressions are meaningful to the
/// recognizer; symbols and compound expressions are intentionally unknown.
static DimMapping mapThroughAffineMap(AffineMap map,
                                      const DimMapping &iterationMapping) {
  DimMapping mapping;
  mapping.reserve(map.getNumResults());
  for (AffineExpr expression : map.getResults()) {
    if (auto dimension = dyn_cast<AffineDimExpr>(expression)) {
      unsigned position = dimension.getPosition();
      if (position < iterationMapping.size()) {
        mapping.push_back(iterationMapping[position]);
        continue;
      }
    }
    mapping.push_back(std::nullopt);
  }
  return mapping;
}

/// Return the iteration-space mapping of an operation from the mapping of its
/// result.  A reduction iterator is explicitly mapped to no final output
/// dimension.  This also handles a linalg.generic with reduction iterators,
/// which is the form produced when simplification rebuilds a non-root reduce.
static DimMapping getIterationMapping(Operation *operation,
                                      const DimMapping &resultMapping) {
  if (auto reduce = dyn_cast<linalg::ReduceOp>(operation)) {
    if (reduce.getInputs().empty())
      return resultMapping;

    auto inputRank = getRank(reduce.getInputs().front());
    if (!inputRank)
      return {};

    DenseSet<int64_t> reductionDimensions;
    for (int64_t dimension : reduce.getDimensions())
      reductionDimensions.insert(dimension);

    DimMapping mapping = makeNullMapping(*inputRank);
    unsigned resultDimension = 0;
    for (unsigned inputDimension = 0; inputDimension < *inputRank;
         ++inputDimension) {
      if (reductionDimensions.contains(inputDimension))
        continue;
      if (resultDimension < resultMapping.size())
        mapping[inputDimension] = resultMapping[resultDimension];
      ++resultDimension;
    }
    return mapping;
  }

  if (auto generic = dyn_cast<linalg::GenericOp>(operation)) {
    SmallVector<utils::IteratorType> iteratorTypes =
        generic.getIteratorTypesArray();
    DimMapping mapping = makeNullMapping(iteratorTypes.size());
    if (generic.getNumDpsInits() == 0)
      return mapping;

    unsigned outputOperandNumber = generic.getNumDpsInputs();
    if (outputOperandNumber >= operation->getNumOperands())
      return mapping;

    OpOperand &output = operation->getOpOperand(outputOperandNumber);
    AffineMap outputMap = generic.getMatchingIndexingMap(&output);
    for (auto [resultDimension, expression] :
         llvm::enumerate(outputMap.getResults())) {
      auto dimension = dyn_cast<AffineDimExpr>(expression);
      if (!dimension || resultDimension >= resultMapping.size())
        continue;
      unsigned position = dimension.getPosition();
      if (position < mapping.size())
        mapping[position] = resultMapping[resultDimension];
    }

    // Do not infer a reduction from an operation name.  The iterator type is
    // the authoritative source, including for generic reductions rebuilt by
    // the dimension-transform pass.
    if (isQGEMMReduction(operation)) {
      for (auto [dimension, iteratorType] : llvm::enumerate(iteratorTypes)) {
        if (linalg::isReductionIterator(iteratorType))
          mapping[dimension] = std::nullopt;
      }
    }
    return mapping;
  }

  auto rank = operation->getNumResults() ? getRank(operation->getResult(0))
                                         : std::optional<unsigned>();
  if (!rank)
    return resultMapping;
  return resizeMapping(resultMapping, *rank);
}

/// Compute the mapping of the producer result for one consumer operand.
static DimMapping getProducerMapping(Operation *consumer, OpOperand &operand,
                                     const DimMapping &iterationMapping) {
  if (auto generic = dyn_cast<linalg::GenericOp>(consumer))
    return mapThroughAffineMap(generic.getMatchingIndexingMap(&operand),
                               iterationMapping);

  if (auto reduce = dyn_cast<linalg::ReduceOp>(consumer)) {
    unsigned inputCount = reduce.getInputs().size();
    if (operand.getOperandNumber() < inputCount)
      return iterationMapping;

    DenseSet<int64_t> reductionDimensions;
    for (int64_t dimension : reduce.getDimensions())
      reductionDimensions.insert(dimension);

    DimMapping mapping;
    for (unsigned dimension = 0; dimension < iterationMapping.size();
         ++dimension) {
      if (!reductionDimensions.contains(dimension))
        mapping.push_back(iterationMapping[dimension]);
    }
    return mapping;
  }

  if (auto transpose = dyn_cast<linalg::TransposeOp>(consumer)) {
    if (operand.getOperandNumber() ==
        transpose.getInputMutable().getOperandNumber()) {
      ArrayRef<int64_t> permutation = transpose.getPermutation();
      DimMapping mapping = makeNullMapping(permutation.size());
      for (unsigned outputDimension = 0; outputDimension < permutation.size();
           ++outputDimension) {
        int64_t inputDimension = permutation[outputDimension];
        if (inputDimension >= 0 &&
            static_cast<unsigned>(inputDimension) < mapping.size() &&
            outputDimension < iterationMapping.size())
          mapping[inputDimension] = iterationMapping[outputDimension];
      }
      return mapping;
    }
    return iterationMapping;
  }

  if (auto expand = dyn_cast<tensor::ExpandShapeOp>(consumer)) {
    SmallVector<ReassociationIndices> reassociation =
        expand.getReassociationIndices();
    DimMapping mapping;
    mapping.reserve(reassociation.size());
    for (ReassociationIndices group : reassociation) {
      std::optional<unsigned> knownDimension;
      for (int64_t outputDimension : group) {
        if (outputDimension < 0 ||
            static_cast<unsigned>(outputDimension) >= iterationMapping.size())
          continue;
        if (iterationMapping[outputDimension]) {
          knownDimension = iterationMapping[outputDimension];
          break;
        }
      }
      mapping.push_back(knownDimension);
    }
    return mapping;
  }

  if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(consumer)) {
    SmallVector<ReassociationIndices> reassociation =
        collapse.getReassociationIndices();
    DimMapping mapping;
    for (unsigned outputDimension = 0; outputDimension < reassociation.size();
         ++outputDimension) {
      for (int64_t ignored : reassociation[outputDimension]) {
        (void)ignored;
        if (outputDimension < iterationMapping.size())
          mapping.push_back(iterationMapping[outputDimension]);
        else
          mapping.push_back(std::nullopt);
      }
    }
    return mapping;
  }

  auto rank = getRank(operand.get());
  if (!rank)
    return {};
  return resizeMapping(iterationMapping, *rank);
}

/// Identify broadcast directions on a generic operand edge.  A dimension is
/// broadcast when the operand indexing map does not depend on a parallel
/// iterator dimension.  Constant map results and omitted dimensions therefore
/// cover both a standalone broadcast generic and a broadcast folded into the
/// consumer.
static BroadcastEdge getBroadcastEdge(Operation *consumer, OpOperand &operand,
                                      const DimMapping &iterationMapping) {
  auto generic = dyn_cast<linalg::GenericOp>(consumer);
  if (!generic)
    return {};

  SmallVector<utils::IteratorType> iteratorTypes =
      generic.getIteratorTypesArray();
  AffineMap map = generic.getMatchingIndexingMap(&operand);
  DenseSet<unsigned> usedDimensions;
  for (AffineExpr expression : map.getResults()) {
    if (auto dimension = dyn_cast<AffineDimExpr>(expression))
      usedDimensions.insert(dimension.getPosition());
  }

  BroadcastEdge edge;
  unsigned dimensionCount =
      std::min<unsigned>(iteratorTypes.size(), iterationMapping.size());
  for (unsigned dimension = 0; dimension < dimensionCount; ++dimension) {
    if (linalg::isReductionIterator(iteratorTypes[dimension]) ||
        usedDimensions.contains(dimension))
      continue;

    edge.isBroadcast = true;
    if (iterationMapping[dimension] &&
        !llvm::is_contained(edge.outputDimensions,
                            *iterationMapping[dimension]))
      edge.outputDimensions.push_back(*iterationMapping[dimension]);
  }
  return edge;
}

class RecognitionAnalyzer {
public:
  explicit RecognitionAnalyzer(const Recognition &recognition)
      : recognition(recognition) {}

  RecognitionAnalysis run() {
    linalg::ReduceOp finalReduce = recognition.finalReduce;
    if (finalReduce) {
      DimMapping resultMapping;
      if (finalReduce->getNumResults() != 0) {
        if (auto rank = getRank(finalReduce->getResult(0)))
          resultMapping = makeIdentityMapping(*rank);
      }
      visitOperation(
          finalReduce.getOperation(),
          getIterationMapping(finalReduce.getOperation(), resultMapping),
          /*dataPrep=*/false);
    }

    for (Operation *operation : recognition.ops) {
      analysis.stages[operation] = analysis.dataPrepOps.contains(operation)
                                       ? kQGEMMDataPrepStage
                                       : kQGEMMComputeStage;
    }
    return std::move(analysis);
  }

private:
  void visitOperation(Operation *operation, const DimMapping &iterationMapping,
                      bool dataPrep) {
    if (!recognition.ops.contains(operation))
      return;

    auto [it, inserted] = visitedStages.try_emplace(operation, dataPrep);
    if (!inserted) {
      if (it->second != dataPrep && !analysis.stageConflict)
        analysis.stageConflict = operation;
      return;
    }

    if (dataPrep)
      analysis.dataPrepOps.insert(operation);

    for (OpOperand &operand : operation->getOpOperands()) {
      if (!isTraceableValue(operand.get()))
        continue;

      BroadcastEdge edge =
          getBroadcastEdge(operation, operand, iterationMapping);
      if (!dataPrep && edge.isBroadcast && edge.outputDimensions.size() == 1) {
        unsigned outputDim = edge.outputDimensions.front();
        analysis.coreBroadcastEdgesByOutputDim[outputDim].insert(&operand);
      }

      Operation *producer = operand.get().getDefiningOp();
      if (!producer || !recognition.ops.contains(producer))
        continue;

      DimMapping producerResultMapping =
          getProducerMapping(operation, operand, iterationMapping);
      DimMapping producerIterationMapping =
          getIterationMapping(producer, producerResultMapping);
      visitOperation(producer, producerIterationMapping,
                     dataPrep || edge.isBroadcast);
    }
  }

  const Recognition &recognition;
  RecognitionAnalysis analysis;
  DenseMap<Operation *, bool> visitedStages;
};

static RecognitionAnalysis analyzeRecognition(const Recognition &recognition) {
  return RecognitionAnalyzer(recognition).run();
}

static bool hasRankTwoResult(linalg::ReduceOp reduce) {
  if (!reduce || reduce->getNumResults() != 1)
    return false;
  auto resultType = dyn_cast<ShapedType>(reduce->getResult(0).getType());
  return resultType && resultType.hasRank() && resultType.getRank() == 2;
}

} // namespace

bool isQGEMMReduction(Operation *operation) {
  if (isa_and_nonnull<linalg::ReduceOp>(operation))
    return true;
  auto generic = dyn_cast_or_null<linalg::GenericOp>(operation);
  if (!generic)
    return false;
  return llvm::any_of(generic.getIteratorTypesArray(),
                      [](utils::IteratorType iteratorType) {
                        return linalg::isReductionIterator(iteratorType);
                      });
}

Recognition buildQGEMMRecognition(const QGEMMOperatorWalk &walk) {
  Recognition recognition;
  recognition.finalReduce = walk.candidate.reduce;
  recognition.ops = walk.operations;

  // OperatorWalk records boundary values.  The Recognition model is the
  // operation-level view needed by ownership arbitration; block arguments
  // have no defining operation and consequently do not enter this set.
  for (const QGEMMBoundary &boundary : walk.boundaries) {
    if (boundary.producer)
      recognition.boundaries.insert(boundary.producer);
  }

  if (recognition.finalReduce &&
      recognition.finalReduce->getNumResults() == 1 &&
      hasRankTwoResult(recognition.finalReduce)) {
    recognition.mDim = 0;
    recognition.nDim = 1;
  }

  RecognitionAnalysis analysis = analyzeRecognition(recognition);
  recognition.dataPrepOps = std::move(analysis.dataPrepOps);
  recognition.stages = std::move(analysis.stages);
  return recognition;
}

std::optional<Rejection> checkQGEMMLegality(const Recognition &recognition) {
  // Fixed check order: (1) final-result rank, (2) stage partition conflict,
  // (3) M/N core broadcast coverage, then (4) cross-candidate ownership
  // arbitration after all candidate records have been reconstructed.  The
  // fourth check is intentionally owned by the next task.
  linalg::ReduceOp finalReduce = recognition.finalReduce;
  if (!hasRankTwoResult(finalReduce))
    return Rejection{finalReduce.getOperation(), "final-result-rank", 4};

  RecognitionAnalysis analysis = analyzeRecognition(recognition);
  if (analysis.stageConflict)
    return Rejection{analysis.stageConflict, "stage-conflict", 2};

  auto hasCoreBroadcastFor = [&](unsigned outputDim) {
    auto it = analysis.coreBroadcastEdgesByOutputDim.find(outputDim);
    return it != analysis.coreBroadcastEdgesByOutputDim.end() &&
           !it->second.empty();
  };
  if (!hasCoreBroadcastFor(recognition.mDim) ||
      !hasCoreBroadcastFor(recognition.nDim))
    return Rejection{finalReduce.getOperation(), "core-broadcast-dimensions",
                     2};

  return std::nullopt;
}

} // namespace buddy
} // namespace mlir
