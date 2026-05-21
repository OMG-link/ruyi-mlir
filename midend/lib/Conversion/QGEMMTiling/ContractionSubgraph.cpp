//===- ContractionSubgraph.cpp - Collect contraction subgraph -----*- C++
//-*-===//
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
// This file implements backward BFS from the tiling root to collect the
// contraction subgraph, compute per-op dimension mappings to the tiling root's
// output space, and detect broadcast dimensions.
//
// The BFS processes each op's operands (ins + outs/inits uniformly) and for
// each defining op decides:
//   - Whether to add it to the subgraph's op set.
//   - Whether to enqueue it for further traversal.
//
// Broadcast inputs (generic input operands whose indexing map contains a
// constant expression) are skipped — their defining ops are not added to the
// subgraph and not traversed.
//
//===----------------------------------------------------------------------===//

#include "ContractionSubgraph.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#include <deque>

#define DEBUG_TYPE "qgemm-tiling"

using namespace mlir;

namespace mlir {
namespace buddy {

//===----------------------------------------------------------------------===//
// Operand classification helpers
//===----------------------------------------------------------------------===//

/// Return true if `operand` is an input of `genericOp` whose indexing map
/// contains a constant expression (i.e., the input is broadcast along at
/// least one iteration-space dimension).
static bool isBroadcastInput(linalg::GenericOp genericOp, OpOperand *operand) {
  AffineMap map = genericOp.getMatchingIndexingMap(operand);
  for (AffineExpr expr : map.getResults()) {
    if (isa<AffineConstantExpr>(expr))
      return true;
  }
  return false;
}

/// Return true if `reduceOp` uses an additive combiner (arith.addi or
/// arith.addf in its combiner region).
static bool hasAdditiveCombiner(linalg::ReduceOp reduceOp) {
  Block &body = reduceOp.getCombiner().front();
  for (Operation &op : body.without_terminator()) {
    if (isa<arith::AddIOp, arith::AddFOp>(op))
      return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// DimMapping composition helpers
//===----------------------------------------------------------------------===//

/// Compose a DimMapping through an affine indexing map. For each result
/// position r of the map:
///   - AffineDimExpr(d) → knownMapping[d]
///   - AffineConstantExpr → nullopt (broadcast dimension)
static DimMapping composeViaIndexingMap(const DimMapping &knownMapping,
                                        AffineMap map) {
  unsigned resultRank = map.getNumResults();
  DimMapping result(resultRank);
  for (unsigned r = 0; r < resultRank; ++r) {
    AffineExpr expr = map.getResult(r);
    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr))
      result[r] = knownMapping[dimExpr.getPosition()];
    else
      result[r] = std::nullopt;
  }
  return result;
}

/// Build identity DimMapping of length `numDims`: dim i → i.
static DimMapping buildIdentityMapping(unsigned numDims) {
  DimMapping mapping(numDims);
  for (unsigned i = 0; i < numDims; ++i)
    mapping[i] = i;
  return mapping;
}

/// Return true if two DimMappings are element-wise equal.
static bool areMappingsConsistent(const DimMapping &a, const DimMapping &b) {
  if (a.size() != b.size())
    return false;
  for (unsigned i = 0; i < a.size(); ++i) {
    if (a[i] != b[i])
      return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Broadcast detection
//===----------------------------------------------------------------------===//

/// For each input operand of `genericOp`, find iteration-space dimensions
/// that are not referenced by the operand's indexing map (i.e., the input is
/// broadcast along those dims). Project each such dim through `opDimMapping`
/// to the tiling root's output space and mark it in `hasBroadcast`.
static void detectBroadcasts(linalg::GenericOp genericOp,
                             const DimMapping &opDimMapping,
                             SmallVectorImpl<bool> &hasBroadcast) {
  for (OpOperand *input : genericOp.getDpsInputOperands()) {
    AffineMap map = genericOp.getMatchingIndexingMap(input);
    unsigned numIterDims = map.getNumDims();
    // Collect which iteration-space dims appear in this map.
    llvm::DenseSet<unsigned> usedIterDims;
    for (AffineExpr expr : map.getResults()) {
      if (auto dimExpr = dyn_cast<AffineDimExpr>(expr))
        usedIterDims.insert(dimExpr.getPosition());
    }
    // Dims not used by this input are broadcast dims.
    for (unsigned k = 0; k < numIterDims; ++k) {
      if (!usedIterDims.contains(k)) {
        if (k < opDimMapping.size() && opDimMapping[k].has_value()) {
          unsigned outputDim = *opDimMapping[k];
          if (outputDim < hasBroadcast.size()) {
            LLVM_DEBUG(llvm::dbgs() << "  Broadcast detected: iter dim " << k
                                    << " -> output dim " << outputDim << "\n");
            hasBroadcast[outputDim] = true;
          } else {
            llvm_unreachable("Out of bounds error.");
          }
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// computeEdgeMapping: knownOp's iter-space → defOp's output tensor
//===----------------------------------------------------------------------===//

/// Given knownOp's iteration-space DimMapping, compute the DimMapping for the
/// result tensor produced by the defining op of `operand`. This represents
/// how defOp's output dimensions map to the tiling root's output space.
///
/// Precondition: knownOp must be a structured linalg op with a DimMapping
/// entry, and operand must be one of its operands. Violating this is a
/// programming error.
static DimMapping computeEdgeMapping(Operation *knownOp, Value operand,
                                     const DimMapping &knownMapping) {
  if (auto genericOp = dyn_cast<linalg::GenericOp>(knownOp)) {
    for (OpOperand &opOperand : genericOp->getOpOperands()) {
      if (opOperand.get() == operand) {
        AffineMap map = genericOp.getMatchingIndexingMap(&opOperand);
        return composeViaIndexingMap(knownMapping, map);
      }
    }
    llvm_unreachable("operand not found in generic's OpOperands");
  }
  if (auto reduceOp = dyn_cast<linalg::ReduceOp>(knownOp)) {
    // Input: same rank as iteration space → pass through.
    for (Value input : reduceOp.getInputs()) {
      if (input == operand)
        return DimMapping(knownMapping);
    }
    // Init: output rank (parallel dims only) → extract non-reduction entries.
    llvm::DenseSet<int64_t> redDimSet(reduceOp.getDimensions().begin(),
                                      reduceOp.getDimensions().end());
    unsigned inputRank =
        cast<RankedTensorType>(reduceOp.getInputs()[0].getType()).getRank();
    DimMapping initMapping;
    for (unsigned i = 0; i < inputRank; ++i) {
      if (!redDimSet.contains(i))
        initMapping.push_back(knownMapping[i]);
    }
    return initMapping;
  }
  if (auto transposeOp = dyn_cast<linalg::TransposeOp>(knownOp)) {
    if (operand == transposeOp.getInput()) {
      // Transpose: output[i] = input[perm[i]], so input[j] = output[invP[j]].
      ArrayRef<int64_t> perm = transposeOp.getPermutation();
      unsigned rank = perm.size();
      SmallVector<unsigned> invPerm(rank);
      for (unsigned i = 0; i < rank; ++i)
        invPerm[perm[i]] = i;
      DimMapping result(rank);
      for (unsigned j = 0; j < rank; ++j)
        result[j] = knownMapping[invPerm[j]];
      return result;
    }
    // Init has same shape as output → same mapping.
    return DimMapping(knownMapping);
  }
  if (isa<linalg::FillOp>(knownOp)) {
    // Fill's output and init have the same shape → same mapping.
    return DimMapping(knownMapping);
  }
  llvm_unreachable("computeEdgeMapping called on op without DimMapping");
}

//===----------------------------------------------------------------------===//
// computeIterSpaceMapping: defOp's output mapping → defOp's iter-space mapping
//===----------------------------------------------------------------------===//

/// Given defOp's output DimMapping (how its result tensor dims map to the
/// tiling root's output space), compute defOp's iteration-space DimMapping.
///
/// For generic, transpose, and fill, output rank = iter-space rank, so the
/// mapping passes through unchanged. For reduce, the iteration space is larger
/// than the output (it includes reduction dims), so nullopt entries are
/// inserted at the reduction dim positions.
///
/// Returns std::nullopt if defOp is not a recognized structured linalg op,
/// signaling the BFS to stop traversal at this op (it is not part of the
/// contraction subgraph).
static std::optional<DimMapping>
computeIterSpaceMapping(Operation *defOp, const DimMapping &outputMapping) {
  if (isa<linalg::GenericOp>(defOp) || isa<linalg::TransposeOp>(defOp) ||
      isa<linalg::FillOp>(defOp)) {
    return DimMapping(outputMapping);
  }
  if (auto reduceOp = dyn_cast<linalg::ReduceOp>(defOp)) {
    ArrayRef<int64_t> reductionDims = reduceOp.getDimensions();
    unsigned inputRank =
        cast<RankedTensorType>(reduceOp.getInputs()[0].getType()).getRank();
    llvm::DenseSet<int64_t> redDimSet(reductionDims.begin(),
                                      reductionDims.end());
    DimMapping iterMapping(inputRank);
    unsigned outputIdx = 0;
    for (unsigned i = 0; i < inputRank; ++i) {
      if (redDimSet.contains(i)) {
        iterMapping[i] = std::nullopt;
      } else {
        iterMapping[i] = (outputIdx < outputMapping.size())
                             ? outputMapping[outputIdx]
                             : std::nullopt;
        ++outputIdx;
      }
    }
    return iterMapping;
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// collectSubgraph: backward BFS
//===----------------------------------------------------------------------===//

/// Collect all operands (ins + outs/inits) of a structured linalg op.
static SmallVector<Value> getAllOperands(Operation *op) {
  SmallVector<Value> operands;
  if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
    for (OpOperand *input : genericOp.getDpsInputOperands())
      operands.push_back(input->get());
    for (Value init : genericOp.getDpsInits())
      operands.push_back(init);
  } else if (auto reduceOp = dyn_cast<linalg::ReduceOp>(op)) {
    for (Value input : reduceOp.getInputs())
      operands.push_back(input);
    for (Value init : reduceOp.getInits())
      operands.push_back(init);
  } else if (auto transposeOp = dyn_cast<linalg::TransposeOp>(op)) {
    operands.push_back(transposeOp.getInput());
    operands.push_back(transposeOp.getInit());
  } else if (auto fillOp = dyn_cast<linalg::FillOp>(op)) {
    // Fill's inputs are scalars (not tensor producers). Only collect outputs.
    for (Value out : fillOp.getOutputs())
      operands.push_back(out);
  }
  return operands;
}

FailureOr<ContractionSubgraph> collectSubgraph(linalg::LinalgOp tilingRoot) {
  ContractionSubgraph subgraph;
  subgraph.tilingRoot = tilingRoot;

  // Determine the output rank of the tiling root (for hasBroadcast sizing).
  // For any LinalgOp, the output rank is the rank of its first result tensor.
  // This works uniformly for ReduceOp, GenericOp, FillOp, etc.
  auto resultType = cast<RankedTensorType>(tilingRoot->getResult(0).getType());
  unsigned outputRank = resultType.getRank();

  subgraph.hasBroadcast.resize(outputRank, false);

  // Initialize the tiling root's DimMapping. The output mapping is identity
  // (output dim i → tiling root output dim i). computeIterSpaceMapping then
  // expands it to the full iteration space (inserting nullopt for reduction
  // dims if the root is a reduce).
  DimMapping rootOutputMapping = buildIdentityMapping(outputRank);
  auto rootIterMapping =
      computeIterSpaceMapping(tilingRoot.getOperation(), rootOutputMapping);
  assert(rootIterMapping.has_value() &&
         "tiling root must be a supported structured op");
  subgraph.dimMappings[tilingRoot.getOperation()] = *rootIterMapping;
  subgraph.ops.insert(tilingRoot.getOperation());

  if (auto reduceOp = dyn_cast<linalg::ReduceOp>(tilingRoot.getOperation()))
    subgraph.reduces.push_back(reduceOp);

  // BFS worklist.
  std::deque<Operation *> worklist;
  worklist.push_back(tilingRoot.getOperation());

  while (!worklist.empty()) {
    Operation *knownOp = worklist.front();
    worklist.pop_front();

    const DimMapping &knownMapping = subgraph.dimMappings[knownOp];
    SmallVector<Value> operands = getAllOperands(knownOp);

    for (Value operand : operands) {
      Operation *defOp = operand.getDefiningOp();
      if (!defOp)
        continue;

      // Skip broadcast inputs: if this operand is an input of a generic
      // whose indexing map has a constant expression, its defining op is
      // not part of the contraction subgraph.
      if (auto genericOp = dyn_cast<linalg::GenericOp>(knownOp)) {
        bool skip = false;
        for (OpOperand *inp : genericOp.getDpsInputOperands()) {
          if (inp->get() == operand && isBroadcastInput(genericOp, inp)) {
            LLVM_DEBUG(llvm::dbgs()
                       << "  Broadcast input (skip): " << *defOp << "\n");
            skip = true;
            break;
          }
        }
        if (skip)
          continue;
      }

      // --- Leaf: tensor.empty has no iteration space, just collect it ---

      if (isa<tensor::EmptyOp>(defOp)) {
        if (!subgraph.ops.contains(defOp)) {
          subgraph.ops.insert(defOp);
          LLVM_DEBUG(llvm::dbgs()
                     << "  tensor.empty (stop): " << *defOp << "\n");
        }
        continue;
      }

      // --- Structured linalg ops: compute mapping, check diamond, add ---

      // Step 1: Compute defOp's output mapping from the edge.
      DimMapping outputMapping =
          computeEdgeMapping(knownOp, operand, knownMapping);

      // Step 2: Compute defOp's iteration-space mapping.
      std::optional<DimMapping> iterMapping =
          computeIterSpaceMapping(defOp, outputMapping);
      if (!iterMapping.has_value()) {
        LLVM_DEBUG(llvm::dbgs() << "  Unrecognized op (stop, not included): "
                                << *defOp << "\n");
        continue;
      }

      // Step 3: Diamond consistency check.
      if (subgraph.dimMappings.count(defOp)) {
        if (!areMappingsConsistent(subgraph.dimMappings[defOp], *iterMapping)) {
          LLVM_DEBUG(llvm::dbgs() << "Inconsistent DimMapping at diamond: "
                                  << *defOp << "\n");
          defOp->emitRemark("inconsistent DimMapping at diamond merge point");
          return failure();
        }
        continue;
      }

      // Step 4: Op-specific validation.
      if (auto reduceOp = dyn_cast<linalg::ReduceOp>(defOp)) {
        if (!hasAdditiveCombiner(reduceOp)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "  Non-additive reduce (stop): " << *defOp << "\n");
          continue;
        }
      }

      // Step 5: Record and enqueue.
      subgraph.dimMappings[defOp] = *iterMapping;
      subgraph.ops.insert(defOp);
      worklist.push_back(defOp);

      if (auto newGeneric = dyn_cast<linalg::GenericOp>(defOp)) {
        detectBroadcasts(newGeneric, *iterMapping, subgraph.hasBroadcast);
        LLVM_DEBUG(llvm::dbgs() << "  Generic (continue): " << *defOp << "\n");
      } else if (auto newReduce = dyn_cast<linalg::ReduceOp>(defOp)) {
        subgraph.reduces.push_back(newReduce);
        LLVM_DEBUG(llvm::dbgs() << "  Reduce (continue): " << *defOp << "\n");
      } else if (isa<linalg::TransposeOp>(defOp)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  Transpose (continue): " << *defOp << "\n");
      } else if (isa<linalg::FillOp>(defOp)) {
        LLVM_DEBUG(llvm::dbgs() << "  Fill (continue): " << *defOp << "\n");
      } else {
        llvm_unreachable("computeIterSpaceMapping succeeded on unexpected op");
      }
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "ContractionSubgraph collected:\n";
    llvm::dbgs() << "  ops: " << subgraph.ops.size() << "\n";
    llvm::dbgs() << "  reduces: " << subgraph.reduces.size() << "\n";
    llvm::dbgs() << "  hasBroadcast: [";
    for (unsigned i = 0; i < subgraph.hasBroadcast.size(); ++i) {
      if (i > 0)
        llvm::dbgs() << ", ";
      llvm::dbgs() << (subgraph.hasBroadcast[i] ? "true" : "false");
    }
    llvm::dbgs() << "]\n";
  });

  return subgraph;
}

} // namespace buddy
} // namespace mlir
