//===- DimensionClassifier.h - Classify M/N/Batch dimensions ------*- C++
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
// This file declares the DimensionClassifier utility, which analyzes broadcast
// patterns in a ContractionSubgraph to classify each output dimension of the
// tiling root as M, N, or Batch.
//
//===----------------------------------------------------------------------===//

#ifndef QGEMM_TILING_DIMENSION_CLASSIFIER_H
#define QGEMM_TILING_DIMENSION_CLASSIFIER_H

#include "ContractionSubgraph.h"

#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace buddy {

/// Role of each dimension in the tiling root's output.
enum class DimRole { M, N, Batch };

/// Result of dimension classification.
struct DimensionClassification {
  /// Role of each dimension in the tiling root's output.
  /// Length = rank of tiling root's output tensor.
  SmallVector<DimRole> roles;

  /// Index of the M dimension in the tiling root's output.
  unsigned mDim;

  /// Index of the N dimension in the tiling root's output.
  unsigned nDim;
};

/// Classify dimensions of the tiling root's output as M, N, or Batch based
/// on the broadcast pattern detected during subgraph collection.
///
/// Returns failure if classification is not possible (0, 1, or >2 broadcast
/// dimensions detected).
FailureOr<DimensionClassification>
classifyDimensions(const ContractionSubgraph &subgraph);

} // namespace buddy
} // namespace mlir

#endif // QGEMM_TILING_DIMENSION_CLASSIFIER_H
