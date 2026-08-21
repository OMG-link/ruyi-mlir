//===- QGEMMAnnotation.h - QGEMM identity annotation declarations -*- C++ -*-===
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
// This header is the single source of the identity-annotation attribute and
// field names shared by QGEMM recognition and its downstream consumers.
//
// Attribute-name measurement (task 1.1): a throwaway input attaching the
// qgemm.-prefixed discardable attributes to linalg.generic and linalg.reduce
// passed buddy-opt verification, and a buddy-opt print-parse-print round trip
// was byte-identical. On linalg.reduce the attributes sit in the trailing
// attribute dictionary, after dimensions = [...] and before the combiner
// region; a dictionary placed immediately after the op name is rejected there,
// flat or nested. On linalg.generic they sit inside the attrs = { ... } clause,
// next to the broadcastDims attribute triton-shared attaches, with entries of
// one dictionary printed in alphabetical order. Both cases print in the custom
// assembly form, so no generic form is needed anywhere. The prefixed names are
// therefore selected -- qgemm.node and qgemm.root -- and the unprefixed
// fallback is unused.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_MLIR_QGEMM_ANNOTATION_H
#define BUDDY_MLIR_QGEMM_ANNOTATION_H

#include "llvm/ADT/StringRef.h"

#include <optional>
#include <utility>

namespace mlir {
class Operation;

namespace buddy {

inline constexpr llvm::StringLiteral kQGEMMNodeAttr = "qgemm.node";
inline constexpr llvm::StringLiteral kQGEMMRootAttr = "qgemm.root";
inline constexpr llvm::StringLiteral kQGEMMNodeRootField = "root";
inline constexpr llvm::StringLiteral kQGEMMNodeStageField = "stage";
inline constexpr llvm::StringLiteral kQGEMMRootMDimField = "m_dim";
inline constexpr llvm::StringLiteral kQGEMMRootNDimField = "n_dim";
inline constexpr llvm::StringLiteral kQGEMMDataPrepStage = "data_prep";
inline constexpr llvm::StringLiteral kQGEMMComputeStage = "compute";

/// Return the root identity carried by a qgemm.node annotation, if present.
std::optional<unsigned> getQGEMMNodeRoot(Operation *op);

/// Return the M/N dimensions carried by a qgemm.root annotation, if present.
std::optional<std::pair<unsigned, unsigned>> getQGEMMRootDims(Operation *op);

} // namespace buddy
} // namespace mlir

#endif // BUDDY_MLIR_QGEMM_ANNOTATION_H
