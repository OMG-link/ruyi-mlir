//===- QGEMMDimTransformPass.h - QGEMM shape simplification ---*- C++ -*-===//
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
// This file declares the QGEMM dimension-transform simplification pass.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_MLIR_QGEMM_DIM_TRANSFORM_PASS_H
#define BUDDY_MLIR_QGEMM_DIM_TRANSFORM_PASS_H

namespace mlir {
namespace buddy {

/// Register the --qgemm-simplify-dim-transforms pass.
void registerQGEMMDimTransformPass();

} // namespace buddy
} // namespace mlir

#endif // BUDDY_MLIR_QGEMM_DIM_TRANSFORM_PASS_H
