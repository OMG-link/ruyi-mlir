//===- QGEMMRecognizePass.cpp - QGEMM recognizer pass -----------*- C++ -*-===//
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
// This file implements the QGEMM recognition pass and its initial candidate
// enumeration stage. Later tasks extend the body with structure recovery and
// annotation while preserving this pass boundary.
//
//===----------------------------------------------------------------------===//

#include "CandidateRoots.h"
#include "OperatorWalk.h"

#include "Conversion/QGEMMRecognize/QGEMMRecognizePass.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "qgemm-recognize"

using namespace mlir;

namespace {

struct QGEMMRecognizePass
    : public PassWrapper<QGEMMRecognizePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QGEMMRecognizePass)

  StringRef getArgument() const override { return "qgemm-recognize"; }
  StringRef getDescription() const override {
    return "Recognize canonical-form QGEMM operators";
  }

  QGEMMRecognizePass() = default;
  QGEMMRecognizePass(const QGEMMRecognizePass &other) : PassWrapper(other) {}

  Option<bool> explain{*this, "explain",
                       llvm::cl::desc("Explain rejected QGEMM candidates"),
                       llvm::cl::init(false)};

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    SmallVector<buddy::QGEMMCandidateRoot> candidates =
        buddy::findQGEMMCandidateRoots(funcOp);
    LLVM_DEBUG(llvm::dbgs() << "[qgemm-recognize] candidates: "
                            << candidates.size() << "\n");
    for (const buddy::QGEMMCandidateRoot &candidate : candidates) {
      buddy::QGEMMOperatorWalk walk = buddy::walkQGEMMOperator(candidate);
      LLVM_DEBUG(llvm::dbgs()
                 << "[qgemm-recognize] op " << candidate.index
                 << " walk: ops=" << walk.operations.size()
                 << " boundaries=" << walk.boundaries.size() << "\n");
    }
  }
};

} // namespace

namespace mlir {
namespace buddy {

void registerQGEMMRecognizePass() { PassRegistration<QGEMMRecognizePass>(); }

} // namespace buddy
} // namespace mlir
