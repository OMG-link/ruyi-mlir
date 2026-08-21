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
#include "Recognition.h"

#include "Conversion/QGEMMRecognize/QGEMMRecognizePass.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "qgemm-recognize"

using namespace mlir;

namespace {

struct CandidateRecord {
  buddy::QGEMMCandidateRoot candidate;
  buddy::QGEMMOperatorWalk walk;
  buddy::Recognition recognition;
  std::optional<buddy::Rejection> rejection;
};

/// Return the first shared operation in the first candidate's walk order.
/// Boundary operations are excluded explicitly even though OperatorWalk does
/// not normally put them in Recognition::ops.  Keeping the exclusion here
/// makes the ownership rule independent of how a boundary is represented.
static Operation *
findSharedNonBoundaryOperation(const buddy::Recognition &first,
                               const buddy::Recognition &second) {
  for (Operation *operation : first.ops) {
    if (!second.ops.contains(operation) ||
        first.boundaries.contains(operation) ||
        second.boundaries.contains(operation))
      continue;
    return operation;
  }
  return nullptr;
}

/// Arbitrate ownership only after every candidate has been walked and its
/// three per-candidate checks have run.  Candidate pairs and each candidate's
/// SetVector walk order are the only ordering inputs; no DenseMap iteration is
/// involved.  A prior rejection is retained because ownership is the fourth
/// fixed-order condition, not a replacement for an earlier failure.
static void
arbitrateOwnership(llvm::SmallVectorImpl<CandidateRecord> &records) {
  for (unsigned firstIndex = 0; firstIndex < records.size(); ++firstIndex) {
    for (unsigned secondIndex = firstIndex + 1; secondIndex < records.size();
         ++secondIndex) {
      Operation *shared = findSharedNonBoundaryOperation(
          records[firstIndex].recognition, records[secondIndex].recognition);
      if (!shared)
        continue;

      // A1 section 13 names this fourth condition as candidates sharing a
      // non-boundary operation.  Keep this trace/remark identifier stable.
      buddy::Rejection ownership{shared, "shared-non-boundary-operation", 13};
      if (!records[firstIndex].rejection)
        records[firstIndex].rejection = ownership;
      if (!records[secondIndex].rejection)
        records[secondIndex].rejection = ownership;
    }
  }
}

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
    AsmState asmState(funcOp.getOperation());
    (void)asmState;
    SmallVector<buddy::QGEMMCandidateRoot> candidates =
        buddy::findQGEMMCandidateRoots(funcOp);
    LLVM_DEBUG(llvm::dbgs() << "[qgemm-recognize] candidates: "
                            << candidates.size() << "\n");

    // Keep all walks and the first three legality results before making any
    // accepted/rejected decision.  Ownership is the fourth check below.
    llvm::SmallVector<CandidateRecord, 1> records;
    records.reserve(candidates.size());
    for (const buddy::QGEMMCandidateRoot &candidate : candidates) {
      CandidateRecord record;
      record.candidate = candidate;
      record.walk = buddy::walkQGEMMOperator(candidate);
      LLVM_DEBUG(llvm::dbgs()
                 << "[qgemm-recognize] op " << candidate.index
                 << " walk: ops=" << record.walk.operations.size()
                 << " boundaries=" << record.walk.boundaries.size() << "\n");

      record.recognition = buddy::buildQGEMMRecognition(record.walk);
      record.rejection = buddy::checkQGEMMLegality(record.recognition);
      records.push_back(std::move(record));
    }

    arbitrateOwnership(records);

    unsigned recognizedRoot = 0;
    for (CandidateRecord &record : records) {
      if (record.rejection) {
        const buddy::Rejection &rejection = *record.rejection;
        LLVM_DEBUG({
          llvm::dbgs() << "[qgemm-recognize] candidate "
                       << record.candidate.index
                       << " rejected: reason=" << rejection.reason
                       << " section=" << rejection.section << " at=";
          if (rejection.at)
            llvm::dbgs() << rejection.at->getName().getStringRef();
          else
            llvm::dbgs() << "<null>";
          llvm::dbgs() << " value=";
          if (rejection.at && rejection.at->getNumResults() == 1)
            rejection.at->getResult(0).printAsOperand(llvm::dbgs(), asmState);
          else
            llvm::dbgs() << "<none>";
          llvm::dbgs() << " loc=";
          if (rejection.at)
            rejection.at->getLoc().print(llvm::dbgs());
          else
            llvm::dbgs() << "<none>";
          llvm::dbgs() << "\n";
        });
        if (explain) {
          record.recognition.finalReduce->emitRemark()
              << "qgemm-recognize: rejected candidate: reason="
              << rejection.reason << " section=" << rejection.section
              << " (A1 canonical-form section)";
        }
        continue;
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "[qgemm-recognize] op " << record.candidate.index
                 << " accepted: root=" << recognizedRoot << "\n");
      ++recognizedRoot;
    }
  }
};

} // namespace

namespace mlir {
namespace buddy {

void registerQGEMMRecognizePass() { PassRegistration<QGEMMRecognizePass>(); }

} // namespace buddy
} // namespace mlir
