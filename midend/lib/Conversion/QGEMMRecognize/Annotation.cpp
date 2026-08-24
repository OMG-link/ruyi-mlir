//===- Annotation.cpp - QGEMM identity annotation ----------------*- C++
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
// This file writes and reads the operation-local identity attributes shared
// by QGEMM recognition and its downstream consumers.  The attributes contain
// only literals, so cloning an operation preserves identity without keeping a
// reference to an operation or value.
//
//===----------------------------------------------------------------------===//

#include "Conversion/QGEMMRecognize/QGEMMAnnotation.h"

#include "Recognition.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdint>
#include <limits>

using namespace mlir;

namespace mlir {
namespace buddy {
namespace {

static Attribute getQGEMMField(Operation *operation, StringRef attributeName,
                               StringRef fieldName) {
  if (!operation)
    return {};
  auto dictionary = operation->getAttrOfType<DictionaryAttr>(attributeName);
  if (!dictionary)
    return {};
  return dictionary.get(fieldName);
}

static std::optional<unsigned> getNonNegativeUnsigned(Attribute attribute) {
  auto integer = dyn_cast_or_null<IntegerAttr>(attribute);
  if (!integer)
    return std::nullopt;

  int64_t value = integer.getInt();
  if (value < 0 ||
      static_cast<uint64_t>(value) >
          static_cast<uint64_t>(std::numeric_limits<unsigned>::max()))
    return std::nullopt;
  return static_cast<unsigned>(value);
}

static DictionaryAttr makeNodeAttribute(MLIRContext *context, unsigned rootId,
                                        StringRef stage) {
  SmallVector<NamedAttribute, 2> fields;
  fields.push_back(
      NamedAttribute(kQGEMMNodeRootField,
                     IntegerAttr::get(IntegerType::get(context, 64), rootId)));
  fields.push_back(
      NamedAttribute(kQGEMMNodeStageField, StringAttr::get(context, stage)));
  return DictionaryAttr::get(context, fields);
}

static DictionaryAttr makeRootAttribute(MLIRContext *context, unsigned mDim,
                                        unsigned nDim) {
  SmallVector<NamedAttribute, 2> fields;
  fields.push_back(
      NamedAttribute(kQGEMMRootMDimField,
                     IntegerAttr::get(IntegerType::get(context, 64), mDim)));
  fields.push_back(
      NamedAttribute(kQGEMMRootNDimField,
                     IntegerAttr::get(IntegerType::get(context, 64), nDim)));
  return DictionaryAttr::get(context, fields);
}

static void validateRootId(unsigned rootId) {
  if (static_cast<uint64_t>(rootId) >
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    llvm_unreachable("QGEMM root identity does not fit in i64");
}

} // namespace

void annotate(const Recognition &recognition, unsigned rootId) {
  validateRootId(rootId);
  if (!recognition.finalReduce)
    llvm_unreachable("accepted QGEMM recognition has no final reduction");

  for (Operation *operation : recognition.ops) {
    if (!operation || recognition.boundaries.contains(operation))
      continue;

    auto stage = recognition.stages.find(operation);
    if (stage == recognition.stages.end())
      llvm_unreachable("accepted QGEMM operation has no stage");
    if (stage->second != kQGEMMDataPrepStage &&
        stage->second != kQGEMMComputeStage)
      llvm_unreachable("accepted QGEMM operation has an invalid stage");

    // setAttr replaces the operation's existing value for this owned
    // annotation key.  It therefore makes a second recognition run a stable
    // print/parse/print operation without touching any unrelated attribute.
    operation->setAttr(
        kQGEMMNodeAttr,
        makeNodeAttribute(operation->getContext(), rootId, stage->second));
  }

  Operation *finalReduce =
      const_cast<linalg::ReduceOp &>(recognition.finalReduce).getOperation();
  if (!recognition.ops.contains(finalReduce) ||
      recognition.boundaries.contains(finalReduce))
    llvm_unreachable("final QGEMM reduction is outside the owned scope");

  finalReduce->setAttr(kQGEMMRootAttr,
                       makeRootAttribute(finalReduce->getContext(),
                                         recognition.mDim, recognition.nDim));
}

std::optional<unsigned> getQGEMMNodeRoot(Operation *operation) {
  return getNonNegativeUnsigned(
      getQGEMMField(operation, kQGEMMNodeAttr, kQGEMMNodeRootField));
}

std::optional<std::pair<unsigned, unsigned>>
getQGEMMRootDims(Operation *operation) {
  auto mDim = getNonNegativeUnsigned(
      getQGEMMField(operation, kQGEMMRootAttr, kQGEMMRootMDimField));
  auto nDim = getNonNegativeUnsigned(
      getQGEMMField(operation, kQGEMMRootAttr, kQGEMMRootNDimField));
  if (!mDim || !nDim)
    return std::nullopt;
  return std::make_pair(*mDim, *nDim);
}

} // namespace buddy
} // namespace mlir
