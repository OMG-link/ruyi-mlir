//===- CpuDialect.cpp - CPU ops dialect -----------------------------===//
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

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "Dialect/Cpu/CpuDialect.h"
#include "Dialect/Cpu/CpuOps.h"

using namespace mlir;
using namespace buddy::cpu;

#include "Cpu/CpuDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Cpu/Cpu.cpp.inc"

void CpuDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Cpu/Cpu.cpp.inc"
      >();
}

LogicalResult SchedBarrierOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands, DictionaryAttr,
    PropertyRef, RegionRange, SmallVectorImpl<Type> &inferredReturnTypes) {
  llvm::append_range(inferredReturnTypes, operands.getTypes());
  return success();
}

LogicalResult SchedBarrierOp::verify() {
  if (getInputs().empty())
    return emitOpError("needs at least one operand");
  return success();
}

void SchedBarrierOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Write::get(),
                       SchedBarrierResource::get());
  if (getMemoryAttr()) {
    effects.emplace_back(MemoryEffects::Read::get(),
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(),
                         SideEffects::DefaultResource::get());
  }
}
