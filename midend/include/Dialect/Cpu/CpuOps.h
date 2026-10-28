//===- CpuOps.h - CPU ops -------------------------------------------===//
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

#ifndef CPU_CPUOPS_H
#define CPU_CPUOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/StringRef.h"

namespace buddy::cpu {
struct SchedBarrierResource
    : public mlir::SideEffects::Resource::Base<SchedBarrierResource> {
  llvm::StringRef getName() const final { return "SchedBarrierResource"; }
};
} // namespace buddy::cpu

#define GET_OP_CLASSES
#include "Cpu/Cpu.h.inc"

#endif // CPU_CPUOPS_H
