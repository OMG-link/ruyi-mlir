// RUN: buddy-opt %s --set-llvm-target="chip=spacemit-x60 vlen=512" --test-target-info -o /dev/null | FileCheck %s
// RUN: not buddy-opt %s --set-llvm-target="chip=spacemit-x60 vlen=128" -o /dev/null 2>&1 | FileCheck %s --check-prefix=RANGE
// RUN: not buddy-opt %s --set-llvm-target="chip=spacemit-x60 vlen=384" -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not buddy-opt %s --set-llvm-target="triple=x86_64-unknown-linux-gnu chip=haswell vlen=256" -o /dev/null 2>&1 | FileCheck %s --check-prefix=X86
// CHECK: min-vlen: 512
// CHECK: max-vector-bits: 512
// CHECK-NEXT: exact-vector-bits: 512
// CHECK: elements i32: m1=16 m8=128
// RANGE: cpu.vlen conflicts with the target VLEN range
// INVALID: cpu.vlen must be zero or a power of two >= 128
// X86: vlen is only supported for RISC-V targets
module {}
