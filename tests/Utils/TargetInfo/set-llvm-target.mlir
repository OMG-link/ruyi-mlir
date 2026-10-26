// RUN: buddy-opt %s -set-llvm-target="chip=spacemit-x60 features=+zvl512b,-zvfh" | FileCheck %s
// RUN: buddy-opt %s -set-llvm-target="chip=spacemit-x60" -test-target-info -o /dev/null | FileCheck %s --check-prefix=INFO
// RUN: not buddy-opt %s -set-llvm-target="chip=no-such-core" -o /dev/null 2>&1 | FileCheck %s --check-prefix=BAD
//
// The driver states the target with -set-llvm-target; the pass writes the
// attribute and rejects targets LLVM does not know.

// CHECK: module attributes {llvm.target = #llvm.target<triple = "riscv64-unknown-linux-gnu", chip = "spacemit-x60", features = <["+zvl512b", "-zvfh"]>>}
// INFO: chip: spacemit-x60
// INFO: vlen: 256
// BAD: unknown RISC-V processor 'no-such-core'
module {
  func.func @f() {
    return
  }
}
