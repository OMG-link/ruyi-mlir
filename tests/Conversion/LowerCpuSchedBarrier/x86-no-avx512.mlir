// RUN: buddy-opt %s --set-llvm-target="triple=x86_64-unknown-linux-gnu chip=haswell" --lower-cpu-sched-barrier | FileCheck %s

func.func @ymm_value(%arg0: vector<8xf32>) -> vector<8xf32> {
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "", "=x,0"
  %0 = cpu.sched_barrier %arg0 : vector<8xf32>
  return %0 : vector<8xf32>
}
