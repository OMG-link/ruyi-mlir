// RUN: buddy-opt %s --set-llvm-target="triple=x86_64-unknown-linux-gnu chip=icelake-server" --convert-tile-tensor-to-vector --flatten-vector-tiles --lower-cpu-sched-barrier | FileCheck %s

func.func @memory_forwarding() {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<4x4xf32>
  %tile = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x4xf32>) -> tensor<4x4xf32>
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "", "=v,0,~{memory}"
  %0 = cpu.sched_barrier %tile : tensor<4x4xf32> <memory>
  return
}
