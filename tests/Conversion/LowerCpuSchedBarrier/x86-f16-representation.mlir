// RUN: buddy-opt %s -set-llvm-target="triple=x86_64-unknown-linux-gnu chip=x86-64" -lower-cpu-sched-barrier | FileCheck %s

// A 128-bit f16 value crosses the x86 barrier as an equally sized i16 vector.
// CHECK: vector.bitcast {{.*}} : vector<8xf16> to vector<8xi16>
// CHECK: llvm.inline_asm has_side_effects asm_dialect = att "", "=x,0" {{.*}} : (vector<8xi16>) -> vector<8xi16>
// CHECK: vector.bitcast {{.*}} : vector<8xi16> to vector<8xf16>
func.func @x86_half(%a: vector<8xf16>) -> vector<8xf16> {
  %r = cpu.sched_barrier %a : vector<8xf16>
  return %r : vector<8xf16>
}
