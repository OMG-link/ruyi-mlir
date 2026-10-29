// RUN: buddy-opt %s -set-llvm-target="triple=riscv64-unknown-linux-gnu chip=generic features=+zve32x,+zvl256b vlen=256" -lower-cpu-sched-barrier | FileCheck %s

// Integer-only Zve32x carries f32 data through an equally sized i32 vector.
// CHECK: vector.bitcast {{.*}} : vector<8xf32> to vector<8xi32>
// CHECK: llvm.intr.vector.insert {{.*}} : vector<8xi32> into vector<[2]xi32>
// CHECK: llvm.inline_asm has_side_effects asm_dialect = att "", "=^vr,0" {{.*}} : (vector<[2]xi32>) -> vector<[2]xi32>
// CHECK: llvm.intr.vector.extract {{.*}} : vector<8xi32> from vector<[2]xi32>
// CHECK: vector.bitcast {{.*}} : vector<8xi32> to vector<8xf32>
func.func @rv_float_bits(%a: vector<8xf32>) -> vector<8xf32> {
  %r = cpu.sched_barrier %a : vector<8xf32>
  return %r : vector<8xf32>
}
