// RUN: buddy-opt %s -set-llvm-target="triple=riscv64-unknown-linux-gnu chip=generic features=+v,+zvl256b,-zvfh,-zvfhmin vlen=256" -lower-cpu-sched-barrier | FileCheck %s

// RVV can hold f16 bits without Zvfh. Use integer SEW=16 for the asm operand.
// CHECK: vector.bitcast {{.*}} : vector<16xf16> to vector<16xi16>
// CHECK: llvm.intr.vector.insert {{.*}} : vector<16xi16> into vector<[4]xi16>
// CHECK: llvm.inline_asm has_side_effects asm_dialect = att "", "=^vr,0" {{.*}} : (vector<[4]xi16>) -> vector<[4]xi16>
// CHECK: llvm.intr.vector.extract {{.*}} : vector<16xi16> from vector<[4]xi16>
// CHECK: vector.bitcast {{.*}} : vector<16xi16> to vector<16xf16>
func.func @rv_half(%a: vector<16xf16>) -> vector<16xf16> {
  %r = cpu.sched_barrier %a : vector<16xf16>
  return %r : vector<16xf16>
}
