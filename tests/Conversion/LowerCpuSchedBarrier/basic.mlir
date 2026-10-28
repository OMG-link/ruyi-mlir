// RUN: buddy-opt %s -set-llvm-target="triple=x86_64-unknown-linux-gnu chip=icelake-server" -lower-cpu-sched-barrier -split-input-file -verify-diagnostics | FileCheck %s --check-prefix=X86
//
// cpu.sched_barrier becomes an empty inline asm with side effects that holds
// every operand in a vector register, with memory clobber opted in.
// On x86 with AVX-512 a vector<16xf32> is one ZMM register.

// X86-LABEL: func.func @two
// X86: llvm.inline_asm has_side_effects asm_dialect = att "", "=v,=v,0,1,~{memory}" {{.*}} : (vector<16xf32>, vector<32xi8>) -> !llvm.struct<(vector<16xf32>, vector<32xi8>)>
func.func @two(%a: vector<16xf32>, %b: vector<32xi8>) -> (vector<16xf32>, vector<32xi8>) {
  %r:2 = cpu.sched_barrier %a, %b : vector<16xf32>, vector<32xi8> <{memory}>
  return %r#0, %r#1 : vector<16xf32>, vector<32xi8>
}

// -----

// A 128-bit value fits one XMM register.
// X86-LABEL: func.func @half
// X86: "=v,0,~{memory}" {{.*}} : (vector<4xi32>) -> vector<4xi32>
func.func @half(%a: vector<4xi32>) -> vector<4xi32> {
  %r = cpu.sched_barrier %a : vector<4xi32> <{memory}>
  return %r : vector<4xi32>
}

// -----

// Without the optional memory attribute, only the SSA values are constrained.
// X86-LABEL: func.func @value_only
// X86: llvm.inline_asm has_side_effects asm_dialect = att "", "=v,0" {{.*}}
func.func @value_only(%a: vector<16xf32>) -> vector<16xf32> {
  %r = cpu.sched_barrier %a : vector<16xf32>
  return %r : vector<16xf32>
}

// -----

// 12 x i32 fills no register group.
func.func @ragged(%a: vector<12xi32>) -> vector<12xi32> {
  // expected-error @+1 {{cannot hold operand in registers}}
  %r = cpu.sched_barrier %a : vector<12xi32>
  return %r : vector<12xi32>
}
