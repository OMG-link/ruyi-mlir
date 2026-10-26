// RUN: buddy-opt %s -split-input-file -test-target-info -verify-diagnostics | FileCheck %s
//
// The target comes from the module's llvm.target attribute: the chip's
// extensions from LLVM's processor table, then the listed features on top.

// The SpacemiT K1 core: VLEN 256 and the IME dot extension.
// CHECK-LABEL: triple: riscv64-unknown-linux-gnu
// CHECK-NEXT: chip: spacemit-x60
// CHECK-NEXT: xlen: 64
// CHECK-NEXT: vector: yes
// CHECK-NEXT: vlen: 256
// CHECK-NEXT: elen: 64
// CHECK-NEXT: elen-fp: 64
// CHECK-NEXT: isa: {{.*}}_zvfh1p0{{.*}}_zvl256b1p0{{.*}}_xsmtvdot1p0
// CHECK-NEXT: elements i8: m1=32 m8=256
// CHECK-NEXT: elements i32: m1=8 m8=64
// CHECK-NEXT: elements i64: m1=4 m8=32
// CHECK-NEXT: elements f16: m1=16 m8=128
// CHECK-NEXT: elements f32: m1=8 m8=64
// CHECK-NEXT: elements bf16: m1=0 m8=0
module attributes {llvm.target = #llvm.target<triple = "riscv64-unknown-linux-gnu", chip = "spacemit-x60">} {
}

// -----

// The SpacemiT A100 (K3): VLEN 1024 and the second IME generation.
// CHECK-LABEL: chip: spacemit-a100
// CHECK: vlen: 1024
// CHECK: isa: {{.*}}_xsmtvdotii1p0
// CHECK: elements i8: m1=128 m8=1024
module attributes {llvm.target = #llvm.target<triple = "riscv64-unknown-linux-gnu", chip = "spacemit-a100">} {
}

// -----

// Features alone describe a target without a named chip, and a feature can
// remove an extension the chip has.
// CHECK-LABEL: chip: generic
// CHECK: vlen: 512
// CHECK: elements f16: m1=0 m8=0
module attributes {llvm.target = #llvm.target<triple = "riscv64-unknown-linux-gnu", chip = "generic", features = #llvm.target_features<["+m", "+f", "+d", "+v", "+zvl512b"]>>} {
}

// -----

// CHECK-LABEL: chip: spacemit-x60
// CHECK: isa:
// CHECK-NOT: xsmtvdot
// CHECK-SAME: {{$}}
// CHECK: elements f16: m1=0 m8=0
module attributes {llvm.target = #llvm.target<triple = "riscv64-unknown-linux-gnu", chip = "spacemit-x60", features = #llvm.target_features<["-xsmtvdot", "-zvfh"]>>} {
}

// -----

// A scalar-only target has no vector element types.
// CHECK-LABEL: chip: generic
// CHECK: vector: no
// CHECK: elements i8: m1=0 m8=0
module attributes {llvm.target = #llvm.target<triple = "riscv64-unknown-linux-gnu", chip = "generic", features = #llvm.target_features<["+m", "+a"]>>} {
}

// -----

// expected-error @below {{no 'llvm.target' attribute on the enclosing module}}
module {
}

// -----

// expected-error @below {{unknown RISC-V processor 'no-such-core'}}
module attributes {llvm.target = #llvm.target<triple = "riscv64-unknown-linux-gnu", chip = "no-such-core">} {
}

// -----

// expected-error @below {{unknown RISC-V extension 'xnosuchext'}}
module attributes {llvm.target = #llvm.target<triple = "riscv64-unknown-linux-gnu", chip = "spacemit-x60", features = #llvm.target_features<["+xnosuchext"]>>} {
}

// -----

// expected-error @below {{target triple 'x86_64-unknown-linux-gnu' is not RISC-V}}
module attributes {llvm.target = #llvm.target<triple = "x86_64-unknown-linux-gnu", chip = "native">} {
}
