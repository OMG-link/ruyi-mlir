// RUN: buddy-opt %s -lower-ime -convert-arith-to-llvm \
// RUN:   -convert-func-to-llvm=use-bare-ptr-memref-call-conv=1 \
// RUN:   -finalize-memref-to-llvm -reconcile-unrealized-casts \
// RUN:   | FileCheck %s --check-prefix=MLIR
// RUN: buddy-opt %s -lower-ime -convert-arith-to-llvm \
// RUN:   -convert-func-to-llvm=use-bare-ptr-memref-call-conv=1 \
// RUN:   -finalize-memref-to-llvm -reconcile-unrealized-casts \
// RUN:   | buddy-translate -buddy-to-llvmir \
// RUN:   | buddy-llc -mtriple=riscv64 -mattr=+m,+v,+xsmtvdot -verify-machineinstrs \
// RUN:   | FileCheck %s --check-prefix=ASM
//
// K1 int8 vmadot on memref tiles: each 4x8 int8 operand is one e8/m1
// register (32 elements) and the 4x4 int32 accumulator one e32/m2 register
// pair (16 elements), so every load and store moves exactly one tile.

// MLIR-LABEL: llvm.func @k1_vmadot
// MLIR-DAG: llvm.mlir.constant(32 : i64)
// MLIR-DAG: llvm.mlir.constant(16 : i64)
// MLIR: llvm.call @llvm.riscv.vle.nxv8i8.p0.i64
// MLIR: llvm.call @llvm.riscv.vle.nxv8i8.p0.i64
// MLIR: llvm.call @llvm.riscv.vle.nxv4i32.p0.i64
// MLIR: llvm.call @llvm.riscv.ime.vmadot.nxv4i32.nxv8i8.nxv8i8
// MLIR: llvm.call @llvm.riscv.vse.nxv4i32.p0.i64
// ASM-LABEL: k1_vmadot:
// ASM-DAG: vsetivli zero, 16, e32, m2
// ASM-DAG: vle32.v [[C:v[0-9]+]], (a0)
// ASM-DAG: vsetvli zero, {{a[0-9]+}}, e8, m1
// ASM-DAG: vle8.v [[A:v[0-9]+]], (a1)
// ASM-DAG: vle8.v [[B:v[0-9]+]], (a2)
// ASM: vsetvli {{.*}}, zero, e8, m1
// ASM-NEXT: smt.vmadot [[C]], [[A]], [[B]]
// ASM-NEXT: vsetivli zero, 16, e32, m2
// ASM-NEXT: vse32.v [[C]], (a0)
func.func @k1_vmadot(%c: memref<4x4xi32>, %a: memref<4x8xi8>, %b: memref<4x8xi8>) {
  ime.vmadot %c, %a, %b : memref<4x4xi32>, memref<4x8xi8>, memref<4x8xi8>
  return
}

// MLIR-LABEL: llvm.func @k1_vmadotu
// MLIR: llvm.call @llvm.riscv.ime.vmadotu.nxv4i32.nxv8i8.nxv8i8
// ASM-LABEL: k1_vmadotu:
// ASM: smt.vmadotu
func.func @k1_vmadotu(%c: memref<4x4xi32>, %a: memref<4x8xui8>, %b: memref<4x8xui8>) {
  ime.vmadotu %c, %a, %b : memref<4x4xi32>, memref<4x8xui8>, memref<4x8xui8>
  return
}

// MLIR-LABEL: llvm.func @k1_vmadotsu
// MLIR: llvm.call @llvm.riscv.ime.vmadotsu.nxv4i32.nxv8i8.nxv8i8
// ASM-LABEL: k1_vmadotsu:
// ASM: smt.vmadotsu
func.func @k1_vmadotsu(%c: memref<4x4xi32>, %a: memref<4x8xi8>, %b: memref<4x8xui8>) {
  ime.vmadotsu %c, %a, %b : memref<4x4xi32>, memref<4x8xi8>, memref<4x8xui8>
  return
}

// MLIR-LABEL: llvm.func @k1_vmadotus
// MLIR: llvm.call @llvm.riscv.ime.vmadotus.nxv4i32.nxv8i8.nxv8i8
// ASM-LABEL: k1_vmadotus:
// ASM: smt.vmadotus
func.func @k1_vmadotus(%c: memref<4x4xi32>, %a: memref<4x8xui8>, %b: memref<4x8xi8>) {
  ime.vmadotus %c, %a, %b : memref<4x4xi32>, memref<4x8xui8>, memref<4x8xi8>
  return
}
