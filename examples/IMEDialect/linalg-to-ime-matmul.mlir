// RUN: buddy-opt %s -lower-linalg-to-ime | FileCheck %s
// RUN: buddy-opt %s \
// RUN:   -lower-linalg-to-ime \
// RUN:   -lower-ime \
// RUN:   -convert-linalg-to-loops \
// RUN:   -lower-affine \
// RUN:   -convert-scf-to-cf \
// RUN:   -convert-cf-to-llvm \
// RUN:   -convert-arith-to-llvm \
// RUN:   -convert-math-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN:   -finalize-memref-to-llvm \
// RUN:   -reconcile-unrealized-casts | \
// RUN: buddy-translate -buddy-to-llvmir | \
// RUN: buddy-llc -filetype=asm -mtriple=riscv64 \
// RUN:   -mattr=+m,+v,+xsmtvdot -o - | FileCheck %s --check-prefix=ASM
//
// This file tests the lowering of linalg.matmul to ime.vmadot operations.
//

// Test case 1: Simple int8 matmul (4x8) * (8x4) = (4x4)
// This maps to ime.vmadot with single iteration loops
// CHECK-LABEL: func.func @matmul_i8_4x8x4
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       ime.vmadot
func.func @matmul_i8_4x8x4(%A: memref<4x8xi8>, %B: memref<8x4xi8>, %C: memref<4x4xi32>) {
  linalg.matmul ins(%A, %B : memref<4x8xi8>, memref<8x4xi8>)
                outs(%C : memref<4x4xi32>)
  return
}

// Test case 2: Larger int8 matmul requiring tiling
// (16x32) * (32x16) = (16x16)
// Should generate nested loops with ime.vmadot
// CHECK-LABEL: func.func @matmul_i8_16x32x16
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       ime.vmadot
func.func @matmul_i8_16x32x16(%A: memref<16x32xi8>, %B: memref<32x16xi8>, %C: memref<16x16xi32>) {
  linalg.matmul ins(%A, %B : memref<16x32xi8>, memref<32x16xi8>)
                outs(%C : memref<16x16xi32>)
  return
}

// ASM: .attribute 5, "{{.*xsmtvdot.*}}"
// ASM: vmadot{{[ \t]}}
