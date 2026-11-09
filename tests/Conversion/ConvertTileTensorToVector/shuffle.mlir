// RUN: buddy-opt %s -set-llvm-target="chip=spacemit-a100 vlen=1024" -convert-tile-tensor-to-vector -split-input-file | FileCheck %s
//
// A cpu.shuffle whose tiles fit in registers becomes vector.shuffle with the
// same mask; one that stays in tensors becomes a linalg.generic reading each
// element with tensor.extract.

// A slide down by one element that shifts in a zero (see
// shuffle-slide-asm.mlir for the instruction it becomes).
// CHECK-LABEL: func.func @slide
// CHECK: %[[X:.+]] = vector.transfer_read
// CHECK: vector.shuffle %[[X]], %{{.+}} [1, 2, 3, {{.*}}, 63, 64] : vector<64xf16>, vector<64xf16>
// CHECK-NOT: cpu.shuffle
func.func @slide(%in: memref<64xf16>, %out: memref<64xf16>) {
  %x = bufferization.to_tensor %in restrict : memref<64xf16> to tensor<64xf16>
  %zero = arith.constant dense<0.0> : tensor<64xf16>
  %y = cpu.shuffle %x, %zero [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64] : tensor<64xf16>, tensor<64xf16> -> tensor<64xf16>
  bufferization.materialize_in_destination %y in writable %out : (tensor<64xf16>, memref<64xf16>) -> ()
  return
}

// -----

// Operands of different first dimensions, on rank 2: the mask picks rows.
// CHECK-LABEL: func.func @rows
// CHECK: vector.shuffle %{{.+}}, %{{.+}} [5, 0, 4, 3] : vector<4x8xf32>, vector<2x8xf32>
func.func @rows(%a: memref<4x8xf32>, %b: memref<2x8xf32>, %out: memref<4x8xf32>) {
  %x = bufferization.to_tensor %a restrict : memref<4x8xf32> to tensor<4x8xf32>
  %y = bufferization.to_tensor %b restrict : memref<2x8xf32> to tensor<2x8xf32>
  %r = cpu.shuffle %x, %y [5, 0, 4, 3] : tensor<4x8xf32>, tensor<2x8xf32> -> tensor<4x8xf32>
  bufferization.materialize_in_destination %r in writable %out : (tensor<4x8xf32>, memref<4x8xf32>) -> ()
  return
}

// -----

// Too many elements for one register group: the shuffle stays in tensors.
// CHECK-LABEL: func.func @too_large
// CHECK: %[[M:.+]] = arith.constant dense<[3, -1, 0, 4096]> : tensor<4xindex>
// CHECK: linalg.generic
// CHECK: tensor.extract %[[M]]
// CHECK: arith.maxsi
// CHECK-COUNT-2: tensor.extract %arg
// CHECK: arith.select
// CHECK-NOT: cpu.shuffle
func.func @too_large(%a: tensor<4096xf32>, %b: tensor<4096xf32>) -> tensor<4xf32> {
  %r = cpu.shuffle %a, %b [3, -1, 0, 4096] : tensor<4096xf32>, tensor<4096xf32> -> tensor<4xf32>
  return %r : tensor<4xf32>
}
