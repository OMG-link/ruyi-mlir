// RUN: buddy-opt %s -set-llvm-target="chip=spacemit-a100 vlen=1024" -convert-tile-tensor-to-vector -flatten-vector-tiles -canonicalize \
// RUN:   -expand-strided-metadata -convert-vector-to-llvm -convert-arith-to-llvm -convert-index-to-llvm -finalize-memref-to-llvm \
// RUN:   -convert-func-to-llvm -reconcile-unrealized-casts \
// RUN:   | buddy-translate -buddy-to-llvmir \
// RUN:   | buddy-llc -O3 -mtriple=riscv64-unknown-linux-gnu -mcpu=spacemit-a100 -mattr=+v,+zvfh -riscv-v-vector-bits-min=1024 -riscv-v-vector-bits-max=1024 \
// RUN:   | FileCheck %s
//
// A slide down by one element that shifts in a zero, the form the IME scale
// construction relies on, reaches RVV as one vfslide1down.vf with a zero
// scalar, not as a general gather.

// CHECK-LABEL: slide:
// CHECK: vfslide1down.vf
// CHECK-NOT: vrgather
// CHECK: ret
func.func @slide(%in: memref<64xf16>, %out: memref<64xf16>) {
  %x = bufferization.to_tensor %in restrict : memref<64xf16> to tensor<64xf16>
  %zero = arith.constant dense<0.0> : tensor<64xf16>
  %y = cpu.shuffle %x, %zero [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64] : tensor<64xf16>, tensor<64xf16> -> tensor<64xf16>
  bufferization.materialize_in_destination %y in writable %out : (tensor<64xf16>, memref<64xf16>) -> ()
  return
}
