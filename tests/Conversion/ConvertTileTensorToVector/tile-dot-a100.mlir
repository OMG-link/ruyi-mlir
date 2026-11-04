// RUN: buddy-opt %s -set-llvm-target="chip=spacemit-a100 vlen=1024" -convert-tile-tensor-to-vector=allow-ime-tile-contracts -canonicalize | FileCheck %s
//
// On the A100, the widening fp16 dot Triton writes — both operands extended
// to fp32 by a generic, B after its transpose — converts to vector.transpose,
// the extensions and a vector.contract on fp32.

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @fp16
// CHECK: %[[A:.+]] = vector.transfer_read {{.*}} vector<8x8xf16>
// CHECK: %[[B:.+]] = vector.transfer_read {{.*}} vector<8x8xf16>
// CHECK: %[[BT:.+]] = vector.transpose %[[B]], [1, 0]
// CHECK-DAG: arith.extf %[[A]] : vector<8x8xf16> to vector<8x8xf32>
// CHECK-DAG: arith.extf %[[BT]] : vector<8x8xf16> to vector<8x8xf32>
// CHECK: vector.contract {{.*}} : vector<8x8xf32>, vector<8x8xf32> into vector<8x8xf32>
// CHECK-NOT: linalg.matmul
// CHECK-NOT: linalg.transpose
func.func @fp16(%a: memref<8x8xf16>, %b: memref<8x8xf16>, %c: memref<8x8xf32>) {
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<8x8xf32>
  %z = linalg.fill ins(%zero : f32) outs(%e : tensor<8x8xf32>) -> tensor<8x8xf32>
  %ta = bufferization.to_tensor %a restrict : memref<8x8xf16> to tensor<8x8xf16>
  %tb = bufferization.to_tensor %b restrict : memref<8x8xf16> to tensor<8x8xf16>
  %eh = tensor.empty() : tensor<8x8xf16>
  %bt = linalg.transpose ins(%tb : tensor<8x8xf16>) outs(%eh : tensor<8x8xf16>) permutation = [1, 0]
  %ea = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%ta : tensor<8x8xf16>) outs(%e : tensor<8x8xf32>) {
  ^bb0(%in: f16, %out: f32):
    %x = arith.extf %in : f16 to f32
    linalg.yield %x : f32
  } -> tensor<8x8xf32>
  %eb = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%bt : tensor<8x8xf16>) outs(%e : tensor<8x8xf32>) {
  ^bb0(%in: f16, %out: f32):
    %x = arith.extf %in : f16 to f32
    linalg.yield %x : f32
  } -> tensor<8x8xf32>
  %m = linalg.matmul ins(%ea, %eb : tensor<8x8xf32>, tensor<8x8xf32>) outs(%z : tensor<8x8xf32>) -> tensor<8x8xf32>
  bufferization.materialize_in_destination %m in writable %c : (tensor<8x8xf32>, memref<8x8xf32>) -> ()
  return
}
