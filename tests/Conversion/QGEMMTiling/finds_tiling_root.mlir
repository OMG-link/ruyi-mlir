// RUN: buddy-opt --qgemm-tiling="tile-m=32 tile-n=32" %s 2>&1 | FileCheck %s

// Verify that when materialize_in_destination is present with a TilingInterface
// source, the pass finds the tiling root (no bail-out remark).

// CHECK-NOT: remark{{.*}}no materialize_in_destination found
// CHECK-LABEL: func.func @has_store

func.func @has_store(%arg0: tensor<64x32xf32>, %out: memref<64x32xf32>) {
  %cst = arith.constant 0.0 : f32
  %init = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<64x32xf32>) -> tensor<64x32xf32>
  bufferization.materialize_in_destination %init in writable %out : (tensor<64x32xf32>, memref<64x32xf32>) -> ()
  return
}
