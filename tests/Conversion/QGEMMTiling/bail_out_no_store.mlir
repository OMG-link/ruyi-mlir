// RUN: buddy-opt --qgemm-tiling="tile-m=32 tile-n=32" %s 2>&1 | FileCheck %s

// Verify that when no bufferization.materialize_in_destination op is present,
// the pass emits a remark and leaves the IR unchanged.

// CHECK: remark{{.*}}no materialize_in_destination found; skipping qgemm-tiling
// CHECK-LABEL: func.func @no_store
// CHECK:         linalg.generic
// CHECK:         return

func.func @no_store(%arg0: tensor<64x32xf32>, %arg1: tensor<64x32xf32>) -> tensor<64x32xf32> {
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<64x32xf32>) outs(%arg1 : tensor<64x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.addf %in, %out : f32
    linalg.yield %1 : f32
  } -> tensor<64x32xf32>
  return %0 : tensor<64x32xf32>
}
