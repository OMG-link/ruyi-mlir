// RUN: buddy-opt '--qgemm-tiling=tile-m=0' %s | FileCheck %s --check-prefix=ZERO
// RUN: not buddy-opt '--qgemm-tiling=tile-m=-1' %s 2>&1 | FileCheck %s --check-prefix=NEG

// Verify that tile-m=0 is valid (no tiling on M dimension).
// ZERO-LABEL: func.func @simple_add
// ZERO:         %[[RESULT:.*]] = linalg.generic
// ZERO:         return %[[RESULT]] : tensor<64x32xf32>

// Verify that a negative tile-m emits an error.
// NEG: error: invalid tile-m value -1 (must be >= 0)

func.func @simple_add(%arg0: tensor<64x32xf32>, %arg1: tensor<64x32xf32>) -> tensor<64x32xf32> {
  %empty = tensor.empty() : tensor<64x32xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<64x32xf32>, tensor<64x32xf32>)
    outs(%empty : tensor<64x32xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %add = arith.addf %in0, %in1 : f32
    linalg.yield %add : f32
  } -> tensor<64x32xf32>
  return %result : tensor<64x32xf32>
}
