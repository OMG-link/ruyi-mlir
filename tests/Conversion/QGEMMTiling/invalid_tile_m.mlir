// RUN: buddy-opt %s > %t.default
// RUN: buddy-opt '--qgemm-tiling=tile-m=0 tile-n=0' %s > %t.zero
// RUN: cmp -s %t.default %t.zero
// RUN: not buddy-opt '--qgemm-tiling=tile-m=-1' %s 2>&1 | FileCheck %s --check-prefix=NEG-M
// RUN: not buddy-opt '--qgemm-tiling=tile-m=0 tile-n=-1' %s 2>&1 | FileCheck %s --check-prefix=NEG-N
//
// Both zero tile options are an exact no-op. Negative options remain hard
// errors and name the offending option and value.

// NEG-M: error: invalid tile-m value -1 (must be >= 0)
// NEG-N: error: invalid tile-n value -1 (must be >= 0)

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
