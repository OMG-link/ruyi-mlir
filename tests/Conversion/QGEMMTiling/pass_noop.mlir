// RUN: buddy-opt %s > %t.default
// RUN: buddy-opt --qgemm-tiling %s > %t.tiled
// RUN: cmp -s %t.default %t.tiled
//
// A valid function without qgemm.root is unchanged by the tiling pass.

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
