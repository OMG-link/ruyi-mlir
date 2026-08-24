// RUN: buddy-opt %s > %t.default
// RUN: buddy-opt --qgemm-tiling="tile-m=32 tile-n=32" %s > %t.tiled
// RUN: cmp -s %t.default %t.tiled
//
// No qgemm.root annotation is present, so annotation-driven tiling is an
// exact no-op on this ordinary tensor generic.

func.func @no_store(%arg0: tensor<64x32xf32>, %arg1: tensor<64x32xf32>) -> tensor<64x32xf32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<64x32xf32>) outs(%arg1 : tensor<64x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %sum = arith.addf %in, %out : f32
    linalg.yield %sum : f32
  } -> tensor<64x32xf32>
  return %result : tensor<64x32xf32>
}
