// RUN: buddy-opt %s > %t.default
// RUN: buddy-opt --qgemm-tiling="tile-m=32 tile-n=32" %s > %t.tiled
// RUN: cmp -s %t.default %t.tiled
//
// A store and an unannotated initializer do not make the tiler invent a root.

func.func @has_store(%arg0: tensor<64x32xf32>, %out: memref<64x32xf32>) {
  %cst = arith.constant 0.0 : f32
  %init = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<64x32xf32>) -> tensor<64x32xf32>
  bufferization.materialize_in_destination %init in writable %out
      : (tensor<64x32xf32>, memref<64x32xf32>) -> ()
  return
}
