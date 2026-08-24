// RUN: buddy-opt --qgemm-tiling="tile-m=4 tile-n=4 explain" %s 2>&1 | FileCheck %s
//
// A malformed hand-written root annotation is not an invitation to rerun
// recognition. The root has dimensions but no qgemm.node identity, so tiling
// skips it silently and leaves the input unchanged.

// CHECK-NOT: remark
// CHECK-LABEL: func.func @malformed_root
// CHECK-NOT: scf.for
// CHECK: linalg.reduce
// CHECK-SAME: qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}

func.func @malformed_root(%input: tensor<8x8x4xf32>) -> tensor<8x8xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<8x8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  %root = linalg.reduce ins(%input : tensor<8x8x4xf32>)
                           outs(%init : tensor<8x8xf32>)
                           dimensions = [2]
                           {qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}}
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %root : tensor<8x8xf32>
}
