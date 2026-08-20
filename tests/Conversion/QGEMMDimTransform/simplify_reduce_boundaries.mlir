// RUN: timeout 8 buddy-opt --qgemm-simplify-dim-transforms %s | FileCheck %s
// RUN: timeout 8 buddy-opt --qgemm-simplify-dim-transforms %S/../QGEMMRecognize/Inputs/q4k_q8k_canonical.ttshared.mlir | FileCheck %s --check-prefix=Q4K

// These fragments exercise both dimension-transform positions that share the
// reshape-fusion machinery. Expansion may rebuild an intermediate reduction as
// a reduction-iterator linalg.generic. The only reduction contract here is the
// final root result's rank-2 type: m_dim = 0 and n_dim = 1 are read from that
// root result's dimension order, rather than from an intermediate reduction
// operation or its spelling.
#identity2 = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @reduce_collapse(%arg: tensor<2x2x2x2xf32>) -> tensor<2x2xf32> {
    %collapsed = tensor.collapse_shape %arg [[0], [1], [2, 3]]
        : tensor<2x2x2x2xf32> into tensor<2x2x4xf32>
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<2x2xf32>
    %init = linalg.fill ins(%zero : f32)
        outs(%empty : tensor<2x2xf32>) -> tensor<2x2xf32>
    %reduced = linalg.reduce ins(%collapsed : tensor<2x2x4xf32>)
        outs(%init : tensor<2x2xf32>) dimensions = [2]
      (%in: f32, %acc: f32) {
        %sum = arith.addf %in, %acc : f32
        linalg.yield %sum : f32
      }
    return %reduced : tensor<2x2xf32>
  }

  func.func @expand_reduce(%arg: tensor<2x2x2xf32>) -> tensor<2x1xf32> {
    %zero = arith.constant 0.0 : f32
    %inner_empty = tensor.empty() : tensor<2x2xf32>
    %inner_init = linalg.fill ins(%zero : f32)
        outs(%inner_empty : tensor<2x2xf32>) -> tensor<2x2xf32>
    %inner = linalg.reduce ins(%arg : tensor<2x2x2xf32>)
        outs(%inner_init : tensor<2x2xf32>) dimensions = [2]
      (%in: f32, %acc: f32) {
        %sum = arith.addf %in, %acc : f32
        linalg.yield %sum : f32
      }
    %expanded = tensor.expand_shape %inner [[0], [1, 2]]
        output_shape [2, 1, 2]
        : tensor<2x2xf32> into tensor<2x1x2xf32>
    %outer_empty = tensor.empty() : tensor<2x1xf32>
    %outer_init = linalg.fill ins(%zero : f32)
        outs(%outer_empty : tensor<2x1xf32>) -> tensor<2x1xf32>
    %outer = linalg.reduce ins(%expanded : tensor<2x1x2xf32>)
        outs(%outer_init : tensor<2x1xf32>) dimensions = [2]
      (%in: f32, %acc: f32) {
        %sum = arith.addf %in, %acc : f32
        linalg.yield %sum : f32
      }
    return %outer : tensor<2x1xf32>
  }
}

// The collapse remains below the input boundary; only the root's rank-2 result
// is part of this test's contract.
// CHECK-LABEL: func.func @reduce_collapse
// CHECK: %[[RED0:[A-Za-z0-9_]+]] = linalg.reduce
// CHECK-NOT: tensor.expand_shape %[[RED0]]
// CHECK-NOT: tensor.collapse_shape %[[RED0]]
// CHECK-NOT: linalg.transpose ins(%[[RED0]]
// CHECK: return %{{.*}} : tensor<2x2xf32>

// The inner reduction may be rebuilt as a reduction-iterator generic. The
// outer root remains rank 2, which is the only shape fact consumed here.
// CHECK-LABEL: func.func @expand_reduce
// CHECK: %[[RED1:[A-Za-z0-9_]+]] = linalg.reduce
// CHECK-NOT: tensor.expand_shape %[[RED1]]
// CHECK-NOT: tensor.collapse_shape %[[RED1]]
// CHECK-NOT: linalg.transpose ins(%[[RED1]]
// CHECK: return %{{.*}} : tensor<2x1xf32>

// Q4_K has several intermediate reduction levels. The checks below cover
// every linalg.reduce present in the actual simplified output. Reduction
// levels rebuilt as reduction-iterator linalg.generic are intentionally not
// matched by this linalg.reduce-specific check; the final root is still the
// rank-2 tensor passed to the output boundary.
// Q4K-LABEL: func.func @q4k_q8k_canonical
// Q4K: %[[RED0:[A-Za-z0-9_]+]] = linalg.reduce
// Q4K-NOT: tensor.expand_shape %[[RED0]]
// Q4K-NOT: tensor.collapse_shape %[[RED0]]
// Q4K-NOT: linalg.transpose ins(%[[RED0]]
// Q4K: %[[RED1:[A-Za-z0-9_]+]] = linalg.reduce
// Q4K-NOT: tensor.expand_shape %[[RED1]]
// Q4K-NOT: tensor.collapse_shape %[[RED1]]
// Q4K-NOT: linalg.transpose ins(%[[RED1]]
// Q4K: %[[RED2:[A-Za-z0-9_]+]] = linalg.reduce
// Q4K-NOT: tensor.expand_shape %[[RED2]]
// Q4K-NOT: tensor.collapse_shape %[[RED2]]
// Q4K-NOT: linalg.transpose ins(%[[RED2]]
// Q4K: bufferization.materialize_in_destination %{{.*}} in writable %{{.*}} : (tensor<192x32xf32>,
