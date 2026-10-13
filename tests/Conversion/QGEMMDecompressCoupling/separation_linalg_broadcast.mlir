// RUN: buddy-opt --qgemm-decompress-coupling=decompress=pre %s 2>&1 | FileCheck %s --check-prefix=PRE
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=on-the-fly %s 2>&1 | FileCheck %s --check-prefix=OTF
//
// Single-dimension linalg.broadcast inputs: each operand enters the other
// parallel dimension through a named broadcast. The pass must accept the named
// form and rebuild it as a generic pure broadcast, so the output equals that of
// separation_multiply_map.mlir.
//
// The input carries a hand-written annotation and skips
// --qgemm-simplify-dim-transforms --qgemm-recognize: the recognizer has no
// mapping for linalg.broadcast, so it never puts one inside a scope and the
// multiply's operands stay unannotated.
// PRE-NOT: remark
// OTF-NOT: remark
// PRE-LABEL: func.func @linalg_broadcast
// OTF-LABEL: func.func @linalg_broadcast
// PRE-NOT: linalg.broadcast
// OTF-NOT: linalg.broadcast
// PRE: linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<4x2x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<4x2x32xi32>) {{.*}}stage = "data_prep"
// PRE-NEXT: ^bb0
// PRE-NEXT: arith.extsi
// PRE: linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<4x2x32xi32>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi32>) {{.*}}stage = "compute"
// PRE-NEXT: ^bb0
// PRE-NEXT: linalg.yield
// PRE: linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<8x2x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<8x2x32xi32>) {{.*}}stage = "data_prep"
// PRE-NEXT: ^bb0
// PRE-NEXT: arith.extsi
// PRE: linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<8x2x32xi32>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi32>) {{.*}}stage = "compute"
// PRE-NEXT: ^bb0
// PRE-NEXT: linalg.yield
// OTF: linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<4x2x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi8>) {{.*}}stage = "compute"
// OTF-NEXT: ^bb0
// OTF-NEXT: linalg.yield
// OTF: linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi32>) {{.*}}stage = "compute"
// OTF-NEXT: ^bb0
// OTF-NEXT: arith.extsi
// OTF: linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<8x2x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi8>) {{.*}}stage = "compute"
// OTF-NEXT: ^bb0
// OTF-NEXT: linalg.yield
// OTF: linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi32>) {{.*}}stage = "compute"
// OTF-NEXT: ^bb0
// OTF-NEXT: arith.extsi
// PRE: linalg.reduce {{.*}} dimensions = [2] {{.*}}qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// PRE-NOT: remark
// OTF: linalg.reduce {{.*}} dimensions = [2] {{.*}}qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// OTF-NOT: remark
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d2)>
module {
  func.func @linalg_broadcast(%arg0: tensor<4x64xi8>, %arg1: tensor<8x64xi8>, %arg2: tensor<4x2xf32>, %arg3: tensor<8x2xf32>) -> tensor<4x8xf32> {
    %expanded = tensor.expand_shape %arg0 [[0], [1, 2]] output_shape [4, 2, 32] {qgemm.node = {root = 0 : i64, stage = "data_prep"}} : tensor<4x64xi8> into tensor<4x2x32xi8>
    %0 = tensor.empty() : tensor<4x2x32xi32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded : tensor<4x2x32xi8>) outs(%0 : tensor<4x2x32xi32>) attrs = {qgemm.node = {root = 0 : i64, stage = "data_prep"}} {
    ^bb0(%in: i8, %out: i32):
      %12 = arith.extsi %in : i8 to i32
      linalg.yield %12 : i32
    } -> tensor<4x2x32xi32>
    %expanded_0 = tensor.expand_shape %arg1 [[0], [1, 2]] output_shape [8, 2, 32] {qgemm.node = {root = 0 : i64, stage = "data_prep"}} : tensor<8x64xi8> into tensor<8x2x32xi8>
    %2 = tensor.empty() : tensor<8x2x32xi32>
    %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_0 : tensor<8x2x32xi8>) outs(%2 : tensor<8x2x32xi32>) attrs = {qgemm.node = {root = 0 : i64, stage = "data_prep"}} {
    ^bb0(%in: i8, %out: i32):
      %12 = arith.extsi %in : i8 to i32
      linalg.yield %12 : i32
    } -> tensor<8x2x32xi32>
    %4 = tensor.empty() : tensor<4x8x2x32xi32>
    %broadcasted = linalg.broadcast ins(%1 : tensor<4x2x32xi32>) outs(%4 : tensor<4x8x2x32xi32>) dimensions = [1] {qgemm.node = {root = 0 : i64, stage = "compute"}}
    %broadcasted_1 = linalg.broadcast ins(%3 : tensor<8x2x32xi32>) outs(%4 : tensor<4x8x2x32xi32>) dimensions = [0] {qgemm.node = {root = 0 : i64, stage = "compute"}}
    %5 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%broadcasted, %broadcasted_1 : tensor<4x8x2x32xi32>, tensor<4x8x2x32xi32>) outs(%4 : tensor<4x8x2x32xi32>) attrs =  {qgemm.node = {root = 0 : i64, stage = "compute"}} {
    ^bb0(%in: i32, %in_3: i32, %out: i32):
      %12 = arith.muli %in, %in_3 : i32
      linalg.yield %12 : i32
    } -> tensor<4x8x2x32xi32>
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %6 = tensor.empty() : tensor<4x8x2xi32>
    %7 = linalg.fill ins(%c0_i32 : i32) outs(%6 : tensor<4x8x2xi32>) -> tensor<4x8x2xi32>
    %reduced = linalg.reduce ins(%5 : tensor<4x8x2x32xi32>) outs(%7 : tensor<4x8x2xi32>) dimensions = [3]  {qgemm.node = {root = 0 : i64, stage = "compute"}}
      (%in: i32, %init: i32) {
        %12 = arith.addi %in, %init : i32
        linalg.yield %12 : i32
      }
    %8 = tensor.empty() : tensor<4x8x2xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map2, #map3, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%reduced, %arg2, %arg3 : tensor<4x8x2xi32>, tensor<4x2xf32>, tensor<8x2xf32>) outs(%8 : tensor<4x8x2xf32>) attrs =  {qgemm.node = {root = 0 : i64, stage = "compute"}} {
    ^bb0(%in: i32, %in_3: f32, %in_4: f32, %out: f32):
      %12 = arith.sitofp %in : i32 to f32
      %13 = arith.mulf %in_3, %in_4 : f32
      %14 = arith.mulf %12, %13 : f32
      linalg.yield %14 : f32
    } -> tensor<4x8x2xf32>
    %10 = tensor.empty() : tensor<4x8xf32>
    %11 = linalg.fill ins(%cst : f32) outs(%10 : tensor<4x8xf32>) -> tensor<4x8xf32>
    %reduced_2 = linalg.reduce ins(%9 : tensor<4x8x2xf32>) outs(%11 : tensor<4x8xf32>) dimensions = [2]  {qgemm.node = {root = 0 : i64, stage = "compute"}, qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}}
      (%in: f32, %init: f32) {
        %12 = arith.addf %in, %init : f32
        linalg.yield %12 : f32
      }
    return %reduced_2 : tensor<4x8xf32>
  }
}

