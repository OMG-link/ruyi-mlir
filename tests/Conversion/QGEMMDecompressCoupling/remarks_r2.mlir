// RUN: buddy-opt --qgemm-decompress-coupling=decompress=pre %s 2>&1 | FileCheck %s --check-prefixes=CHECK,PRE
// RUN: buddy-opt --qgemm-decompress-coupling=decompress=on-the-fly %s 2>&1 | FileCheck %s --check-prefixes=CHECK,OTF
//
// Property 13, R2 (Req 3.2, 7.6, 7.7). The activation is broadcast into 'n'
// early, then passes through a collapse_shape/expand_shape pair whose
// collapse groups 'n' with the non-unit block dim (8x2 -> 16). The collapse
// is invariant along 'n', but the collapsed dim's linear index depends on
// 'n', so it has no strip rule: it gets one R2 and is pinned in the 'n' space,
// keeping its 4x8x2x32 -> 4x16x32 type in both modes. The root is still
// rebuilt: the scale multiply is fed by pure broadcasts of the two scales.
//
// The input carries a hand-written annotation and skips
// --qgemm-simplify-dim-transforms --qgemm-recognize: the simplifier folds the
// collapse/expand pair away. The input is therefore not producible by
// simplify+recognize and the normal pipeline never shows this R2; the test
// exercises the rule-table path directly.
// CHECK-NOT: cannot be inserted
// CHECK: remark: qgemm-decompress-coupling: op is invariant along 'n' but 'n' cannot be stripped; pinned in the 'n' space
// CHECK-NEXT: tensor.collapse_shape %abc
// CHECK-NOT: remark:
// CHECK-LABEL: func.func @r2_grouped_reshape_nonunit
// CHECK-DAG: linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<4x2xf32>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2xf32>)
// CHECK-DAG: linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<8x2xf32>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2xf32>)
// PRE: linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<4x2x32xi32>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi32>) {{.*}}stage = "compute"
// OTF: linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<4x2x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi8>) {{.*}}stage = "compute"
// OTF: linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi32>) {{.*}}stage = "compute"
// The pinned op keeps 'n'.
// CHECK: tensor.collapse_shape %{{[a-z0-9_]+}} {{\[\[}}0], [1, 2], [3]] {{.*}}: tensor<4x8x2x32xi32> into tensor<4x16x32xi32>
// CHECK: tensor.expand_shape %{{[a-z0-9_]+}} {{\[\[}}0], [1, 2], [3]] output_shape [4, 8, 2, 32] {{.*}}: tensor<4x16x32xi32> into tensor<4x8x2x32xi32>
// PRE: linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<8x2x32xi32>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi32>)
// OTF: linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<8x2x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi8>)
// CHECK: arith.muli
// CHECK: qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// CHECK-NOT: remark:
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#a4 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#w4 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#a3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#w3 = affine_map<(d0, d1, d2) -> (d1, d2)>
func.func @r2_grouped_reshape_nonunit(%aq: tensor<4x64xi8>, %wq: tensor<8x64xi8>, %ad: tensor<4x2xf32>, %wd: tensor<8x2xf32>) -> tensor<4x8xf32> {
  %ae = tensor.expand_shape %aq [[0], [1, 2]] output_shape [4, 2, 32] {qgemm.node = {root = 0 : i64, stage = "data_prep"}} : tensor<4x64xi8> into tensor<4x2x32xi8>
  %ace = tensor.empty() : tensor<4x2x32xi32>
  %acast = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%ae : tensor<4x2x32xi8>) outs(%ace : tensor<4x2x32xi32>) attrs = {qgemm.node = {root = 0 : i64, stage = "data_prep"}} {
  ^bb0(%i: i8, %o: i32):
    %x = arith.extsi %i : i8 to i32
    linalg.yield %x : i32
  } -> tensor<4x2x32xi32>
  %abe = tensor.empty() : tensor<4x8x2x32xi32>
  %abc = linalg.generic {indexing_maps = [#a4, #id4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%acast : tensor<4x2x32xi32>) outs(%abe : tensor<4x8x2x32xi32>) attrs = {qgemm.node = {root = 0 : i64, stage = "compute"}} {
  ^bb0(%i: i32, %o: i32):
    linalg.yield %i : i32
  } -> tensor<4x8x2x32xi32>
  %grp = tensor.collapse_shape %abc [[0], [1, 2], [3]] {qgemm.node = {root = 0 : i64, stage = "compute"}} : tensor<4x8x2x32xi32> into tensor<4x16x32xi32>
  %ung = tensor.expand_shape %grp [[0], [1, 2], [3]] output_shape [4, 8, 2, 32] {qgemm.node = {root = 0 : i64, stage = "compute"}} : tensor<4x16x32xi32> into tensor<4x8x2x32xi32>
  %we = tensor.expand_shape %wq [[0], [1, 2]] output_shape [8, 2, 32] {qgemm.node = {root = 0 : i64, stage = "data_prep"}} : tensor<8x64xi8> into tensor<8x2x32xi8>
  %wce = tensor.empty() : tensor<8x2x32xi32>
  %wcast = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%we : tensor<8x2x32xi8>) outs(%wce : tensor<8x2x32xi32>) attrs = {qgemm.node = {root = 0 : i64, stage = "data_prep"}} {
  ^bb0(%i: i8, %o: i32):
    %x = arith.extsi %i : i8 to i32
    linalg.yield %x : i32
  } -> tensor<8x2x32xi32>
  %pe = tensor.empty() : tensor<4x8x2x32xi32>
  %prod = linalg.generic {indexing_maps = [#id4, #w4, #id4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%ung, %wcast : tensor<4x8x2x32xi32>, tensor<8x2x32xi32>) outs(%pe : tensor<4x8x2x32xi32>) attrs = {qgemm.node = {root = 0 : i64, stage = "compute"}} {
  ^bb0(%x: i32, %y: i32, %o: i32):
    %z = arith.muli %x, %y : i32
    linalg.yield %z : i32
  } -> tensor<4x8x2x32xi32>
  %c0 = arith.constant 0 : i32
  %f0 = arith.constant 0.0 : f32
  %re = tensor.empty() : tensor<4x8x2xi32>
  %ri = linalg.fill ins(%c0 : i32) outs(%re : tensor<4x8x2xi32>) -> tensor<4x8x2xi32>
  %blk = linalg.reduce ins(%prod : tensor<4x8x2x32xi32>) outs(%ri : tensor<4x8x2xi32>) dimensions = [3] {qgemm.node = {root = 0 : i64, stage = "compute"}}
    (%in: i32, %acc: i32) {
      %s = arith.addi %in, %acc : i32
      linalg.yield %s : i32
    }
  %se = tensor.empty() : tensor<4x8x2xf32>
  %scaled = linalg.generic {indexing_maps = [#id3, #a3, #w3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%blk, %ad, %wd : tensor<4x8x2xi32>, tensor<4x2xf32>, tensor<8x2xf32>) outs(%se : tensor<4x8x2xf32>) attrs = {qgemm.node = {root = 0 : i64, stage = "compute"}} {
  ^bb0(%v: i32, %x: f32, %y: f32, %o: f32):
    %fv = arith.sitofp %v : i32 to f32
    %xy = arith.mulf %x, %y : f32
    %r = arith.mulf %fv, %xy : f32
    linalg.yield %r : f32
  } -> tensor<4x8x2xf32>
  %fe = tensor.empty() : tensor<4x8xf32>
  %fi = linalg.fill ins(%f0 : f32) outs(%fe : tensor<4x8xf32>) -> tensor<4x8xf32>
  %res = linalg.reduce ins(%scaled : tensor<4x8x2xf32>) outs(%fi : tensor<4x8xf32>) dimensions = [2] {qgemm.node = {root = 0 : i64, stage = "compute"}, qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}}
    (%in: f32, %acc: f32) {
      %s = arith.addf %in, %acc : f32
      linalg.yield %s : f32
    }
  return %res : tensor<4x8xf32>
}
