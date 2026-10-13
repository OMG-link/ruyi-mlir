// RUN: buddy-opt --qgemm-decompress-coupling=decompress=on-the-fly %s 2>&1 | FileCheck %s
//
// Insertion into grouped reshapes (Req 6.8, 7.4, 5.7). Under on-the-fly the
// activation chain is a candidate for 'n', and it passes through a
// collapse_shape/expand_shape pair that groups 'm' with the block dim. The
// multiply reads 'n' right after 'm', inside that group, and an inserted axis
// never splits an existing group, so the expand_shape cannot take 'n' at the
// position its consumer requires: one R3 on it, no operand-map adjustment.
// Its producers have a user outside L_n and stay out silently, and the 'n'
// broadcast stays on the expand_shape -> multiply edge at the position the
// multiply reads. The weight side is lifted into 'm' at its load.
//
// The input carries a hand-written annotation and skips
// --qgemm-simplify-dim-transforms --qgemm-recognize: the simplifier folds the
// collapse/expand pair away.
// CHECK: remark: qgemm-decompress-coupling: 'n' cannot be inserted into op; kept outside the 'n' space
// CHECK-NEXT: %aung = tensor.expand_shape
// CHECK-NOT: remark:
// CHECK-LABEL: func.func @r3_grouped
// The activation chain keeps its input shapes and both groups.
// CHECK: %[[AC:[a-z0-9_]+]] = linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<4x2x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<4x2x32xi32>) {{.*}}stage = "data_prep"
// The weight is lifted into 'm' at the load.
// CHECK: linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<8x2x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi8>)
// CHECK: %[[GRP:[a-z0-9_]+]] = tensor.collapse_shape %[[AC]] {{\[\[}}0, 1], [2]] {{.*}}stage = "data_prep"{{.*}} : tensor<4x2x32xi32> into tensor<8x32xi32>
// CHECK: %[[UNG:[a-z0-9_]+]] = tensor.expand_shape %[[GRP]] {{\[\[}}0, 1], [2]] output_shape [4, 2, 32] {{.*}}stage = "data_prep"{{.*}} : tensor<8x32xi32> into tensor<4x2x32xi32>
// The 'n' broadcast inserts 'n' at position 1, where the multiply reads it.
// CHECK: %[[UNGN:[a-z0-9_]+]] = linalg.generic {indexing_maps = [#[[$AMAP:map[0-9]*]], #[[$ID:map[0-9]*]]], {{.*}} ins(%[[UNG]] : tensor<4x2x32xi32>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi32>) {{.*}}stage = "compute"
// CHECK: linalg.generic {indexing_maps = [#[[$ID]], #[[$ID]], #[[$ID]]], {{.*}} ins(%[[UNGN]], %{{[a-z0-9_]+}} : tensor<4x8x2x32xi32>, tensor<4x8x2x32xi32>)
// CHECK-NEXT: ^bb0
// CHECK-NEXT: arith.muli
// CHECK: qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// CHECK-NOT: remark:
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#a4 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#w4 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#a3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#w3 = affine_map<(d0, d1, d2) -> (d1, d2)>
func.func @r3_grouped(%aq: tensor<4x64xi8>, %wq: tensor<8x64xi8>, %ad: tensor<4x2xf32>, %wd: tensor<8x2xf32>) -> tensor<4x8xf32> {
  %ae = tensor.expand_shape %aq [[0], [1, 2]] output_shape [4, 2, 32] {qgemm.node = {root = 0 : i64, stage = "data_prep"}} : tensor<4x64xi8> into tensor<4x2x32xi8>
  %ace = tensor.empty() : tensor<4x2x32xi32>
  %acast = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%ae : tensor<4x2x32xi8>) outs(%ace : tensor<4x2x32xi32>) attrs = {qgemm.node = {root = 0 : i64, stage = "data_prep"}} {
  ^bb0(%i: i8, %o: i32):
    %x = arith.extsi %i : i8 to i32
    linalg.yield %x : i32
  } -> tensor<4x2x32xi32>
      %we = tensor.expand_shape %wq [[0], [1, 2]] output_shape [8, 2, 32] {qgemm.node = {root = 0 : i64, stage = "data_prep"}} : tensor<8x64xi8> into tensor<8x2x32xi8>
  %wce = tensor.empty() : tensor<8x2x32xi32>
  %wcast = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%we : tensor<8x2x32xi8>) outs(%wce : tensor<8x2x32xi32>) attrs = {qgemm.node = {root = 0 : i64, stage = "data_prep"}} {
  ^bb0(%i: i8, %o: i32):
    %x = arith.extsi %i : i8 to i32
    linalg.yield %x : i32
  } -> tensor<8x2x32xi32>
  %agrp = tensor.collapse_shape %acast [[0, 1], [2]] {qgemm.node = {root = 0 : i64, stage = "data_prep"}} : tensor<4x2x32xi32> into tensor<8x32xi32>
  %aung = tensor.expand_shape %agrp [[0, 1], [2]] output_shape [4, 2, 32] {qgemm.node = {root = 0 : i64, stage = "data_prep"}} : tensor<8x32xi32> into tensor<4x2x32xi32>
  %pe = tensor.empty() : tensor<4x8x2x32xi32>
  %prod = linalg.generic {indexing_maps = [#a4, #w4, #id4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%aung, %wcast : tensor<4x2x32xi32>, tensor<8x2x32xi32>) outs(%pe : tensor<4x8x2x32xi32>) attrs = {qgemm.node = {root = 0 : i64, stage = "compute"}} {
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
