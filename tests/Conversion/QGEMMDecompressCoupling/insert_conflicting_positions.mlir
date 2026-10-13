// RUN: buddy-opt --qgemm-decompress-coupling=decompress=on-the-fly %s 2>&1 | FileCheck %s
//
// Conflicting consumer positions (Req 5.7, 6.8). The activation extension
// %acast has two consumers in V_n that read 'n' at different positions: the
// multiply reads it at position 1 through its folded broadcast, and the
// transpose reads it at position 3 through an explicit pure broadcast whose
// iteration order puts 'n' last. A lifted extension could satisfy only one of
// them, so under on-the-fly it is not insertable: one R3 on %acast, no remedy.
// Its producer has a user outside L_n and stays out silently. No consumer's
// operand map changes: each consumer keeps reading a pure broadcast of the
// unlifted extension at its own position.
//
// The input carries a hand-written annotation and skips
// --qgemm-simplify-dim-transforms --qgemm-recognize, which would forward the
// transpose past the element-wise add and remove the conflict.
// CHECK: remark: qgemm-decompress-coupling: 'n' cannot be inserted into op; kept outside the 'n' space
// CHECK-NEXT: %acast = linalg.generic
// CHECK-NOT: remark:
// CHECK-DAG: #[[$A4:map[0-9]*]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG: #[[$B4:map[0-9]*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-DAG: #[[$ID4:map[0-9]*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @conflicting_positions
// The extension keeps its (m,k) type.
// CHECK: %[[AC:[a-z0-9_]+]] = linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<4x2x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<4x2x32xi32>) {{.*}}stage = "data_prep"
// Two pure broadcasts of it, one per position.
// CHECK-DAG: %[[AN1:[a-z0-9_]+]] = linalg.generic {indexing_maps = [#[[$A4]], #[[$ID4]]], {{.*}} ins(%[[AC]] : tensor<4x2x32xi32>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi32>) {{.*}}stage = "compute"
// CHECK-DAG: %[[AN3:[a-z0-9_]+]] = linalg.generic {indexing_maps = [#[[$B4]], #[[$ID4]]], {{.*}} ins(%[[AC]] : tensor<4x2x32xi32>) outs(%{{[a-z0-9_]+}} : tensor<4x2x32x8xi32>) {{.*}}stage = "compute"
// The multiply reads the position-1 broadcast through an identity map.
// CHECK: linalg.generic {indexing_maps = [#[[$ID4]], #[[$ID4]], #[[$ID4]]], {{.*}} ins(%[[AN1]], %{{[a-z0-9_]+}} : tensor<4x8x2x32xi32>, tensor<4x8x2x32xi32>)
// CHECK-NEXT: ^bb0
// CHECK-NEXT: arith.muli
// The transpose keeps its permutation and reads the position-3 broadcast.
// CHECK: linalg.transpose ins(%[[AN3]] : tensor<4x2x32x8xi32>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi32>) permutation = [0, 3, 1, 2]
// CHECK: qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// CHECK-NOT: remark:
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#a4 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#w4 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#b4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#a3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#w3 = affine_map<(d0, d1, d2) -> (d1, d2)>
func.func @conflicting_positions(%aq: tensor<4x64xi8>, %wq: tensor<8x64xi8>, %ad: tensor<4x2xf32>, %wd: tensor<8x2xf32>) -> tensor<4x8xf32> {
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
  // Consumer 1: the multiply, reading 'n' of %acast at position 1.
  %pe = tensor.empty() : tensor<4x8x2x32xi32>
  %prod = linalg.generic {indexing_maps = [#a4, #w4, #id4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%acast, %wcast : tensor<4x2x32xi32>, tensor<8x2x32xi32>) outs(%pe : tensor<4x8x2x32xi32>) attrs = {qgemm.node = {root = 0 : i64, stage = "compute"}} {
  ^bb0(%x: i32, %y: i32, %o: i32):
    %z = arith.muli %x, %y : i32
    linalg.yield %z : i32
  } -> tensor<4x8x2x32xi32>
  // Consumer 2: a transpose of a pure broadcast putting 'n' at position 3.
  %be = tensor.empty() : tensor<4x2x32x8xi32>
  %abl = linalg.generic {indexing_maps = [#b4, #id4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%acast : tensor<4x2x32xi32>) outs(%be : tensor<4x2x32x8xi32>) attrs = {qgemm.node = {root = 0 : i64, stage = "compute"}} {
  ^bb0(%i: i32, %o: i32):
    linalg.yield %i : i32
  } -> tensor<4x2x32x8xi32>
  %te = tensor.empty() : tensor<4x8x2x32xi32>
  %abt = linalg.transpose ins(%abl : tensor<4x2x32x8xi32>) outs(%te : tensor<4x8x2x32xi32>) permutation = [0, 3, 1, 2] {qgemm.node = {root = 0 : i64, stage = "compute"}}
  %se = tensor.empty() : tensor<4x8x2x32xi32>
  %sum = linalg.generic {indexing_maps = [#id4, #id4, #id4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%prod, %abt : tensor<4x8x2x32xi32>, tensor<4x8x2x32xi32>) outs(%se : tensor<4x8x2x32xi32>) attrs = {qgemm.node = {root = 0 : i64, stage = "compute"}} {
  ^bb0(%x: i32, %y: i32, %o: i32):
    %z = arith.addi %x, %y : i32
    linalg.yield %z : i32
  } -> tensor<4x8x2x32xi32>
  %c0 = arith.constant 0 : i32
  %f0 = arith.constant 0.0 : f32
  %re = tensor.empty() : tensor<4x8x2xi32>
  %ri = linalg.fill ins(%c0 : i32) outs(%re : tensor<4x8x2xi32>) -> tensor<4x8x2xi32>
  %blk = linalg.reduce ins(%sum : tensor<4x8x2x32xi32>) outs(%ri : tensor<4x8x2xi32>) dimensions = [3] {qgemm.node = {root = 0 : i64, stage = "compute"}}
    (%in: i32, %acc: i32) {
      %s = arith.addi %in, %acc : i32
      linalg.yield %s : i32
    }
  %sce = tensor.empty() : tensor<4x8x2xf32>
  %scaled = linalg.generic {indexing_maps = [#id3, #a3, #w3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%blk, %ad, %wd : tensor<4x8x2xi32>, tensor<4x2xf32>, tensor<8x2xf32>) outs(%sce : tensor<4x8x2xf32>) attrs = {qgemm.node = {root = 0 : i64, stage = "compute"}} {
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
