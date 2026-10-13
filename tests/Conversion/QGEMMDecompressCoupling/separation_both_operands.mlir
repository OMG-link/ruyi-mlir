// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre %s 2>&1 | FileCheck %s --check-prefix=PRE
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %s 2>&1 | FileCheck %s --check-prefix=OTF
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#a4 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#w4 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#a3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#w3 = affine_map<(d0, d1, d2) -> (d1, d2)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
//
// Nontrivial decompression on both operands: the activation is cast and
// zero-point corrected, the weight is cast, masked and decoded through a
// lookup table. Neither chain is special to the pass.
// PRE-NOT: remark
// OTF-NOT: remark
//
// Under pre, the activation chain stays in (m,k) and the weight chain,
// including the lookup, stays in (n,k). Under on-the-fly, both chains are
// lifted: the activation gains n and the weight, lookup included, gains m.
// PRE-LABEL: func.func @both_operands
// OTF-LABEL: func.func @both_operands
// PRE: linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<4x2x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<4x2x32xi32>) {{.*}}stage = "data_prep"
// PRE-NEXT: ^bb0
// PRE-NEXT: arith.extsi
// PRE: %[[AZ:[a-z0-9_]+]] = linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<4x2x32xi32>) outs(%{{[a-z0-9_]+}} : tensor<4x2x32xi32>) {{.*}}stage = "data_prep"
// PRE-NEXT: ^bb0
// PRE-NEXT: arith.subi
// PRE: %[[PA:[a-z0-9_]+]] = linalg.generic {{.*}} ins(%[[AZ]] : tensor<4x2x32xi32>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi32>) {{.*}}stage = "compute"
// PRE-NEXT: ^bb0
// PRE-NEXT: linalg.yield
// PRE: linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<8x2x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<8x2x32xi32>) {{.*}}stage = "data_prep"
// PRE-NEXT: ^bb0
// PRE-NEXT: arith.extsi
// PRE-NEXT: arith.andi
// PRE: %[[WL:[a-z0-9_]+]] = linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<8x2x32xi32>) outs(%{{[a-z0-9_]+}} : tensor<8x2x32xi32>) {{.*}}stage = "data_prep"
// PRE-NEXT: ^bb0
// PRE-NEXT: arith.index_cast
// PRE-NEXT: tensor.extract
// PRE: %[[PW:[a-z0-9_]+]] = linalg.generic {{.*}} ins(%[[WL]] : tensor<8x2x32xi32>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi32>) {{.*}}stage = "compute"
// PRE-NEXT: ^bb0
// PRE-NEXT: linalg.yield
// PRE: linalg.generic {{.*}} ins(%[[PA]], %[[PW]] : tensor<4x8x2x32xi32>, tensor<4x8x2x32xi32>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi32>) {{.*}}stage = "compute"
// PRE-NEXT: ^bb0
// PRE-NEXT: arith.muli
// OTF: linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<4x2x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi8>) {{.*}}stage = "compute"
// OTF-NEXT: ^bb0
// OTF-NEXT: linalg.yield
// OTF: %[[AC:[a-z0-9_]+]] = linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi32>) {{.*}}stage = "compute"
// OTF-NEXT: ^bb0
// OTF-NEXT: arith.extsi
// OTF: %[[PA:[a-z0-9_]+]] = linalg.generic {{.*}} ins(%[[AC]] : tensor<4x8x2x32xi32>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi32>) {{.*}}stage = "compute"
// OTF-NEXT: ^bb0
// OTF-NEXT: arith.subi
// OTF: linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<8x2x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi8>) {{.*}}stage = "compute"
// OTF-NEXT: ^bb0
// OTF-NEXT: linalg.yield
// OTF: linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi32>) {{.*}}stage = "compute"
// OTF-NEXT: ^bb0
// OTF-NEXT: arith.extsi
// OTF-NEXT: arith.andi
// OTF: %[[PW:[a-z0-9_]+]] = linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi32>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi32>) {{.*}}stage = "compute"
// OTF-NEXT: ^bb0
// OTF-NEXT: arith.index_cast
// OTF-NEXT: tensor.extract
// OTF: linalg.generic {{.*}} ins(%[[PA]], %[[PW]] : tensor<4x8x2x32xi32>, tensor<4x8x2x32xi32>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi32>) {{.*}}stage = "compute"
// OTF-NEXT: ^bb0
// OTF-NEXT: arith.muli
// PRE: linalg.reduce ins(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi32>) {{.*}} dimensions = [3] {{.*}}stage = "compute"
// PRE: linalg.reduce {{.*}} dimensions = [2] {{.*}}stage = "compute"}, qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// PRE-NOT: remark
// OTF: linalg.reduce ins(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi32>) {{.*}} dimensions = [3] {{.*}}stage = "compute"
// OTF: linalg.reduce {{.*}} dimensions = [2] {{.*}}stage = "compute"}, qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// OTF-NOT: remark
func.func @both_operands(%aq: tensor<4x64xi8>, %wq: tensor<8x64xi8>, %ad: tensor<4x2xf32>, %wd: tensor<8x2xf32>, %tab: tensor<16xi32>) -> tensor<4x8xf32> {
  %ae = tensor.expand_shape %aq [[0], [1, 2]] output_shape [4, 2, 32] : tensor<4x64xi8> into tensor<4x2x32xi8>
  %ace = tensor.empty() : tensor<4x2x32xi32>
  %acast = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%ae : tensor<4x2x32xi8>) outs(%ace : tensor<4x2x32xi32>) {
  ^bb0(%i: i8, %o: i32):
    %x = arith.extsi %i : i8 to i32
    linalg.yield %x : i32
  } -> tensor<4x2x32xi32>
  %c3 = arith.constant 3 : i32
  %c15 = arith.constant 15 : i32
  %aze = tensor.empty() : tensor<4x2x32xi32>
  %az = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%acast : tensor<4x2x32xi32>) outs(%aze : tensor<4x2x32xi32>) {
  ^bb0(%i: i32, %o: i32):
    %x = arith.subi %i, %c3 : i32
    linalg.yield %x : i32
  } -> tensor<4x2x32xi32>
  %we = tensor.expand_shape %wq [[0], [1, 2]] output_shape [8, 2, 32] : tensor<8x64xi8> into tensor<8x2x32xi8>
  %wce = tensor.empty() : tensor<8x2x32xi32>
  %wmask = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%we : tensor<8x2x32xi8>) outs(%wce : tensor<8x2x32xi32>) {
  ^bb0(%i: i8, %o: i32):
    %x = arith.extsi %i : i8 to i32
    %y = arith.andi %x, %c15 : i32
    linalg.yield %y : i32
  } -> tensor<8x2x32xi32>
  %wle = tensor.empty() : tensor<8x2x32xi32>
  %wlut = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%wmask : tensor<8x2x32xi32>) outs(%wle : tensor<8x2x32xi32>) {
  ^bb0(%i: i32, %o: i32):
    %idx = arith.index_cast %i : i32 to index
    %v = tensor.extract %tab[%idx] : tensor<16xi32>
    linalg.yield %v : i32
  } -> tensor<8x2x32xi32>
  %pe = tensor.empty() : tensor<4x8x2x32xi32>
  %prod = linalg.generic {indexing_maps = [#a4, #w4, #id4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%az, %wlut : tensor<4x2x32xi32>, tensor<8x2x32xi32>) outs(%pe : tensor<4x8x2x32xi32>) {
  ^bb0(%x: i32, %y: i32, %o: i32):
    %z = arith.muli %x, %y : i32
    linalg.yield %z : i32
  } -> tensor<4x8x2x32xi32>
  %c0 = arith.constant 0 : i32
  %f0 = arith.constant 0.0 : f32
  %re = tensor.empty() : tensor<4x8x2xi32>
  %ri = linalg.fill ins(%c0 : i32) outs(%re : tensor<4x8x2xi32>) -> tensor<4x8x2xi32>
  %blk = linalg.reduce ins(%prod : tensor<4x8x2x32xi32>) outs(%ri : tensor<4x8x2xi32>) dimensions = [3]
    (%in: i32, %acc: i32) {
      %s = arith.addi %in, %acc : i32
      linalg.yield %s : i32
    }
  %se = tensor.empty() : tensor<4x8x2xf32>
  %scaled = linalg.generic {indexing_maps = [#id3, #a3, #w3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%blk, %ad, %wd : tensor<4x8x2xi32>, tensor<4x2xf32>, tensor<8x2xf32>) outs(%se : tensor<4x8x2xf32>) {
  ^bb0(%v: i32, %x: f32, %y: f32, %o: f32):
    %fv = arith.sitofp %v : i32 to f32
    %xy = arith.mulf %x, %y : f32
    %r = arith.mulf %fv, %xy : f32
    linalg.yield %r : f32
  } -> tensor<4x8x2xf32>
  %fe = tensor.empty() : tensor<4x8xf32>
  %fi = linalg.fill ins(%f0 : f32) outs(%fe : tensor<4x8xf32>) -> tensor<4x8xf32>
  %res = linalg.reduce ins(%scaled : tensor<4x8x2xf32>) outs(%fi : tensor<4x8xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %s = arith.addf %in, %acc : f32
      linalg.yield %s : f32
    }
  return %res : tensor<4x8xf32>
}
