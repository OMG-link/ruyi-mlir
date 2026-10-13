// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre %s 2>&1 | FileCheck %s --check-prefix=PRE
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %s 2>&1 | FileCheck %s --check-prefix=OTF
//
// Uniform chain placement (Property 8) on a nontrivial multi-op Factor_Chain.
// The weight block scale is decoded from packed bits as in Q4_K sc/m:
// zero-extend, shift right, mask the low 6 bits, convert, multiply by the
// per-row super-block scale. The Factor_Chain is every op from the packed
// scale argument to the scaling generic where it joins the block sum.
//
// Under pre, no factor-chain op is in V_m, so each one keeps only (n, b), and
// the single PB inserting m into the chain output feeds the scaling generic
// directly. Under on-the-fly, the factor chain obeys the same maximality rule
// as the value chain: the boundary load keeps its axes, the m broadcast takes
// its result and every later decoding op carries m and n.
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#a4 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#w4 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#a3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#w3 = affine_map<(d0, d1, d2) -> (d1, d2)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>
// PRE-NOT: remark
// OTF-NOT: remark
// PRE-LABEL: func.func @uniform_chain
// OTF-LABEL: func.func @uniform_chain
//
// Pre: factor-chain ops keep shape 8x2 (n, b): extui, shrui, andi, sitofp+mulf.
// PRE: linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<8x2xi8>) outs(%{{[a-z0-9_]+}} : tensor<8x2xi32>) {{.*}}stage = "data_prep"
// PRE-NEXT: ^bb0
// PRE-NEXT: arith.extui
// PRE: linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<8x2xi32>) outs(%{{[a-z0-9_]+}} : tensor<8x2xi32>) {{.*}}stage = "data_prep"
// PRE-NEXT: ^bb0
// PRE-NEXT: arith.constant
// PRE-NEXT: arith.shrui
// PRE: linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<8x2xi32>) outs(%{{[a-z0-9_]+}} : tensor<8x2xi32>) {{.*}}stage = "data_prep"
// PRE-NEXT: ^bb0
// PRE-NEXT: arith.constant
// PRE-NEXT: arith.andi
// PRE: %[[WF:[a-z0-9_]+]] = linalg.generic {{.*}} ins(%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}} : tensor<8x2xi32>, tensor<8xf32>) outs(%{{[a-z0-9_]+}} : tensor<8x2xf32>) {{.*}}stage = "data_prep"
// PRE-NEXT: ^bb0
// PRE-NEXT: arith.sitofp
// PRE-NEXT: arith.mulf
// The only PB inserting m into the factor chain output feeds the join.
// PRE: %[[WFB:[a-z0-9_]+]] = linalg.generic {{.*}} ins(%[[WF]] : tensor<8x2xf32>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2xf32>) {{.*}}stage = "compute"
// PRE-NEXT: ^bb0
// PRE-NEXT: linalg.yield
// PRE: linalg.generic {{.*}} ins(%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %[[WFB]] : tensor<4x8x2xi32>, tensor<4x8x2xf32>, tensor<4x8x2xf32>)
// PRE: linalg.reduce {{.*}} dimensions = [2] {{.*}}qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
//
// On-the-fly: the extui reads the packed argument directly, so it is the
// chain's boundary load and keeps (n, b); the m broadcast takes its result and
// every later decoding op is (m, n, b).
// OTF: %[[L:[a-z0-9_]+]] = linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<8x2xi8>) outs(%{{[a-z0-9_]+}} : tensor<8x2xi32>) {{.*}}stage = "data_prep"
// OTF-NEXT: ^bb0
// OTF-NEXT: arith.extui
// OTF: linalg.generic {{.*}} ins(%[[L]] : tensor<8x2xi32>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2xi32>) {{.*}}stage = "compute"
// OTF-NEXT: ^bb0
// OTF-NEXT: linalg.yield
// OTF: linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<4x8x2xi32>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2xi32>) {{.*}}stage = "compute"
// OTF-NEXT: ^bb0
// OTF-NEXT: arith.constant
// OTF-NEXT: arith.shrui
// OTF: linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<4x8x2xi32>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2xi32>) {{.*}}stage = "compute"
// OTF-NEXT: ^bb0
// OTF-NEXT: arith.constant
// OTF-NEXT: arith.andi
// OTF: %[[WF:[a-z0-9_]+]] = linalg.generic {{.*}} ins(%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}} : tensor<4x8x2xi32>, tensor<4x8xf32>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2xf32>) {{.*}}stage = "compute"
// OTF-NEXT: ^bb0
// OTF-NEXT: arith.sitofp
// OTF-NEXT: arith.mulf
// OTF: linalg.generic {{.*}} ins(%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %[[WF]] : tensor<4x8x2xi32>, tensor<4x8x2xf32>, tensor<4x8x2xf32>)
// OTF-NOT: tensor<8x2xf32>
// OTF: linalg.reduce {{.*}} dimensions = [2] {{.*}}qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// PRE-NOT: remark
// OTF-NOT: remark
func.func @uniform_chain(%aq: tensor<4x64xi8>, %wq: tensor<8x64xi8>, %ad: tensor<4x2xf32>,
                         %wsc: tensor<8x2xi8>, %wsup: tensor<8xf32>) -> tensor<4x8xf32> {
  %ae = tensor.expand_shape %aq [[0], [1, 2]] output_shape [4, 2, 32] : tensor<4x64xi8> into tensor<4x2x32xi8>
  %ace = tensor.empty() : tensor<4x2x32xi32>
  %acast = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%ae : tensor<4x2x32xi8>) outs(%ace : tensor<4x2x32xi32>) {
  ^bb0(%i: i8, %o: i32):
    %x = arith.extsi %i : i8 to i32
    linalg.yield %x : i32
  } -> tensor<4x2x32xi32>
  %we = tensor.expand_shape %wq [[0], [1, 2]] output_shape [8, 2, 32] : tensor<8x64xi8> into tensor<8x2x32xi8>
  %wce = tensor.empty() : tensor<8x2x32xi32>
  %wcast = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%we : tensor<8x2x32xi8>) outs(%wce : tensor<8x2x32xi32>) {
  ^bb0(%i: i8, %o: i32):
    %x = arith.extsi %i : i8 to i32
    linalg.yield %x : i32
  } -> tensor<8x2x32xi32>
  // Factor_Chain: decode the 6-bit block scale from packed bits.
  %fe = tensor.empty() : tensor<8x2xi32>
  %f1 = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%wsc : tensor<8x2xi8>) outs(%fe : tensor<8x2xi32>) {
  ^bb0(%i: i8, %o: i32):
    %x = arith.extui %i : i8 to i32
    linalg.yield %x : i32
  } -> tensor<8x2xi32>
  %f2 = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%f1 : tensor<8x2xi32>) outs(%fe : tensor<8x2xi32>) {
  ^bb0(%i: i32, %o: i32):
    %c2 = arith.constant 2 : i32
    %x = arith.shrui %i, %c2 : i32
    linalg.yield %x : i32
  } -> tensor<8x2xi32>
  %f3 = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%f2 : tensor<8x2xi32>) outs(%fe : tensor<8x2xi32>) {
  ^bb0(%i: i32, %o: i32):
    %c63 = arith.constant 63 : i32
    %x = arith.andi %i, %c63 : i32
    linalg.yield %x : i32
  } -> tensor<8x2xi32>
  %wde = tensor.empty() : tensor<8x2xf32>
  %wd = linalg.generic {indexing_maps = [#id2, #row2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%f3, %wsup : tensor<8x2xi32>, tensor<8xf32>) outs(%wde : tensor<8x2xf32>) {
  ^bb0(%i: i32, %s: f32, %o: f32):
    %x = arith.sitofp %i : i32 to f32
    %y = arith.mulf %x, %s : f32
    linalg.yield %y : f32
  } -> tensor<8x2xf32>
  %pe = tensor.empty() : tensor<4x8x2x32xi32>
  %prod = linalg.generic {indexing_maps = [#a4, #w4, #id4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%acast, %wcast : tensor<4x2x32xi32>, tensor<8x2x32xi32>) outs(%pe : tensor<4x8x2x32xi32>) {
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
  %oe = tensor.empty() : tensor<4x8xf32>
  %oi = linalg.fill ins(%f0 : f32) outs(%oe : tensor<4x8xf32>) -> tensor<4x8xf32>
  %res = linalg.reduce ins(%scaled : tensor<4x8x2xf32>) outs(%oi : tensor<4x8xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %s = arith.addf %in, %acc : f32
      linalg.yield %s : f32
    }
  return %res : tensor<4x8xf32>
}
