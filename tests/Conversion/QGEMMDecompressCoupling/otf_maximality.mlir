// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %s 2>&1 | FileCheck %s
//
// Property 6 (on-the-fly maximality), Requirements 5.2-5.5.
// The weight chain is expand_shape -> ext -> off -> {lo, hi} -> join -> multiply,
// and the function also returns %off, an Outside_User of the decompression chain.
//  (a) %off has an Outside_User, so it stays in (n,k) space; %ext reaches L_m
//      only through %off, so it stays too. Everything downstream (lo, hi, join)
//      is lifted into m, fed by a single broadcast of %off.
//  (b) Neither boundary load (the two tensor.expand_shape ops) gains a
//      parallel dim; the broadcasts are inserted after them.
//  (c) The lo/hi diamond is lifted together with its join, one clone each.
// CHECK-NOT: remark
// CHECK-LABEL: func.func @otf_max
// Activation boundary load keeps its shape; the n broadcast sits after it.
// CHECK: %[[AE:[a-z0-9_]+]] = tensor.expand_shape %{{[a-z0-9_]+}} {{.*}} into tensor<4x2x32xi8>
// CHECK: linalg.generic {{.*}} ins(%[[AE]] : tensor<4x2x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi8>)
// CHECK: %[[PA:[a-z0-9_]+]] = linalg.generic {{.*}} outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi32>)
// CHECK-NEXT: ^bb0
// CHECK-NEXT: arith.extsi
// Weight boundary load keeps its shape.
// CHECK: %[[WE:[a-z0-9_]+]] = tensor.expand_shape %{{[a-z0-9_]+}} {{.*}} into tensor<8x2x32xi8>
// %ext and %off stay outside H_m (blocked by the Outside_User of %off).
// CHECK: %[[EXT:[a-z0-9_]+]] = linalg.generic {{.*}} ins(%[[WE]] : tensor<8x2x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<8x2x32xi32>) {{.*}}stage = "data_prep"
// CHECK-NEXT: ^bb0
// CHECK-NEXT: arith.extsi
// CHECK: %[[OFF:[a-z0-9_]+]] = linalg.generic {{.*}} ins(%[[EXT]] : tensor<8x2x32xi32>) outs(%{{[a-z0-9_]+}} : tensor<8x2x32xi32>) {{.*}}stage = "data_prep"
// CHECK-NEXT: ^bb0
// CHECK-NEXT: arith.subi
// One m broadcast of %off feeds both diamond branches.
// CHECK: %[[WB:[a-z0-9_]+]] = linalg.generic {{.*}} ins(%[[OFF]] : tensor<8x2x32xi32>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi32>) {{.*}}stage = "compute"
// CHECK-NEXT: ^bb0
// CHECK-NEXT: linalg.yield
// CHECK: %[[LO:[a-z0-9_]+]] = linalg.generic {{.*}} ins(%[[WB]] : tensor<4x8x2x32xi32>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi32>) {{.*}}stage = "compute"
// CHECK-NEXT: ^bb0
// CHECK-NEXT: arith.andi
// CHECK: %[[HI:[a-z0-9_]+]] = linalg.generic {{.*}} ins(%[[WB]] : tensor<4x8x2x32xi32>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi32>) {{.*}}stage = "compute"
// CHECK-NEXT: ^bb0
// CHECK-NEXT: arith.shrsi
// CHECK: %[[J:[a-z0-9_]+]] = linalg.generic {{.*}} ins(%[[LO]], %[[HI]] : tensor<4x8x2x32xi32>, tensor<4x8x2x32xi32>) {{.*}}stage = "compute"
// CHECK-NEXT: ^bb0
// CHECK-NEXT: arith.subi
// CHECK: linalg.generic {{.*}} ins(%[[PA]], %[[J]] : tensor<4x8x2x32xi32>, tensor<4x8x2x32xi32>) {{.*}}stage = "compute"
// CHECK-NEXT: ^bb0
// CHECK-NEXT: arith.muli
// No (n,k)-space copy of the diamond remains.
// CHECK-NOT: arith.andi
// CHECK-NOT: arith.shrsi
// CHECK: qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// The Outside_User still sees the original (n,k)-space %off.
// CHECK: return %{{[a-z0-9_]+}}, %[[OFF]] : tensor<4x8xf32>, tensor<8x2x32xi32>
// CHECK-NOT: remark
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#a4 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#w4 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#a3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#w3 = affine_map<(d0, d1, d2) -> (d1, d2)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
func.func @otf_max(%aq: tensor<4x64xi8>, %wq: tensor<8x64xi8>, %ad: tensor<4x2xf32>, %wd: tensor<8x2xf32>) -> (tensor<4x8xf32>, tensor<8x2x32xi32>) {
  %ae = tensor.expand_shape %aq [[0], [1, 2]] output_shape [4, 2, 32] : tensor<4x64xi8> into tensor<4x2x32xi8>
  %ace = tensor.empty() : tensor<4x2x32xi32>
  %acast = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%ae : tensor<4x2x32xi8>) outs(%ace : tensor<4x2x32xi32>) {
  ^bb0(%i: i8, %o: i32):
    %x = arith.extsi %i : i8 to i32
    linalg.yield %x : i32
  } -> tensor<4x2x32xi32>
  %we = tensor.expand_shape %wq [[0], [1, 2]] output_shape [8, 2, 32] : tensor<8x64xi8> into tensor<8x2x32xi8>
  %c8 = arith.constant 8 : i32
  %exe = tensor.empty() : tensor<8x2x32xi32>
  %ext = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%we : tensor<8x2x32xi8>) outs(%exe : tensor<8x2x32xi32>) {
  ^bb0(%i: i8, %o: i32):
    %x = arith.extsi %i : i8 to i32
    linalg.yield %x : i32
  } -> tensor<8x2x32xi32>
  %ofe = tensor.empty() : tensor<8x2x32xi32>
  %off = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%ext : tensor<8x2x32xi32>) outs(%ofe : tensor<8x2x32xi32>) {
  ^bb0(%i: i32, %o: i32):
    %x = arith.subi %i, %c8 : i32
    linalg.yield %x : i32
  } -> tensor<8x2x32xi32>
  %c15 = arith.constant 15 : i32
  %c4 = arith.constant 4 : i32
  %loe = tensor.empty() : tensor<8x2x32xi32>
  %lo = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%off : tensor<8x2x32xi32>) outs(%loe : tensor<8x2x32xi32>) {
  ^bb0(%x: i32, %o: i32):
    %lo_v = arith.andi %x, %c15 : i32
    linalg.yield %lo_v : i32
  } -> tensor<8x2x32xi32>
  %hie = tensor.empty() : tensor<8x2x32xi32>
  %hi = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%off : tensor<8x2x32xi32>) outs(%hie : tensor<8x2x32xi32>) {
  ^bb0(%x: i32, %o: i32):
    %hi_v = arith.shrsi %x, %c4 : i32
    linalg.yield %hi_v : i32
  } -> tensor<8x2x32xi32>
  %je = tensor.empty() : tensor<8x2x32xi32>
  %join = linalg.generic {indexing_maps = [#id3, #id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%lo, %hi : tensor<8x2x32xi32>, tensor<8x2x32xi32>) outs(%je : tensor<8x2x32xi32>) {
  ^bb0(%l: i32, %h: i32, %o: i32):
    %s = arith.subi %l, %h : i32
    linalg.yield %s : i32
  } -> tensor<8x2x32xi32>
  %pe = tensor.empty() : tensor<4x8x2x32xi32>
  %prod = linalg.generic {indexing_maps = [#a4, #w4, #id4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%acast, %join : tensor<4x2x32xi32>, tensor<8x2x32xi32>) outs(%pe : tensor<4x8x2x32xi32>) {
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
  return %res, %off : tensor<4x8xf32>, tensor<8x2x32xi32>
}
