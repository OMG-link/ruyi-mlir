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
// A prefix shared by a value chain and a factor chain: the packed weight row
// holds one scale byte followed by 32 quantized values per block. One cast
// feeds both the value slice and the factor slice.
// PRE-NOT: remark
// OTF-NOT: remark
//
// Under pre, the shared cast, the value slice and the factor slice keep only
// the n axis, and the factor reaches the scaling generic through one broadcast
// along m. Under on-the-fly, the shared prefix is lifted into m once and both
// slices read the lifted cast.
// PRE-LABEL: func.func @shared_prefix
// OTF-LABEL: func.func @shared_prefix
// PRE: linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<4x2x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<4x2x32xi32>) {{.*}}stage = "data_prep"
// PRE-NEXT: ^bb0
// PRE-NEXT: arith.extsi
// PRE: %[[P:[a-z0-9_]+]] = linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<8x2x33xi8>) outs(%{{[a-z0-9_]+}} : tensor<8x2x33xi32>) {{.*}}stage = "data_prep"
// PRE-NEXT: ^bb0
// PRE-NEXT: arith.extsi
// PRE: %[[WV:[a-z0-9_]+]] = tensor.extract_slice %[[P]][0, 0, 1] [8, 2, 32] [1, 1, 1] {{.*}}stage = "data_prep"
// PRE: linalg.generic {{.*}} ins(%[[WV]] : tensor<8x2x32xi32>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi32>) {{.*}}stage = "compute"
// PRE-NEXT: ^bb0
// PRE-NEXT: linalg.yield
// PRE: %[[WS:[a-z0-9_]+]] = tensor.extract_slice %[[P]][0, 0, 0] [8, 2, 1] [1, 1, 1] {{.*}}stage = "data_prep"{{.*}} to tensor<8x2xi32>
// PRE: %[[WD:[a-z0-9_]+]] = linalg.generic {{.*}} ins(%[[WS]] : tensor<8x2xi32>) outs(%{{[a-z0-9_]+}} : tensor<8x2xf32>) {{.*}}stage = "data_prep"
// PRE-NEXT: ^bb0
// PRE-NEXT: arith.sitofp
// PRE: linalg.generic {{.*}} ins(%[[WD]] : tensor<8x2xf32>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2xf32>) {{.*}}stage = "compute"
// PRE-NEXT: ^bb0
// PRE-NEXT: linalg.yield
// OTF: linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<4x2x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi8>) {{.*}}stage = "compute"
// OTF-NEXT: ^bb0
// OTF-NEXT: linalg.yield
// OTF: %[[PA:[a-z0-9_]+]] = linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi32>) {{.*}}stage = "compute"
// OTF-NEXT: ^bb0
// OTF-NEXT: arith.extsi
// OTF: linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<8x2x33xi8>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x33xi8>) {{.*}}stage = "compute"
// OTF-NEXT: ^bb0
// OTF-NEXT: linalg.yield
// OTF: %[[P:[a-z0-9_]+]] = linalg.generic {{.*}} ins(%{{[a-z0-9_]+}} : tensor<4x8x2x33xi8>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x33xi32>) {{.*}}stage = "compute"
// OTF-NEXT: ^bb0
// OTF-NEXT: arith.extsi
// OTF: %[[WV:[a-z0-9_]+]] = tensor.extract_slice %[[P]][0, 0, 0, 1] [4, 8, 2, 32] [1, 1, 1, 1] {{.*}}stage = "compute"
// OTF: %[[WS:[a-z0-9_]+]] = tensor.extract_slice %[[P]][0, 0, 0, 0] [4, 8, 2, 1] [1, 1, 1, 1] {{.*}}stage = "compute"{{.*}} to tensor<4x8x2xi32>
// OTF: %[[WD:[a-z0-9_]+]] = linalg.generic {{.*}} ins(%[[WS]] : tensor<4x8x2xi32>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2xf32>) {{.*}}stage = "compute"
// OTF-NEXT: ^bb0
// OTF-NEXT: arith.sitofp
// OTF: linalg.generic {{.*}} ins(%[[PA]], %[[WV]] : tensor<4x8x2x32xi32>, tensor<4x8x2x32xi32>)
// OTF-NEXT: ^bb0
// OTF-NEXT: arith.muli
// OTF: linalg.generic {{.*}} ins(%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %[[WD]] : tensor<4x8x2xi32>, tensor<4x8x2xf32>, tensor<4x8x2xf32>)
// OTF-NOT: tensor<8x2xf32>
// PRE: linalg.reduce {{.*}} dimensions = [2] {{.*}}qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// PRE-NOT: remark
// OTF: linalg.reduce {{.*}} dimensions = [2] {{.*}}qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// OTF-NOT: remark
func.func @shared_prefix(%aq: tensor<4x64xi8>, %wq: tensor<8x66xi8>, %ad: tensor<4x2xf32>) -> tensor<4x8xf32> {
  %ae = tensor.expand_shape %aq [[0], [1, 2]] output_shape [4, 2, 32] : tensor<4x64xi8> into tensor<4x2x32xi8>
  %ace = tensor.empty() : tensor<4x2x32xi32>
  %acast = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%ae : tensor<4x2x32xi8>) outs(%ace : tensor<4x2x32xi32>) {
  ^bb0(%i: i8, %o: i32):
    %x = arith.extsi %i : i8 to i32
    linalg.yield %x : i32
  } -> tensor<4x2x32xi32>
  %we = tensor.expand_shape %wq [[0], [1, 2]] output_shape [8, 2, 33] : tensor<8x66xi8> into tensor<8x2x33xi8>
  %pfe = tensor.empty() : tensor<8x2x33xi32>
  %prefix = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%we : tensor<8x2x33xi8>) outs(%pfe : tensor<8x2x33xi32>) {
  ^bb0(%i: i8, %o: i32):
    %x = arith.extsi %i : i8 to i32
    linalg.yield %x : i32
  } -> tensor<8x2x33xi32>
  %wv = tensor.extract_slice %prefix[0, 0, 1] [8, 2, 32] [1, 1, 1] : tensor<8x2x33xi32> to tensor<8x2x32xi32>
  %ws = tensor.extract_slice %prefix[0, 0, 0] [8, 2, 1] [1, 1, 1] : tensor<8x2x33xi32> to tensor<8x2x1xi32>
  %wsc = tensor.collapse_shape %ws [[0], [1, 2]] : tensor<8x2x1xi32> into tensor<8x2xi32>
  %wde = tensor.empty() : tensor<8x2xf32>
  %wd = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%wsc : tensor<8x2xi32>) outs(%wde : tensor<8x2xf32>) {
  ^bb0(%i: i32, %o: f32):
    %x = arith.sitofp %i : i32 to f32
    linalg.yield %x : f32
  } -> tensor<8x2xf32>
  %pe = tensor.empty() : tensor<4x8x2x32xi32>
  %prod = linalg.generic {indexing_maps = [#a4, #w4, #id4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%acast, %wv : tensor<4x2x32xi32>, tensor<8x2x32xi32>) outs(%pe : tensor<4x8x2x32xi32>) {
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
