// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre %s 2>&1 | FileCheck %s --check-prefix=PRE
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %s 2>&1 | FileCheck %s --check-prefix=OTF
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre %s | FileCheck %s --check-prefix=PRES
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %s | FileCheck %s --check-prefix=OTFS
//
// Outside_Users interleaved with Omega(r) (Req 6.9). The decompressed weight
// value %wcast and the partially decoded factor %f2 are each consumed by an
// unannotated func.call placed between ops of Omega(r), before the multiply
// and before the rest of the factor chain. The rebuild must still dominate
// every use: the verifier runs after the pass, no remark is emitted, and each
// call keeps the operand type it had in the input (Req 6.7).
//
// PRES / OTFS tie each call operand to its defining op through captured SSA
// names and check the per-mode clone and broadcast shapes and stages. The
// recognize-only output keeps the multiply's folded broadcast maps and has no
// Pure_Broadcast, so it fails both; each mode fails the other's checks
// (under OTF the factor sitofp and the activation extsi carry the other dim,
// under PRE they do not). The ops with an Outside_User (%wcast, %f2) gain no
// dim in either mode; their broadcast is created right after them, ahead of
// the call (Req 6.9).
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#a4 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#w4 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#a3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#w3 = affine_map<(d0, d1, d2) -> (d1, d2)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
// PRE-NOT: remark
// OTF-NOT: remark
// PRE-LABEL: func.func @interleaved
// OTF-LABEL: func.func @interleaved
// Both calls keep their original operand types.
// PRE-DAG: call @sink_w(%{{[a-z0-9_]+}}) : (tensor<8x2x32xi32>) -> ()
// PRE-DAG: call @sink_f(%{{[a-z0-9_]+}}) : (tensor<8x2xi32>) -> ()
// PRE: linalg.reduce {{.*}} dimensions = [2] {{.*}}qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// OTF-DAG: call @sink_w(%{{[a-z0-9_]+}}) : (tensor<8x2x32xi32>) -> ()
// OTF-DAG: call @sink_f(%{{[a-z0-9_]+}}) : (tensor<8x2xi32>) -> ()
// OTF: linalg.reduce {{.*}} dimensions = [2] {{.*}}qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// PRE-NOT: remark
// OTF-NOT: remark
// PRES-LABEL: func.func @interleaved
// PRES: %[[WC:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%{{[a-z0-9_]+}} : tensor<8x2x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<8x2x32xi32>) {{.*}}stage = "data_prep"
// PRES: %[[WCM:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%[[WC]] : tensor<8x2x32xi32>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi32>) {{.*}}stage = "compute"
// PRES: call @sink_w(%[[WC]]) : (tensor<8x2x32xi32>) -> ()
// PRES: %[[F2:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%{{[a-z0-9_]+}} : tensor<8x2xi32>) outs(%{{[a-z0-9_]+}} : tensor<8x2xi32>) {{.*}}stage = "data_prep"
// PRES: call @sink_f(%[[F2]]) : (tensor<8x2xi32>) -> ()
// PRES: %[[WD:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%[[F2]] : tensor<8x2xi32>) outs(%{{[a-z0-9_]+}} : tensor<8x2xf32>) {{.*}}stage = "data_prep"
// PRES: %[[WDM:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%[[WD]] : tensor<8x2xf32>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2xf32>) {{.*}}stage = "compute"
// PRES: %[[AC:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%{{[a-z0-9_]+}} : tensor<4x2x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<4x2x32xi32>) {{.*}}stage = "data_prep"
// PRES: %[[ACN:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%[[AC]] : tensor<4x2x32xi32>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi32>) {{.*}}stage = "compute"
// PRES: linalg.generic {{.*}}ins(%[[ACN]], %[[WCM]] : tensor<4x8x2x32xi32>, tensor<4x8x2x32xi32>) {{.*}}stage = "compute"
// PRES: linalg.generic {{.*}}ins(%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %[[WDM]] : tensor<4x8x2xi32>, tensor<4x8x2xf32>, tensor<4x8x2xf32>) {{.*}}stage = "compute"
// PRES: qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// OTFS-LABEL: func.func @interleaved
// OTFS: %[[WC:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%{{[a-z0-9_]+}} : tensor<8x2x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<8x2x32xi32>) {{.*}}stage = "data_prep"
// OTFS: %[[WCM:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%[[WC]] : tensor<8x2x32xi32>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi32>) {{.*}}stage = "compute"
// OTFS: call @sink_w(%[[WC]]) : (tensor<8x2x32xi32>) -> ()
// OTFS: %[[F2:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%{{[a-z0-9_]+}} : tensor<8x2xi32>) outs(%{{[a-z0-9_]+}} : tensor<8x2xi32>) {{.*}}stage = "data_prep"
// OTFS: %[[F2M:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%[[F2]] : tensor<8x2xi32>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2xi32>) {{.*}}stage = "compute"
// OTFS: call @sink_f(%[[F2]]) : (tensor<8x2xi32>) -> ()
// OTFS: %[[WD:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%[[F2M]] : tensor<4x8x2xi32>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2xf32>) {{.*}}stage = "compute"
// OTFS: %[[AQN:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%{{[a-z0-9_]+}} : tensor<4x2x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi8>) {{.*}}stage = "compute"
// OTFS: %[[AC:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%[[AQN]] : tensor<4x8x2x32xi8>) outs(%{{[a-z0-9_]+}} : tensor<4x8x2x32xi32>) {{.*}}stage = "compute"
// OTFS: linalg.generic {{.*}}ins(%[[AC]], %[[WCM]] : tensor<4x8x2x32xi32>, tensor<4x8x2x32xi32>) {{.*}}stage = "compute"
// OTFS: linalg.generic {{.*}}ins(%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %[[WD]] : tensor<4x8x2xi32>, tensor<4x8x2xf32>, tensor<4x8x2xf32>) {{.*}}stage = "compute"
// OTFS: qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
func.func private @sink_w(tensor<8x2x32xi32>)
func.func private @sink_f(tensor<8x2xi32>)
func.func @interleaved(%aq: tensor<4x64xi8>, %wq: tensor<8x64xi8>, %ad: tensor<4x2xf32>,
                       %wsc: tensor<8x2xi8>) -> tensor<4x8xf32> {
  %we = tensor.expand_shape %wq [[0], [1, 2]] output_shape [8, 2, 32] : tensor<8x64xi8> into tensor<8x2x32xi8>
  %wce = tensor.empty() : tensor<8x2x32xi32>
  %wcast = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%we : tensor<8x2x32xi8>) outs(%wce : tensor<8x2x32xi32>) {
  ^bb0(%i: i8, %o: i32):
    %x = arith.extsi %i : i8 to i32
    linalg.yield %x : i32
  } -> tensor<8x2x32xi32>
  // Outside_User of a value-chain op, before later ops of Omega(r).
  call @sink_w(%wcast) : (tensor<8x2x32xi32>) -> ()
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
    %c63 = arith.constant 63 : i32
    %x = arith.andi %i, %c63 : i32
    linalg.yield %x : i32
  } -> tensor<8x2xi32>
  // Outside_User of a factor-chain op, before the rest of the chain.
  call @sink_f(%f2) : (tensor<8x2xi32>) -> ()
  %wde = tensor.empty() : tensor<8x2xf32>
  %wd = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%f2 : tensor<8x2xi32>) outs(%wde : tensor<8x2xf32>) {
  ^bb0(%i: i32, %o: f32):
    %x = arith.sitofp %i : i32 to f32
    linalg.yield %x : f32
  } -> tensor<8x2xf32>
  %ae = tensor.expand_shape %aq [[0], [1, 2]] output_shape [4, 2, 32] : tensor<4x64xi8> into tensor<4x2x32xi8>
  %ace = tensor.empty() : tensor<4x2x32xi32>
  %acast = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%ae : tensor<4x2x32xi8>) outs(%ace : tensor<4x2x32xi32>) {
  ^bb0(%i: i8, %o: i32):
    %x = arith.extsi %i : i8 to i32
    linalg.yield %x : i32
  } -> tensor<4x2x32xi32>
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
