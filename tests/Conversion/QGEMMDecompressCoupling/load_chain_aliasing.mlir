// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %s 2>&1 | FileCheck %s --check-prefixes=CHECK,OTF
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre %s 2>&1 | FileCheck %s --check-prefixes=CHECK,PRE
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize %s -o %t.ref
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly %s | %python %S/Inputs/load_chain_check.py --same %t.ref | FileCheck %s --check-prefix=SAME
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre %s | %python %S/Inputs/load_chain_check.py --same %t.ref | FileCheck %s --check-prefix=SAME
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#a4 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#w4 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#a3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#w3 = affine_map<(d0, d1, d2) -> (d1, d2)>
//
// Req 12.22, 14.7, 14.8: forwarding a load chain's private copy turns it into
// a direct read of the function argument, which is only sound when the
// function writes no memref derived from that argument. Here %arg0 feeds the
// weight load chain and, through %wv, the target of a second memref.copy, so
// Load_Copy_Forwarding keeps the chain's alloc and copy under both modes,
// silently, and buddy-opt exits 0 (a non-zero exit would leave FileCheck
// without the function).
//
// The kept memref.copy is an ordinary op of Omega(r). Under on-the-fly it
// cannot take 'm' and gets one Remark R3 for it; the bufferization.to_tensor
// of the alloc cannot take 'm' either (only a memref.reinterpret_cast has an
// insert rule, Req 7.10) and gets its own R3. The view op therefore gains no
// stride-0 dimension, and the weight nibble decode is lifted into m through a
// broadcast of the to_tensor result instead. Under pre nothing is reported.
// The view lacks 'n' in no mode: 'n' is the weight's own dimension.
//
// OTF-DAG: :59:9: remark: qgemm-decompress-coupling: 'm' cannot be inserted into op; kept outside the 'm' space
// OTF-DAG: :58:3: remark: qgemm-decompress-coupling: 'm' cannot be inserted into op; kept outside the 'm' space
// OTF-NOT: remark
// PRE-NOT: remark
// CHECK-LABEL: func.func @aliasing
// The chain keeps the recognizer's form: a 2-D view, a private alloc, the
// copy filling it and a writable to_tensor.
// CHECK: memref.reinterpret_cast %{{[a-z0-9_]+}} to offset: [0], sizes: [8, 64], strides: [64, 1] {{.*}}stage = "data_prep"{{.*}} : memref<*xi8> to memref<8x64xi8, strided<[64, 1]>>
// CHECK: memref.alloc() {{.*}}stage = "data_prep"
// CHECK: memref.copy %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}} {{.*}}stage = "data_prep"{{.*}} : memref<8x64xi8, strided<[64, 1]>> to memref<8x64xi8>
// CHECK: %[[WQ:[a-z0-9_]+]] = bufferization.to_tensor %{{[a-z0-9_]+}} restrict writable {{.*}}stage = "data_prep"{{.*}} : memref<8x64xi8> to tensor<8x64xi8>
// Under on-the-fly the operator around it is rebuilt: the m broadcast sits
// on the load edge.
// OTF: %[[WB:[a-z0-9_]+]] = linalg.generic {{.*}}ins(%[[WQ]] : tensor<8x64xi8>) outs(%{{[a-z0-9_]+}} : tensor<4x8x64xi8>) {{.*}}stage = "compute"
// Under pre the weight decode stays in (n,k) space.
// PRE: tensor.expand_shape %[[WQ]] {{.*}} into tensor<8x2x32xi8>
// CHECK: linalg.reduce {{.*}}qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}
// CHECK-NOT: remark
//
// The chain signatures match the recognizer output exactly in both modes.
// SAME-NOT: VIOLATION
// SAME: LOAD CHAINS SAME {{[1-9][0-9]*}}
func.func @aliasing(%arg0: memref<*xi8>, %aq: tensor<4x64xi8>, %ad: tensor<4x2xf32>, %wd: tensor<8x2xf32>) -> tensor<4x8xf32> {
  %c15 = arith.constant 15 : i32
  %c0 = arith.constant 0 : i32
  %f0 = arith.constant 0.0 : f32
  // The weight load chain.
  %view = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8, 64], strides: [64, 1] : memref<*xi8> to memref<8x64xi8, strided<[64, 1]>>
  %alloc = memref.alloc() : memref<8x64xi8>
  memref.copy %view, %alloc : memref<8x64xi8, strided<[64, 1]>> to memref<8x64xi8>
  %wq = bufferization.to_tensor %alloc restrict writable : memref<8x64xi8> to tensor<8x64xi8>
  // A second view of %arg0, written by the copy below. This is the aliasing
  // write that blocks the copy-drop; neither view op below is reachable from
  // a bufferization.to_tensor, so neither belongs to a load chain.
  %wv = memref.reinterpret_cast %arg0 to offset: [512], sizes: [8, 64], strides: [64, 1] : memref<*xi8> to memref<8x64xi8, strided<[64, 1], offset: 512>>
  %rv = memref.reinterpret_cast %arg0 to offset: [256], sizes: [8, 64], strides: [64, 1] : memref<*xi8> to memref<8x64xi8, strided<[64, 1], offset: 256>>
  memref.copy %rv, %wv : memref<8x64xi8, strided<[64, 1], offset: 256>> to memref<8x64xi8, strided<[64, 1], offset: 512>>

  %ae = tensor.expand_shape %aq [[0], [1, 2]] output_shape [4, 2, 32] : tensor<4x64xi8> into tensor<4x2x32xi8>
  %ace = tensor.empty() : tensor<4x2x32xi32>
  %acast = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%ae : tensor<4x2x32xi8>) outs(%ace : tensor<4x2x32xi32>) {
  ^bb0(%i: i8, %o: i32):
    %x = arith.extsi %i : i8 to i32
    linalg.yield %x : i32
  } -> tensor<4x2x32xi32>
  %we = tensor.expand_shape %wq [[0], [1, 2]] output_shape [8, 2, 32] : tensor<8x64xi8> into tensor<8x2x32xi8>
  %loe = tensor.empty() : tensor<8x2x32xi32>
  %lo = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%we : tensor<8x2x32xi8>) outs(%loe : tensor<8x2x32xi32>) {
  ^bb0(%i: i8, %o: i32):
    %x = arith.extsi %i : i8 to i32
    %lo_v = arith.andi %x, %c15 : i32
    linalg.yield %lo_v : i32
  } -> tensor<8x2x32xi32>
  %pe = tensor.empty() : tensor<4x8x2x32xi32>
  %prod = linalg.generic {indexing_maps = [#a4, #w4, #id4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%acast, %lo : tensor<4x2x32xi32>, tensor<8x2x32xi32>) outs(%pe : tensor<4x8x2x32xi32>) {
  ^bb0(%x: i32, %y: i32, %o: i32):
    %z = arith.muli %x, %y : i32
    linalg.yield %z : i32
  } -> tensor<4x8x2x32xi32>
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
