// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --mlir-print-op-generic --mlir-print-local-scope %s -o %t.ref
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=pre --mlir-print-op-generic --mlir-print-local-scope %s -o %t.pre 2> %t.pre.err
// RUN: %python %S/Inputs/topo_equal.py %t.ref %t.pre
// RUN: FileCheck %s < %t.pre.err
// RUN: buddy-opt --qgemm-simplify-dim-transforms --qgemm-recognize --qgemm-decompress-coupling=decompress=on-the-fly --mlir-print-op-generic --mlir-print-local-scope %s -o %t.otf 2> %t.otf.err
// RUN: %python %S/Inputs/topo_equal.py %t.ref %t.otf
// RUN: FileCheck %s < %t.otf.err
//
// Property 13, R1 (Req 2.5, 12.6). The multiply reads the activation through a
// non-zero constant map result (d0, 1, d2, d3): the activation cast carries an
// extent-2 axis at the n position and the multiply picks slice 1 of it. That
// introduction of 'n' is outside Req 2.1-2.4, so the root is skipped. The
// output of each RUN with the coupling pass is compared with the Topology_Checker against the output of
// the same pipeline without it, so the root must be Topologically_Identical to it; the
// buddy-opt invocations must exit 0, and stderr carries exactly one R1 and no
// R2/R3.
// CHECK-NOT: cannot be stripped
// CHECK-NOT: cannot be inserted
// CHECK: remark: qgemm-decompress-coupling: unsupported introduction of parallel dimension 'n' on operand #0 (non-zero constant map result); root 0 left unchanged
// CHECK-NOT: cannot be stripped
// CHECK-NOT: cannot be inserted
// CHECK-NOT: remark:
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#a4 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#w4 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#a3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#w3 = affine_map<(d0, d1, d2) -> (d1, d2)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#ua = affine_map<(d0, d1, d2, d3) -> (d0, 1, d2, d3)>
#ac = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
#ub = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
func.func @r1_nonzero_constant(%aq: tensor<4x64xi8>, %wq: tensor<8x64xi8>, %ad: tensor<4x2xf32>, %wd: tensor<8x2xf32>) -> tensor<4x8xf32> {
  %ae = tensor.expand_shape %aq [[0, 1], [2, 3]] output_shape [4, 1, 2, 32] : tensor<4x64xi8> into tensor<4x1x2x32xi8>
  %ace = tensor.empty() : tensor<4x2x2x32xi32>
  %acast = linalg.generic {indexing_maps = [#ac, #id4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%ae : tensor<4x1x2x32xi8>) outs(%ace : tensor<4x2x2x32xi32>) {
  ^bb0(%i: i8, %o: i32):
    %x = arith.extsi %i : i8 to i32
    linalg.yield %x : i32
  } -> tensor<4x2x2x32xi32>
  %we = tensor.expand_shape %wq [[0, 1], [2, 3]] output_shape [1, 8, 2, 32] : tensor<8x64xi8> into tensor<1x8x2x32xi8>
  %wce = tensor.empty() : tensor<1x8x2x32xi32>
  %wcast = linalg.generic {indexing_maps = [#id4, #id4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%we : tensor<1x8x2x32xi8>) outs(%wce : tensor<1x8x2x32xi32>) {
  ^bb0(%i: i8, %o: i32):
    %x = arith.extsi %i : i8 to i32
    linalg.yield %x : i32
  } -> tensor<1x8x2x32xi32>
  %pe = tensor.empty() : tensor<4x8x2x32xi32>
  %prod = linalg.generic {indexing_maps = [#ua, #ub, #id4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%acast, %wcast : tensor<4x2x2x32xi32>, tensor<1x8x2x32xi32>) outs(%pe : tensor<4x8x2x32xi32>) {
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
