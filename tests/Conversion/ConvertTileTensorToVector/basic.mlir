// RUN: buddy-opt %s -set-llvm-target="chip=spacemit-x60" -convert-tile-tensor-to-vector -canonicalize -split-input-file | FileCheck %s
// RUN: buddy-opt %s -set-llvm-target="triple=x86_64-unknown-linux-gnu chip=x86-64" -convert-tile-tensor-to-vector -canonicalize -split-input-file | FileCheck %s
// RUN: buddy-opt %s -set-llvm-target="chip=spacemit-x60" -convert-tile-tensor-to-vector -split-input-file -remarks-filter=tile-to-vector -o /dev/null 2>&1 | FileCheck %s --check-prefix=REMARK
// RUN: not buddy-opt %s -convert-tile-tensor-to-vector -split-input-file -o /dev/null 2>&1 | FileCheck %s --check-prefix=NOTARGET
//
// Register-sized tensor components become vectors of the same shape; memory
// is reached through vector transfers, and how they map onto registers is
// left to the vector lowering. A component that touches an op without a
// vector form stays in tensors, with a remark.

#map = affine_map<(d0, d1) -> (d0, d1)>

// A (4,8) i8 tile: the shape is kept, and only the types change.
// CHECK-LABEL: func.func @contiguous
// CHECK: %[[X:.+]] = vector.transfer_read {{.*}} {in_bounds = [true, true]} : memref<4x8xi8>, vector<4x8xi8>
// CHECK: %[[Y:.+]] = arith.andi %[[X]], {{.*}} : vector<4x8xi8>
// CHECK: vector.transfer_write %[[Y]], {{.*}} {in_bounds = [true, true]} : vector<4x8xi8>, memref<4x8xi8>
// CHECK-NOT: bufferization
// CHECK-NOT: linalg.generic
// CHECK-NOT: memref.collapse_shape
// REMARK: [Passed] component | Category:tile-to-vector |{{.*}}converted to vectors
// NOTARGET: error
func.func @contiguous(%src: memref<4x8xi8>, %dst: memref<4x8xi8>) {
  %c15 = arith.constant 15 : i8
  %e = tensor.empty() : tensor<4x8xi8>
  %m = linalg.fill ins(%c15 : i8) outs(%e : tensor<4x8xi8>) -> tensor<4x8xi8>
  %x = bufferization.to_tensor %src restrict : memref<4x8xi8> to tensor<4x8xi8>
  %y = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%x, %m : tensor<4x8xi8>, tensor<4x8xi8>) outs(%e : tensor<4x8xi8>) {
  ^bb0(%a: i8, %b: i8, %o: i8):
    %r = arith.andi %a, %b : i8
    linalg.yield %r : i8
  } -> tensor<4x8xi8>
  bufferization.materialize_in_destination %y in writable %dst : (tensor<4x8xi8>, memref<4x8xi8>) -> ()
  return
}

// -----

// A strided destination is one n-D transfer; the conversion does not split it.
// CHECK-LABEL: func.func @strided_store
// CHECK: vector.transfer_write {{.*}} : vector<4x4xi32>, memref<4x4xi32, strided<[16, 1]>>
// CHECK-NOT: vector.transfer_write
// CHECK-NOT: materialize_in_destination
func.func @strided_store(%dst: memref<4x4xi32, strided<[16, 1]>>) {
  %c1 = arith.constant 1 : i32
  %e = tensor.empty() : tensor<4x4xi32>
  %f = linalg.fill ins(%c1 : i32) outs(%e : tensor<4x4xi32>) -> tensor<4x4xi32>
  bufferization.materialize_in_destination %f in writable %dst : (tensor<4x4xi32>, memref<4x4xi32, strided<[16, 1]>>) -> ()
  return
}

// -----

// A reshape is a vector.shape_cast.
// CHECK-LABEL: func.func @reshape
// CHECK: vector.transfer_read {{.*}} vector<32xi8>
// CHECK: vector.shape_cast {{.*}} : vector<32xi8> to vector<4x8xi8>
// CHECK: vector.transfer_write {{.*}} vector<4x8xi8>
func.func @reshape(%src: memref<32xi8>, %dst: memref<4x8xi8>) {
  %x = bufferization.to_tensor %src restrict : memref<32xi8> to tensor<32xi8>
  %y = tensor.expand_shape %x [[0, 1]] output_shape [4, 8] : tensor<32xi8> into tensor<4x8xi8>
  bufferization.materialize_in_destination %y in writable %dst : (tensor<4x8xi8>, memref<4x8xi8>) -> ()
  return
}

// -----

// Loop-carried tiles become vector iter_args.
// CHECK-LABEL: func.func @loop
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (vector<16xf32>)
// CHECK: arith.addf {{.*}} : vector<16xf32>
// CHECK: vector.transfer_write {{.*}} vector<16xf32>
#map1 = affine_map<(d0) -> (d0)>
func.func @loop(%src: memref<16xf32>, %dst: memref<16xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %z = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<16xf32>
  %init = linalg.fill ins(%z : f32) outs(%e : tensor<16xf32>) -> tensor<16xf32>
  %res = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> (tensor<16xf32>) {
    %x = bufferization.to_tensor %src restrict : memref<16xf32> to tensor<16xf32>
    %s = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]}
        ins(%acc, %x : tensor<16xf32>, tensor<16xf32>) outs(%e : tensor<16xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %r = arith.addf %a, %b : f32
      linalg.yield %r : f32
    } -> tensor<16xf32>
    scf.yield %s : tensor<16xf32>
  }
  bufferization.materialize_in_destination %res in writable %dst : (tensor<16xf32>, memref<16xf32>) -> ()
  return
}

// -----

// A component reaching linalg.matmul stays in tensors.
// CHECK-LABEL: func.func @blocked
// CHECK-NOT: vector.transfer_read
// CHECK: linalg.matmul
// REMARK: [Missed] component | Category:tile-to-vector | Reason="'linalg.matmul' has no vector form here"
func.func @blocked(%a: memref<4x4xf32>, %b: memref<4x4xf32>, %c: memref<4x4xf32>) {
  %z = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<4x4xf32>
  %acc = linalg.fill ins(%z : f32) outs(%e : tensor<4x4xf32>) -> tensor<4x4xf32>
  %x = bufferization.to_tensor %a restrict : memref<4x4xf32> to tensor<4x4xf32>
  %y = bufferization.to_tensor %b restrict : memref<4x4xf32> to tensor<4x4xf32>
  %m = linalg.matmul ins(%x, %y : tensor<4x4xf32>, tensor<4x4xf32>) outs(%acc : tensor<4x4xf32>) -> tensor<4x4xf32>
  bufferization.materialize_in_destination %m in writable %c : (tensor<4x4xf32>, memref<4x4xf32>) -> ()
  return
}
