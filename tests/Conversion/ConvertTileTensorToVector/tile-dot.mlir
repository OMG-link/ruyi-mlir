// RUN: buddy-opt %s -set-llvm-target="chip=spacemit-x60 vlen=256" -convert-tile-tensor-to-vector=contract-k1-tiles -canonicalize -split-input-file | FileCheck %s
// RUN: buddy-opt %s -set-llvm-target="chip=spacemit-a100" -convert-tile-tensor-to-vector=contract-k1-tiles -split-input-file -remarks-filter-missed=tile-to-vector -o /dev/null 2>&1 | FileCheck %s --check-prefix=NODOT
// RUN: buddy-opt %s -set-llvm-target="triple=x86_64-unknown-linux-gnu chip=icelake-server" -convert-tile-tensor-to-vector=contract-k1-tiles -split-input-file -remarks-filter-missed=tile-to-vector -o /dev/null 2>&1 | FileCheck %s --check-prefix=NODOT
// RUN: buddy-opt %s -set-llvm-target="chip=spacemit-x60 vlen=256" -convert-tile-tensor-to-vector -split-input-file -remarks-filter-missed=tile-to-vector -o /dev/null 2>&1 | FileCheck %s --check-prefix=NODOT
//
// With contract-k1-tiles, on the K1 (XSMTVDot, VLEN = 256),
// matmul(A, transpose(B)) on 4x8 int8 tiles into a 4x4 int32 tile becomes
// vector.transpose plus vector.contract on extended operands. Targets without the K1 tile dot, or pipelines that do not set
// the option, keep the matmul in tensors.

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @accumulate
// CHECK: scf.for {{.*}} iter_args(%[[ACC:[a-z0-9_]+]] = {{.*}}) -> (vector<4x4xi32>)
// CHECK-DAG: %[[A:.+]] = vector.transfer_read {{.*}} vector<4x8xi8>
// CHECK-DAG: %[[B:.+]] = vector.transfer_read {{.*}} vector<4x8xi8>
// CHECK: %[[BT:.+]] = vector.transpose %[[B]], [1, 0] : vector<4x8xi8> to vector<8x4xi8>
// CHECK-DAG: %[[EA:.+]] = arith.extsi %[[A]] : vector<4x8xi8> to vector<4x8xi32>
// CHECK-DAG: %[[EB:.+]] = arith.extsi %[[BT]] : vector<8x4xi8> to vector<8x4xi32>
// The add into the running tile folds into the contraction (upstream
// canonicalization).
// CHECK: %[[D:.+]] = vector.contract {{.*}} %[[EA]], %[[EB]], %[[ACC]] : vector<4x8xi32>, vector<8x4xi32> into vector<4x4xi32>
// CHECK: scf.yield %[[D]]
// CHECK-NOT: linalg.transpose
// CHECK-NOT: linalg.matmul
// NODOT: [Missed] component | Category:tile-to-vector | Reason="'linalg.matmul' has no vector form here"
func.func @accumulate(%a: memref<4x8xi8>, %b: memref<4x8xi8>, %c: memref<4x4xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %zero = arith.constant 0 : i32
  %e = tensor.empty() : tensor<4x4xi32>
  %z = linalg.fill ins(%zero : i32) outs(%e : tensor<4x4xi32>) -> tensor<4x4xi32>
  %res = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %z) -> (tensor<4x4xi32>) {
    %ta = bufferization.to_tensor %a restrict : memref<4x8xi8> to tensor<4x8xi8>
    %tb = bufferization.to_tensor %b restrict : memref<4x8xi8> to tensor<4x8xi8>
    %et = tensor.empty() : tensor<8x4xi8>
    %bt = linalg.transpose ins(%tb : tensor<4x8xi8>) outs(%et : tensor<8x4xi8>) permutation = [1, 0]
    %m = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%ta, %bt : tensor<4x8xi8>, tensor<8x4xi8>) outs(%z : tensor<4x4xi32>) -> tensor<4x4xi32>
    %s = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%acc, %m : tensor<4x4xi32>, tensor<4x4xi32>) outs(%acc : tensor<4x4xi32>) {
    ^bb0(%x: i32, %y: i32, %o: i32):
      %r = arith.addi %x, %y : i32
      linalg.yield %r : i32
    } -> tensor<4x4xi32>
    scf.yield %s : tensor<4x4xi32>
  }
  bufferization.materialize_in_destination %res in writable %c : (tensor<4x4xi32>, memref<4x4xi32>) -> ()
  return
}

// -----

// Unsigned inputs extend with extui.
// CHECK-LABEL: func.func @unsigned
// CHECK: arith.extui
// CHECK: vector.contract
func.func @unsigned(%a: memref<4x8xi8>, %b: memref<4x8xi8>, %c: memref<4x4xi32>) {
  %zero = arith.constant 0 : i32
  %e = tensor.empty() : tensor<4x4xi32>
  %z = linalg.fill ins(%zero : i32) outs(%e : tensor<4x4xi32>) -> tensor<4x4xi32>
  %ta = bufferization.to_tensor %a restrict : memref<4x8xi8> to tensor<4x8xi8>
  %tb = bufferization.to_tensor %b restrict : memref<4x8xi8> to tensor<4x8xi8>
  %et = tensor.empty() : tensor<8x4xi8>
  %bt = linalg.transpose ins(%tb : tensor<4x8xi8>) outs(%et : tensor<8x4xi8>) permutation = [1, 0]
  %m = linalg.matmul {cast = #linalg.type_fn<cast_unsigned>} ins(%ta, %bt : tensor<4x8xi8>, tensor<8x4xi8>) outs(%z : tensor<4x4xi32>) -> tensor<4x4xi32>
  bufferization.materialize_in_destination %m in writable %c : (tensor<4x4xi32>, memref<4x4xi32>) -> ()
  return
}

// -----

// A transpose with a reader other than a K1 tile product blocks the whole
// component.
// CHECK-LABEL: func.func @shared_transpose
// CHECK: linalg.transpose
// CHECK: linalg.matmul
// CHECK-NOT: vector.contract
func.func @shared_transpose(%a: memref<4x8xi8>, %b: memref<4x8xi8>, %c: memref<4x4xi32>, %d: memref<8x4xi8>) {
  %zero = arith.constant 0 : i32
  %e = tensor.empty() : tensor<4x4xi32>
  %z = linalg.fill ins(%zero : i32) outs(%e : tensor<4x4xi32>) -> tensor<4x4xi32>
  %ta = bufferization.to_tensor %a restrict : memref<4x8xi8> to tensor<4x8xi8>
  %tb = bufferization.to_tensor %b restrict : memref<4x8xi8> to tensor<4x8xi8>
  %et = tensor.empty() : tensor<8x4xi8>
  %bt = linalg.transpose ins(%tb : tensor<4x8xi8>) outs(%et : tensor<8x4xi8>) permutation = [1, 0]
  %m = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%ta, %bt : tensor<4x8xi8>, tensor<8x4xi8>) outs(%z : tensor<4x4xi32>) -> tensor<4x4xi32>
  bufferization.materialize_in_destination %m in writable %c : (tensor<4x4xi32>, memref<4x4xi32>) -> ()
  bufferization.materialize_in_destination %bt in writable %d : (tensor<8x4xi8>, memref<8x4xi8>) -> ()
  return
}
