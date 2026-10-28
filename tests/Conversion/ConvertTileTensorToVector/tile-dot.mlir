// RUN: buddy-opt %s -set-llvm-target="chip=spacemit-x60" -convert-tile-tensor-to-vector=contract-k1-tiles -canonicalize -split-input-file | FileCheck %s
// RUN: buddy-opt %s -set-llvm-target="chip=spacemit-a100" -convert-tile-tensor-to-vector=contract-k1-tiles -split-input-file -remarks-filter-missed=tile-to-vector -o /dev/null 2>&1 | FileCheck %s --check-prefix=NODOT
// RUN: buddy-opt %s -set-llvm-target="triple=x86_64-unknown-linux-gnu chip=icelake-server" -convert-tile-tensor-to-vector=contract-k1-tiles -split-input-file -remarks-filter-missed=tile-to-vector -o /dev/null 2>&1 | FileCheck %s --check-prefix=NODOT
// RUN: buddy-opt %s -set-llvm-target="chip=spacemit-x60" -convert-tile-tensor-to-vector -split-input-file -remarks-filter-missed=tile-to-vector -o /dev/null 2>&1 | FileCheck %s --check-prefix=NODOT
//
// A named K1 supplies a minimum VLEN of 256, not an exact VLEN.
// The fixed K1 tile specialization must remain disabled without exact VLEN.

#map = affine_map<(d0, d1) -> (d0, d1)>

// The add into the running tile folds into the contraction (upstream
// canonicalization).
// NODOT: [Missed] component | Category:tile-to-vector | Reason="'linalg.matmul' has no vector form here"
// CHECK-LABEL: func.func @accumulate
// CHECK: linalg.matmul
// CHECK-NOT: vector.contract
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
// CHECK: linalg.matmul
// CHECK-NOT: vector.contract
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
