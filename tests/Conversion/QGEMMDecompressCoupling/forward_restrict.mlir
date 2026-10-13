// RUN: buddy-opt --qgemm-decompress-coupling %s 2>&1 | FileCheck %s
//
// Req 12.27, 14.7 (3), 14.14: the forwarded to_tensor keeps `restrict`, so it
// must be the only tensor-IR access to the source memory and its aliases.
// Every other to_tensor and materialize_in_destination counts, evaluated on
// the current IR. No remark is emitted either way.
//
// CHECK-NOT: remark

// Two loads of one argument: whichever the driver visits first is forwarded;
// the other then sees the forwarded restrict to_tensor and keeps its copy.
// CHECK-LABEL: func.func @two_loads_same_arg
// CHECK-COUNT-1: memref.alloc
// CHECK-NOT: memref.alloc
// CHECK-COUNT-1: bufferization.to_tensor %{{[a-z0-9_]+}} restrict :
// CHECK-NOT: bufferization.to_tensor %{{[a-z0-9_]+}} restrict :
// CHECK: return
func.func @two_loads_same_arg(%arg0: memref<*xi8> {llvm.noalias}) -> (tensor<8x64xi8>, tensor<8x64xi8>) {
  %v0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8, 64], strides: [64, 1] : memref<*xi8> to memref<8x64xi8, strided<[64, 1]>>
  %a0 = memref.alloc() : memref<8x64xi8>
  memref.copy %v0, %a0 : memref<8x64xi8, strided<[64, 1]>> to memref<8x64xi8>
  %t0 = bufferization.to_tensor %a0 restrict writable : memref<8x64xi8> to tensor<8x64xi8>
  %v1 = memref.reinterpret_cast %arg0 to offset: [512], sizes: [8, 64], strides: [64, 1] : memref<*xi8> to memref<8x64xi8, strided<[64, 1], offset: 512>>
  %a1 = memref.alloc() : memref<8x64xi8>
  memref.copy %v1, %a1 : memref<8x64xi8, strided<[64, 1], offset: 512>> to memref<8x64xi8>
  %t1 = bufferization.to_tensor %a1 restrict writable : memref<8x64xi8> to tensor<8x64xi8>
  return %t0, %t1 : tensor<8x64xi8>, tensor<8x64xi8>
}

// A materialize_in_destination into the same argument: kept.
// CHECK-LABEL: func.func @materialize_same_arg
// CHECK: memref.alloc
// CHECK: memref.copy
// CHECK: bufferization.to_tensor %{{[a-z0-9_]+}} restrict writable
func.func @materialize_same_arg(%arg0: memref<*xi8> {llvm.noalias}, %src: tensor<8x64xi8>) -> tensor<8x64xi8> {
  %v0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8, 64], strides: [64, 1] : memref<*xi8> to memref<8x64xi8, strided<[64, 1]>>
  %a0 = memref.alloc() : memref<8x64xi8>
  memref.copy %v0, %a0 : memref<8x64xi8, strided<[64, 1]>> to memref<8x64xi8>
  %t0 = bufferization.to_tensor %a0 restrict writable : memref<8x64xi8> to tensor<8x64xi8>
  %out = memref.reinterpret_cast %arg0 to offset: [512], sizes: [8, 64], strides: [64, 1] : memref<*xi8> to memref<8x64xi8, strided<[64, 1], offset: 512>>
  bufferization.materialize_in_destination %src in writable %out : (tensor<8x64xi8>, memref<8x64xi8, strided<[64, 1], offset: 512>>) -> ()
  return %t0 : tensor<8x64xi8>
}

// A to_tensor of another `llvm.noalias` argument cannot alias: both loads are
// forwarded.
// CHECK-LABEL: func.func @other_noalias_arg
// CHECK-NOT: memref.alloc
// CHECK-COUNT-2: bufferization.to_tensor %{{[a-z0-9_]+}} restrict :
func.func @other_noalias_arg(%arg0: memref<*xi8> {llvm.noalias}, %arg1: memref<*xi8> {llvm.noalias}) -> (tensor<8x64xi8>, tensor<8x64xi8>) {
  %v0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8, 64], strides: [64, 1] : memref<*xi8> to memref<8x64xi8, strided<[64, 1]>>
  %a0 = memref.alloc() : memref<8x64xi8>
  memref.copy %v0, %a0 : memref<8x64xi8, strided<[64, 1]>> to memref<8x64xi8>
  %t0 = bufferization.to_tensor %a0 restrict writable : memref<8x64xi8> to tensor<8x64xi8>
  %v1 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [8, 64], strides: [64, 1] : memref<*xi8> to memref<8x64xi8, strided<[64, 1]>>
  %a1 = memref.alloc() : memref<8x64xi8>
  memref.copy %v1, %a1 : memref<8x64xi8, strided<[64, 1]>> to memref<8x64xi8>
  %t1 = bufferization.to_tensor %a1 restrict writable : memref<8x64xi8> to tensor<8x64xi8>
  return %t0, %t1 : tensor<8x64xi8>, tensor<8x64xi8>
}

// A lookup-table `restrict` view of the same argument already exists: kept.
// CHECK-LABEL: func.func @lookup_view_same_arg
// CHECK: memref.alloc
// CHECK: memref.copy
// CHECK: bufferization.to_tensor %{{[a-z0-9_]+}} restrict writable
func.func @lookup_view_same_arg(%arg0: memref<*xi8> {llvm.noalias}, %n: index) -> (tensor<8x64xi8>, tensor<?xi8>) {
  %v0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8, 64], strides: [64, 1] : memref<*xi8> to memref<8x64xi8, strided<[64, 1]>>
  %a0 = memref.alloc() : memref<8x64xi8>
  memref.copy %v0, %a0 : memref<8x64xi8, strided<[64, 1]>> to memref<8x64xi8>
  %t0 = bufferization.to_tensor %a0 restrict writable : memref<8x64xi8> to tensor<8x64xi8>
  %lut = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%n], strides: [1] : memref<*xi8> to memref<?xi8>
  %t1 = bufferization.to_tensor %lut restrict : memref<?xi8> to tensor<?xi8>
  return %t0, %t1 : tensor<8x64xi8>, tensor<?xi8>
}
