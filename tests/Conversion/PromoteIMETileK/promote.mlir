// RUN: buddy-opt %s -set-llvm-target="chip=spacemit-a100 vlen=1024" -promote-ime-tile-k -split-input-file | FileCheck %s
// RUN: buddy-opt %s -set-llvm-target="chip=spacemit-a100 vlen=1024" -promote-ime-tile-k -canonicalize -cse -lower-vector-contract-to-ime-tile -split-input-file | FileCheck %s --check-prefix=TILE
// RUN: buddy-opt %s -set-llvm-target="chip=spacemit-x60 vlen=256" -promote-ime-tile-k -split-input-file | FileCheck %s --check-prefix=K1
//
// A K-short tile contraction reads its operands as masked full-K tiles of
// the same address and contracts over the tile's K, which the IME tile
// lowering then maps onto one dot. Contractions it would not map stay as
// they are.

#mk = affine_map<(m, n, k) -> (m, k)>
#kn = affine_map<(m, n, k) -> (k, n)>
#mn = affine_map<(m, n, k) -> (m, n)>

// The scale outer product, on a view with a dynamic offset.
// CHECK-LABEL: func.func @outer_product
// CHECK-DAG: %[[MA:.+]] = memref.extract_strided_metadata %arg0
// CHECK-DAG: %[[VA:.+]] = memref.reinterpret_cast %{{.+}} to offset: [%{{.+}}], sizes: [8, 8], strides: [8, 1] : memref<f16> to memref<8x8xf16, strided<[8, 1], offset: ?>>
// CHECK-DAG: %[[KA:.+]] = vector.create_mask %c8{{[_0-9]*}}, %c1{{[_0-9]*}} : vector<8x8xi1>
// CHECK-DAG: %[[A:.+]] = vector.transfer_read %[[VA]][%{{.+}}, %{{.+}}], %{{.+}}, %[[KA]] {in_bounds = [true, true]}
// CHECK-DAG: %[[B:.+]] = vector.transfer_read %{{.+}}[%{{.+}}, %{{.+}}], %{{.+}}, %{{.+}} {in_bounds = [true, true]} : memref<8x8xf16, strided<[8, 1], offset: ?>>, vector<8x8xf16>
// CHECK-DAG: %[[BT:.+]] = vector.transpose %[[B]], [1, 0] : vector<8x8xf16> to vector<8x8xf16>
// CHECK-DAG: %[[EA:.+]] = arith.extf %[[A]] : vector<8x8xf16> to vector<8x8xf32>
// CHECK-DAG: %[[EB:.+]] = arith.extf %[[BT]] : vector<8x8xf16> to vector<8x8xf32>
// CHECK: vector.contract {{.+}} %[[EA]], %[[EB]], %{{.+}} : vector<8x8xf32>, vector<8x8xf32> into vector<8x8xf32>
// CHECK-NOT: vector<8x1x
// TILE-LABEL: func.func @outer_product
// TILE: ime.tile.vfwmadot
// TILE-NOT: vector.contract
// The K1 has no fp16 tile.
// K1-LABEL: func.func @outer_product
// K1: vector.contract {{.+}} : vector<8x1xf32>, vector<1x8xf32> into vector<8x8xf32>
func.func @outer_product(%a: memref<8x1xf16, strided<[8, 1], offset: ?>>,
                         %b: memref<8x1xf16, strided<[8, 1], offset: ?>>,
                         %acc: vector<8x8xf32>) -> vector<8x8xf32> {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0.0 : f16
  %ra = vector.transfer_read %a[%c0, %c0], %p {in_bounds = [true, true]} : memref<8x1xf16, strided<[8, 1], offset: ?>>, vector<8x1xf16>
  %rb = vector.transfer_read %b[%c0, %c0], %p {in_bounds = [true, true]} : memref<8x1xf16, strided<[8, 1], offset: ?>>, vector<8x1xf16>
  %bt = vector.transpose %rb, [1, 0] : vector<8x1xf16> to vector<1x8xf16>
  %ea = arith.extf %ra : vector<8x1xf16> to vector<8x1xf32>
  %eb = arith.extf %bt : vector<1x8xf16> to vector<1x8xf32>
  %r = vector.contract {indexing_maps = [#mk, #kn, #mn], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %ea, %eb, %acc : vector<8x1xf32>, vector<1x8xf32> into vector<8x8xf32>
  return %r : vector<8x8xf32>
}

// -----

// int8 with extui on both sides: vmadotu. B is read directly as K x N, its
// N elements the tile's K apart, and one extended A serves two products,
// which share one widened read.
// CHECK-LABEL: func.func @unsigned_direct_b
// CHECK-COUNT-2: vector.transfer_read {{.+}} : memref<8x16xi8, strided<[16, 1], offset: ?>>, vector<8x16xi8>
// CHECK-NOT: vector.transfer_read
// CHECK: vector.transpose {{.+}} : vector<8x16xi8> to vector<16x8xi8>
// CHECK-COUNT-2: vector.contract {{.+}} : vector<8x16xi32>, vector<16x8xi32> into vector<8x8xi32>
// TILE-LABEL: func.func @unsigned_direct_b
// TILE-COUNT-2: ime.tile.vmadotu
// TILE-NOT: vector.contract
#mk = affine_map<(m, n, k) -> (m, k)>
#kn = affine_map<(m, n, k) -> (k, n)>
#mn = affine_map<(m, n, k) -> (m, n)>
func.func @unsigned_direct_b(%a: memref<8x2xi8, strided<[16, 1]>>,
                             %b: memref<2x8xi8, strided<[1, 16]>>,
                             %acc: vector<8x8xi32>) -> vector<8x8xi32> {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0 : i8
  %ra = vector.transfer_read %a[%c0, %c0], %p {in_bounds = [true, true]} : memref<8x2xi8, strided<[16, 1]>>, vector<8x2xi8>
  %rb = vector.transfer_read %b[%c0, %c0], %p {in_bounds = [true, true]} : memref<2x8xi8, strided<[1, 16]>>, vector<2x8xi8>
  %ea = arith.extui %ra : vector<8x2xi8> to vector<8x2xi32>
  %eb = arith.extui %rb : vector<2x8xi8> to vector<2x8xi32>
  %r = vector.contract {indexing_maps = [#mk, #kn, #mn], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %ea, %eb, %acc : vector<8x2xi32>, vector<2x8xi32> into vector<8x8xi32>
  %s = vector.contract {indexing_maps = [#mk, #kn, #mn], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %ea, %eb, %r : vector<8x2xi32>, vector<2x8xi32> into vector<8x8xi32>
  return %s : vector<8x8xi32>
}

// -----

// Mixed signedness has no tile dot.
// CHECK-LABEL: func.func @mixed_extension
// CHECK: vector.contract {{.+}} : vector<8x2xi32>, vector<2x8xi32> into vector<8x8xi32>
// TILE-LABEL: func.func @mixed_extension
// TILE: vector.contract
#mk = affine_map<(m, n, k) -> (m, k)>
#kn = affine_map<(m, n, k) -> (k, n)>
#mn = affine_map<(m, n, k) -> (m, n)>
func.func @mixed_extension(%a: memref<8x2xi8, strided<[16, 1]>>,
                           %b: memref<8x2xi8, strided<[16, 1]>>,
                           %acc: vector<8x8xi32>) -> vector<8x8xi32> {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0 : i8
  %ra = vector.transfer_read %a[%c0, %c0], %p {in_bounds = [true, true]} : memref<8x2xi8, strided<[16, 1]>>, vector<8x2xi8>
  %rb = vector.transfer_read %b[%c0, %c0], %p {in_bounds = [true, true]} : memref<8x2xi8, strided<[16, 1]>>, vector<8x2xi8>
  %bt = vector.transpose %rb, [1, 0] : vector<8x2xi8> to vector<2x8xi8>
  %ea = arith.extsi %ra : vector<8x2xi8> to vector<8x2xi32>
  %eb = arith.extui %bt : vector<2x8xi8> to vector<2x8xi32>
  %r = vector.contract {indexing_maps = [#mk, #kn, #mn], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %ea, %eb, %acc : vector<8x2xi32>, vector<2x8xi32> into vector<8x8xi32>
  return %r : vector<8x8xi32>
}

// -----

// The A read has another reader, which a widened read would change.
// CHECK-LABEL: func.func @other_reader
// CHECK: vector.contract {{.+}} : vector<8x1xf32>, vector<1x8xf32> into vector<8x8xf32>
#mk = affine_map<(m, n, k) -> (m, k)>
#kn = affine_map<(m, n, k) -> (k, n)>
#mn = affine_map<(m, n, k) -> (m, n)>
func.func @other_reader(%a: memref<8x1xf16, strided<[8, 1]>>,
                        %b: memref<8x1xf16, strided<[8, 1]>>,
                        %acc: vector<8x8xf32>) -> (vector<8x8xf32>, vector<8x1xf32>) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0.0 : f16
  %ra = vector.transfer_read %a[%c0, %c0], %p {in_bounds = [true, true]} : memref<8x1xf16, strided<[8, 1]>>, vector<8x1xf16>
  %rb = vector.transfer_read %b[%c0, %c0], %p {in_bounds = [true, true]} : memref<8x1xf16, strided<[8, 1]>>, vector<8x1xf16>
  %bt = vector.transpose %rb, [1, 0] : vector<8x1xf16> to vector<1x8xf16>
  %ea = arith.extf %ra : vector<8x1xf16> to vector<8x1xf32>
  %eb = arith.extf %bt : vector<1x8xf16> to vector<1x8xf32>
  %r = vector.contract {indexing_maps = [#mk, #kn, #mn], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %ea, %eb, %acc : vector<8x1xf32>, vector<1x8xf32> into vector<8x8xf32>
  return %r, %ea : vector<8x8xf32>, vector<8x1xf32>
}

// -----

// One read is the A of one product and the B of another; it cannot be
// widened for both.
// CHECK-LABEL: func.func @two_roles
// CHECK-COUNT-2: vector.contract {{.+}} : vector<8x1xf32>, vector<1x8xf32> into vector<8x8xf32>
#mk = affine_map<(m, n, k) -> (m, k)>
#kn = affine_map<(m, n, k) -> (k, n)>
#mn = affine_map<(m, n, k) -> (m, n)>
func.func @two_roles(%a: memref<8x1xf16, strided<[8, 1]>>,
                     %acc: vector<8x8xf32>) -> vector<8x8xf32> {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0.0 : f16
  %ra = vector.transfer_read %a[%c0, %c0], %p {in_bounds = [true, true]} : memref<8x1xf16, strided<[8, 1]>>, vector<8x1xf16>
  %bt = vector.transpose %ra, [1, 0] : vector<8x1xf16> to vector<1x8xf16>
  %ea = arith.extf %ra : vector<8x1xf16> to vector<8x1xf32>
  %eb = arith.extf %bt : vector<1x8xf16> to vector<1x8xf32>
  %r = vector.contract {indexing_maps = [#mk, #kn, #mn], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %ea, %eb, %acc : vector<8x1xf32>, vector<1x8xf32> into vector<8x8xf32>
  %s = vector.contract {indexing_maps = [#mk, #kn, #mn], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %ea, %eb, %r : vector<8x1xf32>, vector<1x8xf32> into vector<8x8xf32>
  return %s : vector<8x8xf32>
}

// -----

// The weight-stationary form: B is read once before the loop, carried as an
// iter_arg through a cpu.sched_barrier and yielded back. The read before the
// loop is widened and the iter_arg, the barrier and the yield take the full
// K; A is read in the loop.
// CHECK-LABEL: func.func @loop_carried_b
// CHECK: %[[B:.+]] = vector.transfer_read {{.+}} : memref<8x8xf16, strided<[8, 1], offset: ?>>, vector<8x8xf16>
// CHECK: scf.for {{.+}} iter_args(%[[ARG:.+]] = %[[B]], %{{.+}} = %{{.+}}) -> (vector<8x8xf16>, vector<8x8xf32>)
// CHECK: %[[HELD:.+]] = cpu.sched_barrier %[[ARG]] : vector<8x8xf16>
// CHECK: vector.transpose %[[HELD]], [1, 0] : vector<8x8xf16> to vector<8x8xf16>
// CHECK: vector.contract {{.+}} : vector<8x8xf32>, vector<8x8xf32> into vector<8x8xf32>
// CHECK: scf.yield %[[HELD]], %{{.+}} : vector<8x8xf16>, vector<8x8xf32>
// TILE-LABEL: func.func @loop_carried_b
// TILE: ime.tile.vfwmadot
// TILE-NOT: vector.contract
#mk = affine_map<(m, n, k) -> (m, k)>
#kn = affine_map<(m, n, k) -> (k, n)>
#mn = affine_map<(m, n, k) -> (m, n)>
func.func @loop_carried_b(%a: memref<64xf16>, %b: memref<8x1xf16, strided<[8, 1], offset: ?>>,
                          %acc: vector<8x8xf32>, %n: index) -> vector<8x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %p = arith.constant 0.0 : f16
  %rb = vector.transfer_read %b[%c0, %c0], %p {in_bounds = [true, true]} : memref<8x1xf16, strided<[8, 1], offset: ?>>, vector<8x1xf16>
  %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%wb = %rb, %c = %acc) -> (vector<8x1xf16>, vector<8x8xf32>) {
    %held = cpu.sched_barrier %wb : vector<8x1xf16>
    %va = memref.reinterpret_cast %a to offset: [%i], sizes: [8, 1], strides: [8, 1] : memref<64xf16> to memref<8x1xf16, strided<[8, 1], offset: ?>>
    %ra = vector.transfer_read %va[%c0, %c0], %p {in_bounds = [true, true]} : memref<8x1xf16, strided<[8, 1], offset: ?>>, vector<8x1xf16>
    %bt = vector.transpose %held, [1, 0] : vector<8x1xf16> to vector<1x8xf16>
    %ea = arith.extf %ra : vector<8x1xf16> to vector<8x1xf32>
    %eb = arith.extf %bt : vector<1x8xf16> to vector<1x8xf32>
    %d = vector.contract {indexing_maps = [#mk, #kn, #mn], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %ea, %eb, %c : vector<8x1xf32>, vector<1x8xf32> into vector<8x8xf32>
    scf.yield %held, %d : vector<8x1xf16>, vector<8x8xf32>
  }
  return %r#1 : vector<8x8xf32>
}

// -----

// The carried B is also read after the loop, which a wider carry would
// change.
// CHECK-LABEL: func.func @loop_result_used
// CHECK: vector.contract {{.+}} : vector<8x1xf32>, vector<1x8xf32> into vector<8x8xf32>
// TILE-LABEL: func.func @loop_result_used
// TILE: vector.contract
#mk = affine_map<(m, n, k) -> (m, k)>
#kn = affine_map<(m, n, k) -> (k, n)>
#mn = affine_map<(m, n, k) -> (m, n)>
func.func @loop_result_used(%a: memref<8x1xf16, strided<[8, 1]>>, %b: memref<8x1xf16, strided<[8, 1]>>,
                            %acc: vector<8x8xf32>, %n: index) -> (vector<8x8xf32>, vector<8x1xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %p = arith.constant 0.0 : f16
  %rb = vector.transfer_read %b[%c0, %c0], %p {in_bounds = [true, true]} : memref<8x1xf16, strided<[8, 1]>>, vector<8x1xf16>
  %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%wb = %rb, %c = %acc) -> (vector<8x1xf16>, vector<8x8xf32>) {
    %ra = vector.transfer_read %a[%c0, %c0], %p {in_bounds = [true, true]} : memref<8x1xf16, strided<[8, 1]>>, vector<8x1xf16>
    %bt = vector.transpose %wb, [1, 0] : vector<8x1xf16> to vector<1x8xf16>
    %ea = arith.extf %ra : vector<8x1xf16> to vector<8x1xf32>
    %eb = arith.extf %bt : vector<1x8xf16> to vector<1x8xf32>
    %d = vector.contract {indexing_maps = [#mk, #kn, #mn], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %ea, %eb, %c : vector<8x1xf32>, vector<1x8xf32> into vector<8x8xf32>
    scf.yield %wb, %d : vector<8x1xf16>, vector<8x8xf32>
  }
  return %r#1, %r#0 : vector<8x8xf32>, vector<8x1xf16>
}
