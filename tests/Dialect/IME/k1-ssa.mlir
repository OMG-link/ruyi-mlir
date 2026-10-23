// RUN: buddy-translate -buddy-to-llvmir %s | buddy-llc -mtriple=riscv64 -mattr=+m,+v,+xsmtvdot -verify-machineinstrs -o - | FileCheck %s
// The K1 intrinsic ABI: one register per int8 input tile (a register pair
// for the sliding-window A operand) and a register pair for the int32
// accumulator.
llvm.func @k1_dot(%c: vector<[4]xi32>, %a: vector<[8]xi8>) -> vector<[4]xi32> {
  %r = "ime.intr.vmadot"(%c, %a, %a) : (vector<[4]xi32>, vector<[8]xi8>, vector<[8]xi8>) -> vector<[4]xi32>
  llvm.return %r : vector<[4]xi32>
}
// CHECK-LABEL: k1_dot:
// CHECK: smt.vmadot v8, v10, v10
llvm.func @k1_slide(%c: vector<[4]xi32>, %a: vector<[16]xi8>, %b: vector<[8]xi8>) -> vector<[4]xi32> {
  %r = "ime.intr.vmadot1"(%c, %a, %b) : (vector<[4]xi32>, vector<[16]xi8>, vector<[8]xi8>) -> vector<[4]xi32>
  llvm.return %r : vector<[4]xi32>
}
// CHECK-LABEL: k1_slide:
// CHECK: smt.vmadot1 v8, v10, v12
