// RUN: buddy-opt %s > %t.no-pass
// RUN: buddy-opt --qgemm-recognize %s > %t.with-pass
// RUN: diff -u %t.no-pass %t.with-pass
// RUN: buddy-opt --debug-only=qgemm-recognize --qgemm-recognize %s 2>&1 | FileCheck %s
//
// A function with no reduction has no candidate and is byte-identical after
// running the recognizer.

// CHECK: [qgemm-recognize] candidates: 0
// CHECK-LABEL: func.func @no_reduction
// CHECK: return
// CHECK-NOT: qgemm.node
func.func @no_reduction(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  return %arg0 : tensor<4xf32>
}
