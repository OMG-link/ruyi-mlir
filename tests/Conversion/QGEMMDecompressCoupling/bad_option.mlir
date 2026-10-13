// RUN: not buddy-opt --qgemm-decompress-coupling=decompress=foo %s 2>&1 | FileCheck %s
//
// An unknown `decompress` value is rejected at option parsing, and the
// diagnostic names the offending value. (The LLVM enum option parser does
// not list the accepted values `pre` / `on-the-fly` in this message.)

// CHECK: --decompress option: Cannot find option named 'foo'
func.func @f() {
  return
}
