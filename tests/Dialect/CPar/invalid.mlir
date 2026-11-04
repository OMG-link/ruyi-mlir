// RUN: buddy-opt %s --split-input-file --verify-diagnostics

func.func @bad_entry_args() {
  // expected-error @+1 {{expects the entry block to have two i64 arguments (tid, nthreads)}}
  cpar.parallel num_threads(4) {
  ^bb0(%tid: i32, %nt: i64):
    cpar.yield
  }
  return
}

// -----

func.func @bad_num_threads() {
  // expected-error @+1 {{attribute 'num_threads' failed to satisfy constraint}}
  cpar.parallel num_threads(0) {
  ^bb0(%tid: i64, %nt: i64):
    cpar.yield
  }
  return
}

// -----

func.func @bad_exit() {
  cpar.parallel num_threads(2) {
  ^bb0(%tid: i64, %nt: i64):
    // expected-error @+1 {{leaves a cpar.parallel region; only cpar.yield may do so}}
    func.return
  }
  return
}

// -----

func.func @yield_outside() {
  // expected-error @+1 {{expects parent op 'cpar.parallel'}}
  cpar.yield
}
