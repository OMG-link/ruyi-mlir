// RUN: buddy-opt %s --set-llvm-target="chip=spacemit-x60" --lower-cpu-sched-barrier --split-input-file --verify-diagnostics
// RUN: buddy-opt %s --set-llvm-target="features=+v,+zvl256b" --lower-cpu-sched-barrier --split-input-file --verify-diagnostics
// A named CPU and an explicit Zvl lower bound both leave exact VLEN unknown.
func.func @requires_exact_vlen(%a: vector<16xf32>) -> vector<16xf32> {
  // expected-error @+1 {{exact RISC-V VLEN is required for scheduling barrier lowering}}
  %r = cpu.sched_barrier %a : vector<16xf32>
  return %r : vector<16xf32>
}

// -----

// Functions without a barrier do not require exact VLEN.
func.func @no_barrier(%a: vector<16xf32>) -> vector<16xf32> {
  return %a : vector<16xf32>
}
