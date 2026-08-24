// RUN: buddy-opt --qgemm-tiling="tile-m=4 tile-n=4" %s 2>&1 | FileCheck %s
//
// Fusion control is driven only by the hand-written qgemm.node identity. The
// destination/outs tensor fill is the sole root-neutral exception; an ins fill
// and an ordinary unannotated producer stay outside the tile. The final
// function checks that two roots sharing one root-neutral fill clone it once
// per root instead of sharing a fused operation.

// CHECK-COUNT-1: qgemm-tiling: destination tensor fill carries QGEMM annotation; continuing with tile-local cloning
// CHECK-NOT: qgemm-tiling: destination tensor fill carries QGEMM annotation; continuing with tile-local cloning
// CHECK-LABEL: func.func @destination_fill_annotation_warning
// CHECK: {{^    }}%{{[a-zA-Z0-9_]+}} = scf.for %{{[a-zA-Z0-9_]+}} = %c0{{[_0-9]*}} to %c8{{[_0-9]*}} step %c4{{[_0-9]*}}
// CHECK: {{^      }}%{{[a-zA-Z0-9_]+}} = scf.for %{{[a-zA-Z0-9_]+}} = %c0{{[_0-9]*}} to %c8{{[_0-9]*}} step %c4{{[_0-9]*}}
// CHECK: {{^        }}%{{[a-zA-Z0-9_]+}} = linalg.fill{{.*}}outs(%{{[a-zA-Z0-9_]+}} : tensor<4x4xf32>) -> tensor<4x4xf32>{{[[:space:]]*$}}
// CHECK: {{^        }}%{{[a-zA-Z0-9_]+}} = linalg.reduce{{.*}}qgemm.node = {root = 7 : i64, stage = "compute"}, qgemm.root =
// CHECK-SAME: {m_dim = 0 : i64, n_dim = 1 : i64}
// CHECK-NOT: qgemm-tiling: destination tensor fill carries QGEMM annotation; continuing with tile-local cloning

func.func @destination_fill_annotation_warning(%input: tensor<8x8x4xf32>)
    -> tensor<8x8xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<8x8xf32>
  %init = linalg.fill {qgemm.node = {root = 7 : i64, stage = "data_prep"}}
      ins(%zero : f32) outs(%empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  %root = linalg.reduce ins(%input : tensor<8x8x4xf32>)
                           outs(%init : tensor<8x8xf32>)
                           dimensions = [2]
                           {qgemm.node = {root = 7 : i64, stage = "compute"},
                            qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}}
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %root : tensor<8x8xf32>
}

// CHECK-LABEL: func.func @destination_fill_exception
// CHECK: {{^    }}%{{[a-zA-Z0-9_]+}} = scf.for %{{[a-zA-Z0-9_]+}} = %c0{{[_0-9]*}} to %c8{{[_0-9]*}} step %c4{{[_0-9]*}}
// CHECK: {{^      }}%{{[a-zA-Z0-9_]+}} = scf.for %{{[a-zA-Z0-9_]+}} = %c0{{[_0-9]*}} to %c8{{[_0-9]*}} step %c4{{[_0-9]*}}
// CHECK: {{^        }}%{{[a-zA-Z0-9_]+}} = linalg.fill ins(%{{[a-zA-Z0-9_]+}} : f32) outs(%{{[a-zA-Z0-9_]+}} : tensor<4x4xf32>) -> tensor<4x4xf32>{{[[:space:]]*$}}
// CHECK: {{^        }}%{{[a-zA-Z0-9_]+}} = linalg.reduce{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}, qgemm.root =
// CHECK-SAME: {m_dim = 0 : i64, n_dim = 1 : i64}

func.func @destination_fill_exception(%input: tensor<8x8x4xf32>) -> tensor<8x8xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<8x8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  %root = linalg.reduce ins(%input : tensor<8x8x4xf32>)
                           outs(%init : tensor<8x8xf32>)
                           dimensions = [2]
                           {qgemm.node = {root = 0 : i64, stage = "compute"},
                            qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}}
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %root : tensor<8x8xf32>
}

// The ins-edge fill is present before the loops. The destination fill may
// be eliminated from that position, but its tile-local clone stays free of
// identity attributes; the original input fill must not enter the loop body.
// CHECK-LABEL: func.func @ins_fill_rejected
// CHECK: {{^    }}%{{[a-zA-Z0-9_]+}} = linalg.fill {test.marker = "input_fill"} ins(%{{[a-zA-Z0-9_]+}} : f32) outs(%{{[a-zA-Z0-9_]+}} : tensor<8x8x4xf32>) -> tensor<8x8x4xf32>{{[[:space:]]*$}}
// CHECK: {{^    }}%{{[a-zA-Z0-9_]+}} = scf.for %{{[a-zA-Z0-9_]+}} = %c0{{[_0-9]*}} to %c8{{[_0-9]*}} step %c4{{[_0-9]*}}
// CHECK-NOT: test.marker = "input_fill"
// CHECK: {{^        }}%{{[a-zA-Z0-9_]+}} = linalg.fill ins(%{{[a-zA-Z0-9_]+}} : f32) outs(%{{[a-zA-Z0-9_]+}} : tensor<4x4xf32>) -> tensor<4x4xf32>{{[[:space:]]*$}}
// CHECK-NOT: test.marker = "input_fill"
// CHECK: {{^        }}%{{[a-zA-Z0-9_]+}} = linalg.reduce{{.*}}qgemm.node = {root = 1 : i64, stage = "compute"}, qgemm.root =
// CHECK-SAME: {m_dim = 0 : i64, n_dim = 1 : i64}

func.func @ins_fill_rejected() -> tensor<8x8xf32> {
  %zero = arith.constant 0.0 : f32
  %input_empty = tensor.empty() : tensor<8x8x4xf32>
  %input_fill = linalg.fill {test.marker = "input_fill"} ins(%zero : f32)
      outs(%input_empty : tensor<8x8x4xf32>) -> tensor<8x8x4xf32>
  %output_empty = tensor.empty() : tensor<8x8xf32>
  %output_fill = linalg.fill ins(%zero : f32)
      outs(%output_empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  %root = linalg.reduce ins(%input_fill : tensor<8x8x4xf32>)
                           outs(%output_fill : tensor<8x8xf32>)
                           dimensions = [2]
                           {qgemm.node = {root = 1 : i64, stage = "compute"},
                            qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}}
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %root : tensor<8x8xf32>
}

// The ordinary producer is also before the loops and must not be duplicated
// into the loop body; only the destination fill is tile-local and unannotated.
// CHECK-LABEL: func.func @unannotated_producer_rejected
// CHECK: {{^    }}%{{[a-zA-Z0-9_]+}} = linalg.generic
// CHECK-SAME: test.marker = "ordinary_producer"
// CHECK: {{^    }}%{{[a-zA-Z0-9_]+}} = scf.for %{{[a-zA-Z0-9_]+}} = %c0{{[_0-9]*}} to %c8{{[_0-9]*}} step %c4{{[_0-9]*}}
// CHECK-NOT: test.marker = "ordinary_producer"
// CHECK: {{^        }}%{{[a-zA-Z0-9_]+}} = linalg.fill ins(%{{[a-zA-Z0-9_]+}} : f32) outs(%{{[a-zA-Z0-9_]+}} : tensor<4x4xf32>) -> tensor<4x4xf32>{{[[:space:]]*$}}
// CHECK-NOT: test.marker = "ordinary_producer"
// CHECK: {{^        }}%{{[a-zA-Z0-9_]+}} = linalg.reduce{{.*}}qgemm.node = {root = 2 : i64, stage = "compute"}, qgemm.root =
// CHECK-SAME: {m_dim = 0 : i64, n_dim = 1 : i64}

func.func @unannotated_producer_rejected(%input: tensor<8x8x4xf32>) -> tensor<8x8xf32> {
  %zero = arith.constant 0.0 : f32
  %input_empty = tensor.empty() : tensor<8x8x4xf32>
  %input_generic = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%input : tensor<8x8x4xf32>) outs(%input_empty : tensor<8x8x4xf32>)
    attrs = {test.marker = "ordinary_producer"} {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x4xf32>
  %output_empty = tensor.empty() : tensor<8x8xf32>
  %output_fill = linalg.fill ins(%zero : f32)
      outs(%output_empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  %root = linalg.reduce ins(%input_generic : tensor<8x8x4xf32>)
                           outs(%output_fill : tensor<8x8xf32>)
                           dimensions = [2]
                           {qgemm.node = {root = 2 : i64, stage = "compute"},
                            qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}}
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %root : tensor<8x8xf32>
}

// Two roots share a root-neutral tensor fill. Each root receives its own
// M/N loop context and its own unannotated tile-local fill.
// CHECK-LABEL: func.func @two_roots_shared_fill
// CHECK-NOT: qgemm.root
// CHECK: {{^    }}%{{[a-zA-Z0-9_]+}} = scf.for %{{[a-zA-Z0-9_]+}} = %c0{{[_0-9]*}} to %c8{{[_0-9]*}} step %c4{{[_0-9]*}}
// CHECK: {{^        }}%{{[a-zA-Z0-9_]+}} = linalg.fill ins(%{{[a-zA-Z0-9_]+}} : f32) outs(%{{[a-zA-Z0-9_]+}} : tensor<4x4xf32>) -> tensor<4x4xf32>{{[[:space:]]*$}}
// CHECK: {{^        }}%{{[a-zA-Z0-9_]+}} = linalg.reduce{{.*}}qgemm.node = {root = 3 : i64, stage = "compute"}, qgemm.root =
// CHECK-SAME: {m_dim = 0 : i64, n_dim = 1 : i64}
// CHECK-NOT: qgemm.root
// CHECK: {{^    }}%{{[a-zA-Z0-9_]+}} = scf.for %{{[a-zA-Z0-9_]+}} = %c0{{[_0-9]*}} to %c8{{[_0-9]*}} step %c4{{[_0-9]*}}
// CHECK: {{^        }}%{{[a-zA-Z0-9_]+}} = linalg.fill ins(%{{[a-zA-Z0-9_]+}} : f32) outs(%{{[a-zA-Z0-9_]+}} : tensor<4x4xf32>) -> tensor<4x4xf32>{{[[:space:]]*$}}
// CHECK: {{^        }}%{{[a-zA-Z0-9_]+}} = linalg.reduce{{.*}}qgemm.node = {root = 4 : i64, stage = "compute"}, qgemm.root =
// CHECK-SAME: {m_dim = 0 : i64, n_dim = 1 : i64}
// CHECK-NOT: qgemm.root

func.func @two_roots_shared_fill(%input0: tensor<8x8x4xf32>,
                                 %input1: tensor<8x8x4xf32>)
    -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  %zero = arith.constant 0.0 : f32
  %shared_empty = tensor.empty() : tensor<8x8xf32>
  %shared_fill = linalg.fill ins(%zero : f32)
      outs(%shared_empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  %root0 = linalg.reduce ins(%input0 : tensor<8x8x4xf32>)
                            outs(%shared_fill : tensor<8x8xf32>)
                            dimensions = [2]
                            {qgemm.node = {root = 3 : i64, stage = "compute"},
                             qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}}
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  %root1 = linalg.reduce ins(%input1 : tensor<8x8x4xf32>)
                            outs(%shared_fill : tensor<8x8xf32>)
                            dimensions = [2]
                            {qgemm.node = {root = 4 : i64, stage = "compute"},
                             qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}}
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %root0, %root1 : tensor<8x8xf32>, tensor<8x8xf32>
}

// Two separately located root operations may carry the same root identity
// after an earlier tiling round. The tiler processes both independently.
// CHECK-LABEL: func.func @same_root_multiple_roots
// CHECK-NOT: qgemm.root
// CHECK: {{^    }}%{{[a-zA-Z0-9_]+}} = scf.for %{{[a-zA-Z0-9_]+}} = %c0{{[_0-9]*}} to %c8{{[_0-9]*}} step %c4{{[_0-9]*}}
// CHECK: {{^        }}%{{[a-zA-Z0-9_]+}} = linalg.fill ins(%{{[a-zA-Z0-9_]+}} : f32) outs(%{{[a-zA-Z0-9_]+}} : tensor<4x4xf32>) -> tensor<4x4xf32>{{[[:space:]]*$}}
// CHECK: {{^        }}%{{[a-zA-Z0-9_]+}} = linalg.reduce{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}, qgemm.root =
// CHECK-SAME: {m_dim = 0 : i64, n_dim = 1 : i64}
// CHECK-NOT: qgemm.root
// CHECK: {{^    }}%{{[a-zA-Z0-9_]+}} = scf.for %{{[a-zA-Z0-9_]+}} = %c0{{[_0-9]*}} to %c8{{[_0-9]*}} step %c4{{[_0-9]*}}
// CHECK: {{^        }}%{{[a-zA-Z0-9_]+}} = linalg.fill ins(%{{[a-zA-Z0-9_]+}} : f32) outs(%{{[a-zA-Z0-9_]+}} : tensor<4x4xf32>) -> tensor<4x4xf32>{{[[:space:]]*$}}
// CHECK: {{^        }}%{{[a-zA-Z0-9_]+}} = linalg.reduce{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}, qgemm.root =
// CHECK-SAME: {m_dim = 0 : i64, n_dim = 1 : i64}
// CHECK-NOT: qgemm.root

// A producer carrying a different root identity is not fused into the
// consumer's tile. It remains before the consumer's loops.
// CHECK-LABEL: func.func @different_root_producer
// CHECK: {{^    }}%{{[a-zA-Z0-9_]+}} = linalg.generic{{.*}}qgemm.node = {root = 1 : i64, stage = "compute"}} {{{[[:space:]]*$}}
// CHECK-NOT: qgemm.node = {root = 1 : i64
// CHECK: {{^    }}%{{[a-zA-Z0-9_]+}} = scf.for %{{[a-zA-Z0-9_]+}} = %c0{{[_0-9]*}} to %c8{{[_0-9]*}} step %c4{{[_0-9]*}}
// CHECK-NOT: qgemm.node = {root = 1 : i64
// CHECK: {{^        }}%{{[a-zA-Z0-9_]+}} = linalg.fill ins(%{{[a-zA-Z0-9_]+}} : f32) outs(%{{[a-zA-Z0-9_]+}} : tensor<4x4xf32>) -> tensor<4x4xf32>{{[[:space:]]*$}}
// CHECK-NOT: qgemm.node = {root = 1 : i64
// CHECK: {{^        }}%{{[a-zA-Z0-9_]+}} = linalg.reduce{{.*}}qgemm.node = {root = 0 : i64, stage = "compute"}, qgemm.root =
// CHECK-SAME: {m_dim = 0 : i64, n_dim = 1 : i64}
// CHECK-NOT: qgemm.node = {root = 1 : i64

func.func @same_root_multiple_roots(%input0: tensor<8x8x4xf32>,
                                     %input1: tensor<8x8x4xf32>)
    -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  %zero = arith.constant 0.0 : f32
  %empty0 = tensor.empty() : tensor<8x8xf32>
  %fill0 = linalg.fill ins(%zero : f32) outs(%empty0 : tensor<8x8xf32>)
      -> tensor<8x8xf32>
  %root0 = linalg.reduce ins(%input0 : tensor<8x8x4xf32>)
      outs(%fill0 : tensor<8x8xf32>) dimensions = [2]
      {qgemm.node = {root = 0 : i64, stage = "compute"},
       qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}}
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  %empty1 = tensor.empty() : tensor<8x8xf32>
  %fill1 = linalg.fill ins(%zero : f32) outs(%empty1 : tensor<8x8xf32>)
      -> tensor<8x8xf32>
  %root1 = linalg.reduce ins(%input1 : tensor<8x8x4xf32>)
      outs(%fill1 : tensor<8x8xf32>) dimensions = [2]
      {qgemm.node = {root = 0 : i64, stage = "compute"},
       qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}}
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %root0, %root1 : tensor<8x8xf32>, tensor<8x8xf32>
}

func.func @different_root_producer(%input: tensor<8x8x4xf32>) -> tensor<8x8xf32> {
  %empty_input = tensor.empty() : tensor<8x8x4xf32>
  %producer = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%input : tensor<8x8x4xf32>) outs(%empty_input : tensor<8x8x4xf32>)
    attrs = {qgemm.node = {root = 1 : i64, stage = "compute"}} {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x4xf32>
  %zero = arith.constant 0.0 : f32
  %empty_output = tensor.empty() : tensor<8x8xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%empty_output : tensor<8x8xf32>)
      -> tensor<8x8xf32>
  %root = linalg.reduce ins(%producer : tensor<8x8x4xf32>)
      outs(%fill : tensor<8x8xf32>) dimensions = [2]
      {qgemm.node = {root = 0 : i64, stage = "compute"},
       qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}}
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %root : tensor<8x8xf32>
}

// A same-root data-preparation producer is a stage boundary, while the
// compute-stage broadcast generic and final reduction are fused into the
// tile. The local destination fill remains root-neutral.
// CHECK-LABEL: func.func @same_root_data_prep_rejected
// CHECK: {{^    }}%{{[a-zA-Z0-9_]+}} = linalg.generic{{.*}}qgemm.node = {root = 5 : i64, stage = "data_prep"}
// CHECK-SAME: test.marker = "data_prep"
// CHECK: {{^    }}%{{[a-zA-Z0-9_]+}} = scf.for %{{[a-zA-Z0-9_]+}} = %c0{{[_0-9]*}} to %c8{{[_0-9]*}} step %c4{{[_0-9]*}}
// CHECK-NOT: test.marker = "data_prep"
// CHECK: linalg.generic{{.*}}qgemm.node = {root = 5 : i64, stage = "compute"}
// CHECK-SAME: test.marker = "compute_broadcast"
// CHECK: linalg.fill ins(%{{[a-zA-Z0-9_]+}} : f32) outs(%{{[a-zA-Z0-9_]+}} : tensor<4x4xf32>) -> tensor<4x4xf32>{{[[:space:]]*$}}
// CHECK-NOT: qgemm.node
// CHECK: linalg.reduce{{.*}}qgemm.node = {root = 5 : i64, stage = "compute"}, qgemm.root =
// CHECK-SAME: {m_dim = 0 : i64, n_dim = 1 : i64}
// CHECK-NOT: test.marker = "data_prep"

func.func @same_root_data_prep_rejected(%input: tensor<1x8x4xf32>)
    -> tensor<8x8xf32> {
  %prep_empty = tensor.empty() : tensor<1x8x4xf32>
  %prep = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%input : tensor<1x8x4xf32>) outs(%prep_empty : tensor<1x8x4xf32>)
    attrs = {qgemm.node = {root = 5 : i64, stage = "data_prep"},
            test.marker = "data_prep"} {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x8x4xf32>
  %compute_empty = tensor.empty() : tensor<8x8x4xf32>
  %compute = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%prep : tensor<1x8x4xf32>) outs(%compute_empty : tensor<8x8x4xf32>)
    attrs = {broadcastDims = array<i64: 0>,
            qgemm.node = {root = 5 : i64, stage = "compute"},
            test.marker = "compute_broadcast"} {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x4xf32>
  %zero = arith.constant 0.0 : f32
  %output_empty = tensor.empty() : tensor<8x8xf32>
  %output_fill = linalg.fill ins(%zero : f32)
      outs(%output_empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  %root = linalg.reduce ins(%compute : tensor<8x8x4xf32>)
      outs(%output_fill : tensor<8x8xf32>) dimensions = [2]
      {qgemm.node = {root = 5 : i64, stage = "compute"},
       qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}}
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %root : tensor<8x8xf32>
}

// A malformed stage is not a valid compute-stage identity and therefore
// remains outside the tile just like a missing stage.
// CHECK-LABEL: func.func @same_root_malformed_stage_rejected
// CHECK: {{^    }}%{{[a-zA-Z0-9_]+}} = linalg.generic{{.*}}qgemm.node = {root = 6 : i64, stage = "malformed"}
// CHECK-SAME: test.marker = "malformed_stage"
// CHECK: {{^    }}%{{[a-zA-Z0-9_]+}} = scf.for %{{[a-zA-Z0-9_]+}} = %c0{{[_0-9]*}} to %c8{{[_0-9]*}} step %c4{{[_0-9]*}}
// CHECK-NOT: test.marker = "malformed_stage"
// CHECK: {{^        }}%{{[a-zA-Z0-9_]+}} = linalg.reduce{{.*}}qgemm.node = {root = 6 : i64, stage = "compute"}, qgemm.root =
// CHECK-SAME: {m_dim = 0 : i64, n_dim = 1 : i64}
// CHECK-NOT: test.marker = "malformed_stage"

func.func @same_root_malformed_stage_rejected(
    %input: tensor<8x8x4xf32>) -> tensor<8x8xf32> {
  %input_empty = tensor.empty() : tensor<8x8x4xf32>
  %producer = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%input : tensor<8x8x4xf32>) outs(%input_empty : tensor<8x8x4xf32>)
    attrs = {qgemm.node = {root = 6 : i64, stage = "malformed"},
            test.marker = "malformed_stage"} {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<8x8x4xf32>
  %zero = arith.constant 0.0 : f32
  %output_empty = tensor.empty() : tensor<8x8xf32>
  %output_fill = linalg.fill ins(%zero : f32)
      outs(%output_empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  %root = linalg.reduce ins(%producer : tensor<8x8x4xf32>)
      outs(%output_fill : tensor<8x8xf32>) dimensions = [2]
      {qgemm.node = {root = 6 : i64, stage = "compute"},
       qgemm.root = {m_dim = 0 : i64, n_dim = 1 : i64}}
    (%in: f32, %acc: f32) {
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    }
  return %root : tensor<8x8xf32>
}
