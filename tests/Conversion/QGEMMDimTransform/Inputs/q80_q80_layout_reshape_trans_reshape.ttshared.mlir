#loc = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_layout.py":169:0)
#loc11 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_layout.py":254:26)
#loc12 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_layout.py":256:26)
#loc13 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_layout.py":260:19)
#loc14 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_layout.py":260:39)
#loc16 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_layout.py":263:29)
#loc17 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_layout.py":264:26)
#loc18 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/triton-riscv/.venv/lib/python3.10/site-packages/triton/language/standard.py":291:36)
#loc19 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_layout.py":264:17)
#loc21 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_layout.py":266:26)
#loc22 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_layout.py":268:26)
#loc23 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_layout.py":270:16)
#loc24 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_layout.py":270:38)
#loc25 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_layout.py":270:30)
#loc26 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_layout.py":271:18)
#loc27 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_layout.py":271:32)
#loc28 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_layout.py":274:15)
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#loc29 = loc("a_q_ptr"(#loc))
#loc30 = loc("a_d_ptr"(#loc))
#loc31 = loc("w_q_ptr"(#loc))
#loc32 = loc("w_d_ptr"(#loc))
#loc33 = loc("c_ptr"(#loc))
#loc42 = loc("a_q3"(#loc11))
#loc43 = loc("w_q3"(#loc12))
#loc44 = loc("prod"(#loc13))
#loc45 = loc("prod"(#loc14))
#loc47 = loc("prod4"(#loc16))
#loc48 = loc("blk"(#loc17))
#loc49 = loc("blk"(#loc19))
#loc51 = loc("a_d3"(#loc21))
#loc52 = loc("w_d3"(#loc22))
#loc53 = loc("d"(#loc23))
#loc54 = loc("d"(#loc24))
#loc55 = loc("d"(#loc25))
#loc56 = loc("lvl1"(#loc26))
#loc57 = loc("lvl1"(#loc27))
#loc58 = loc("c"(#loc28))
#loc60 = loc(callsite(#loc18 at #loc49))
#loc61 = loc(callsite(#loc18 at #loc58))
module {
  func.func @q80_q80_layout_reshape_trans_reshape(%arg0: memref<*xi8> {tt.divisibility = 16 : i32} loc("a_q_ptr"(#loc)), %arg1: memref<*xf16> {tt.divisibility = 16 : i32} loc("a_d_ptr"(#loc)), %arg2: memref<*xi8> {tt.divisibility = 16 : i32} loc("w_q_ptr"(#loc)), %arg3: memref<*xf16> {tt.divisibility = 16 : i32} loc("w_d_ptr"(#loc)), %arg4: memref<*xf32> {tt.divisibility = 16 : i32} loc("c_ptr"(#loc)), %arg5: i32 loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_layout.py":169:0), %arg6: i32 loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_layout.py":169:0), %arg7: i32 loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_layout.py":169:0), %arg8: i32 loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_layout.py":169:0), %arg9: i32 loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_layout.py":169:0), %arg10: i32 loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_layout.py":169:0)) {
    %c32_i32 = arith.constant 32 : i32 loc(#loc1)
    %c0_i32 = arith.constant 0 : i32 loc(#loc1)
    %cst = arith.constant 0.000000e+00 : f32 loc(#loc1)
    %0 = arith.muli %arg8, %c32_i32 : i32 loc(#loc34)
    %1 = arith.index_cast %0 : i32 to index loc(#loc59)
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [192, 512], strides: [512, 1] : memref<*xi8> to memref<192x512xi8, strided<[512, 1]>> loc(#loc36)
    %alloc = memref.alloc() : memref<192x512xi8> loc(#loc36)
    memref.copy %reinterpret_cast, %alloc : memref<192x512xi8, strided<[512, 1]>> to memref<192x512xi8> loc(#loc36)
    %2 = bufferization.to_tensor %alloc restrict writable : memref<192x512xi8> to tensor<192x512xi8> loc(#loc36)
    %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%1], sizes: [16, 32, 32], strides: [4096, 128, 1] : memref<*xi8> to memref<16x32x32xi8, strided<[4096, 128, 1], offset: ?>> loc(#loc37)
    %alloc_1 = memref.alloc() : memref<16x32x32xi8> loc(#loc37)
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<16x32x32xi8, strided<[4096, 128, 1], offset: ?>> to memref<16x32x32xi8> loc(#loc37)
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<16x32x32xi8> to tensor<16x32x32xi8> loc(#loc37)
    %collapsed = tensor.collapse_shape %3 [[0, 1], [2]] : tensor<16x32x32xi8> into tensor<512x32xi8> loc(#loc38)
    %4 = tensor.empty() : tensor<32x512xi8> loc(#loc39)
    %transposed = linalg.transpose ins(%collapsed : tensor<512x32xi8>) outs(%4 : tensor<32x512xi8>) permutation = [1, 0]  loc(#loc39)
    %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%1], sizes: [16, 32], strides: [128, 1] : memref<*xf16> to memref<16x32xf16, strided<[128, 1], offset: ?>> loc(#loc40)
    %alloc_3 = memref.alloc() : memref<16x32xf16> loc(#loc40)
    memref.copy %reinterpret_cast_2, %alloc_3 : memref<16x32xf16, strided<[128, 1], offset: ?>> to memref<16x32xf16> loc(#loc40)
    %5 = bufferization.to_tensor %alloc_3 restrict writable : memref<16x32xf16> to tensor<16x32xf16> loc(#loc40)
    %6 = tensor.empty() : tensor<32x16xf16> loc(#loc41)
    %transposed_4 = linalg.transpose ins(%5 : tensor<16x32xf16>) outs(%6 : tensor<32x16xf16>) permutation = [1, 0]  loc(#loc41)
    %expanded = tensor.expand_shape %2 [[0], [1, 2]] output_shape [192, 1, 512] : tensor<192x512xi8> into tensor<192x1x512xi8> loc(#loc42)
    %expanded_5 = tensor.expand_shape %transposed [[0, 1], [2]] output_shape [1, 32, 512] : tensor<32x512xi8> into tensor<1x32x512xi8> loc(#loc43)
    %7 = tensor.empty() : tensor<192x1x512xi16> loc(#loc44)
    %8 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded : tensor<192x1x512xi8>) outs(%7 : tensor<192x1x512xi16>) {
    ^bb0(%in: i8 loc("a_q3"(#loc11)), %out: i16 loc("prod"(#loc13))):
      %32 = arith.extsi %in : i8 to i16 loc(#loc44)
      linalg.yield %32 : i16 loc(#loc44)
    } -> tensor<192x1x512xi16> loc(#loc44)
    %9 = tensor.empty() : tensor<192x32x512xi16> loc(#loc44)
    %10 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8 : tensor<192x1x512xi16>) outs(%9 : tensor<192x32x512xi16>) attrs =  {broadcastDims = array<i64: 1>} {
    ^bb0(%in: i16 loc("prod"(#loc13)), %out: i16 loc("prod"(#loc13))):
      linalg.yield %in : i16 loc(#loc44)
    } -> tensor<192x32x512xi16> loc(#loc44)
    %11 = tensor.empty() : tensor<1x32x512xi16> loc(#loc45)
    %12 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_5 : tensor<1x32x512xi8>) outs(%11 : tensor<1x32x512xi16>) {
    ^bb0(%in: i8 loc("w_q3"(#loc12)), %out: i16 loc("prod"(#loc14))):
      %32 = arith.extsi %in : i8 to i16 loc(#loc45)
      linalg.yield %32 : i16 loc(#loc45)
    } -> tensor<1x32x512xi16> loc(#loc45)
    %13 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%12 : tensor<1x32x512xi16>) outs(%9 : tensor<192x32x512xi16>) attrs =  {broadcastDims = array<i64: 0>} {
    ^bb0(%in: i16 loc("prod"(#loc14)), %out: i16 loc("prod"(#loc14))):
      linalg.yield %in : i16 loc(#loc45)
    } -> tensor<192x32x512xi16> loc(#loc45)
    %14 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10, %13 : tensor<192x32x512xi16>, tensor<192x32x512xi16>) outs(%10 : tensor<192x32x512xi16>) {
    ^bb0(%in: i16 loc("prod"(#loc13)), %in_13: i16 loc("prod"(#loc14)), %out: i16 loc("prod"(#loc13))):
      %32 = arith.muli %in, %in_13 : i16 loc(#loc46)
      linalg.yield %32 : i16 loc(#loc46)
    } -> tensor<192x32x512xi16> loc(#loc46)
    %expanded_6 = tensor.expand_shape %14 [[0], [1], [2, 3]] output_shape [192, 32, 16, 32] : tensor<192x32x512xi16> into tensor<192x32x16x32xi16> loc(#loc47)
    %15 = tensor.empty() : tensor<192x32x16x32xi32> loc(#loc48)
    %16 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_6 : tensor<192x32x16x32xi16>) outs(%15 : tensor<192x32x16x32xi32>) {
    ^bb0(%in: i16 loc("prod4"(#loc16)), %out: i32 loc("blk"(#loc17))):
      %32 = arith.extsi %in : i16 to i32 loc(#loc48)
      linalg.yield %32 : i32 loc(#loc48)
    } -> tensor<192x32x16x32xi32> loc(#loc48)
    %17 = tensor.empty() : tensor<192x32x16xi32> loc(#loc60)
    %18 = linalg.fill ins(%c0_i32 : i32) outs(%17 : tensor<192x32x16xi32>) -> tensor<192x32x16xi32> loc(#loc60)
    %reduced = linalg.reduce ins(%16 : tensor<192x32x16x32xi32>) outs(%18 : tensor<192x32x16xi32>) dimensions = [3] 
      (%in: i32 loc("blk"(#loc17)), %init: i32 loc(callsite(#loc18 at #loc49))) {
        %32 = arith.addi %in, %init : i32 loc(#loc60)
        linalg.yield %32 : i32 loc(#loc60)
      } loc(#loc60)
    %reinterpret_cast_7 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [192, 16], strides: [16, 1] : memref<*xf16> to memref<192x16xf16, strided<[16, 1]>> loc(#loc50)
    %alloc_8 = memref.alloc() : memref<192x16xf16> loc(#loc50)
    memref.copy %reinterpret_cast_7, %alloc_8 : memref<192x16xf16, strided<[16, 1]>> to memref<192x16xf16> loc(#loc50)
    %19 = bufferization.to_tensor %alloc_8 restrict writable : memref<192x16xf16> to tensor<192x16xf16> loc(#loc50)
    %expanded_9 = tensor.expand_shape %19 [[0], [1, 2]] output_shape [192, 1, 16] : tensor<192x16xf16> into tensor<192x1x16xf16> loc(#loc51)
    %expanded_10 = tensor.expand_shape %transposed_4 [[0, 1], [2]] output_shape [1, 32, 16] : tensor<32x16xf16> into tensor<1x32x16xf16> loc(#loc52)
    %20 = tensor.empty() : tensor<192x1x16xf32> loc(#loc53)
    %21 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_9 : tensor<192x1x16xf16>) outs(%20 : tensor<192x1x16xf32>) {
    ^bb0(%in: f16 loc("a_d3"(#loc21)), %out: f32 loc("d"(#loc23))):
      %32 = arith.extf %in : f16 to f32 loc(#loc53)
      linalg.yield %32 : f32 loc(#loc53)
    } -> tensor<192x1x16xf32> loc(#loc53)
    %22 = tensor.empty() : tensor<192x32x16xf32> loc(#loc53)
    %23 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%21 : tensor<192x1x16xf32>) outs(%22 : tensor<192x32x16xf32>) attrs =  {broadcastDims = array<i64: 1>} {
    ^bb0(%in: f32 loc("d"(#loc23)), %out: f32 loc("d"(#loc23))):
      linalg.yield %in : f32 loc(#loc53)
    } -> tensor<192x32x16xf32> loc(#loc53)
    %24 = tensor.empty() : tensor<1x32x16xf32> loc(#loc54)
    %25 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_10 : tensor<1x32x16xf16>) outs(%24 : tensor<1x32x16xf32>) {
    ^bb0(%in: f16 loc("w_d3"(#loc22)), %out: f32 loc("d"(#loc24))):
      %32 = arith.extf %in : f16 to f32 loc(#loc54)
      linalg.yield %32 : f32 loc(#loc54)
    } -> tensor<1x32x16xf32> loc(#loc54)
    %26 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25 : tensor<1x32x16xf32>) outs(%22 : tensor<192x32x16xf32>) attrs =  {broadcastDims = array<i64: 0>} {
    ^bb0(%in: f32 loc("d"(#loc24)), %out: f32 loc("d"(#loc24))):
      linalg.yield %in : f32 loc(#loc54)
    } -> tensor<192x32x16xf32> loc(#loc54)
    %27 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %26 : tensor<192x32x16xf32>, tensor<192x32x16xf32>) outs(%23 : tensor<192x32x16xf32>) {
    ^bb0(%in: f32 loc("d"(#loc23)), %in_13: f32 loc("d"(#loc24)), %out: f32 loc("d"(#loc23))):
      %32 = arith.mulf %in, %in_13 : f32 loc(#loc55)
      linalg.yield %32 : f32 loc(#loc55)
    } -> tensor<192x32x16xf32> loc(#loc55)
    %28 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%reduced : tensor<192x32x16xi32>) outs(%22 : tensor<192x32x16xf32>) {
    ^bb0(%in: i32 loc(callsite(#loc18 at #loc49)), %out: f32 loc("lvl1"(#loc26))):
      %32 = arith.sitofp %in : i32 to f32 loc(#loc56)
      linalg.yield %32 : f32 loc(#loc56)
    } -> tensor<192x32x16xf32> loc(#loc56)
    %29 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%28, %27 : tensor<192x32x16xf32>, tensor<192x32x16xf32>) outs(%28 : tensor<192x32x16xf32>) {
    ^bb0(%in: f32 loc("lvl1"(#loc26)), %in_13: f32 loc("d"(#loc25)), %out: f32 loc("lvl1"(#loc26))):
      %32 = arith.mulf %in, %in_13 : f32 loc(#loc57)
      linalg.yield %32 : f32 loc(#loc57)
    } -> tensor<192x32x16xf32> loc(#loc57)
    %30 = tensor.empty() : tensor<192x32xf32> loc(#loc61)
    %31 = linalg.fill ins(%cst : f32) outs(%30 : tensor<192x32xf32>) -> tensor<192x32xf32> loc(#loc61)
    %reduced_11 = linalg.reduce ins(%29 : tensor<192x32x16xf32>) outs(%31 : tensor<192x32xf32>) dimensions = [2] 
      (%in: f32 loc("lvl1"(#loc27)), %init: f32 loc(callsite(#loc18 at #loc58))) {
        %32 = arith.addf %in, %init : f32 loc(#loc61)
        linalg.yield %32 : f32 loc(#loc61)
      } loc(#loc61)
    %reinterpret_cast_12 = memref.reinterpret_cast %arg4 to offset: [%1], sizes: [192, 32], strides: [128, 1] : memref<*xf32> to memref<192x32xf32, strided<[128, 1], offset: ?>> loc(#loc3)
    bufferization.materialize_in_destination %reduced_11 in writable %reinterpret_cast_12 : (tensor<192x32xf32>, memref<192x32xf32, strided<[128, 1], offset: ?>>) -> () loc(#loc3)
    return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_layout.py":185:20)
#loc3 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_layout.py":275:26)
#loc4 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_layout.py":217:8)
#loc5 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_layout.py":242:18)
#loc6 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_layout.py":243:18)
#loc7 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_layout.py":244:26)
#loc8 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_layout.py":245:27)
#loc9 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_layout.py":250:18)
#loc10 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_layout.py":251:27)
#loc15 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_layout.py":260:31)
#loc20 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_layout.py":265:18)
#loc34 = loc("n_start"(#loc2))
#loc35 = loc("w_q_block_ptr"(#loc4))
#loc36 = loc("a_q"(#loc5))
#loc37 = loc("w_q"(#loc6))
#loc38 = loc("w_q"(#loc7))
#loc39 = loc("w_q"(#loc8))
#loc40 = loc("w_d"(#loc9))
#loc41 = loc("w_d"(#loc10))
#loc46 = loc("prod"(#loc15))
#loc50 = loc("a_d"(#loc20))
#loc59 = loc(fused[#loc3, #loc35])

