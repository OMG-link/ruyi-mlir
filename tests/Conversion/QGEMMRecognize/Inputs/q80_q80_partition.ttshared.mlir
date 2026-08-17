#loc = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_partition.py":163:0)
#loc7 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_partition.py":242:26)
#loc8 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_partition.py":244:26)
#loc9 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_partition.py":248:19)
#loc10 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_partition.py":248:39)
#loc12 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_partition.py":251:29)
#loc13 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_partition.py":253:32)
#loc14 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/triton-riscv/.venv/lib/python3.10/site-packages/triton/language/standard.py":291:36)
#loc15 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_partition.py":253:23)
#loc17 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_partition.py":256:25)
#loc19 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_partition.py":259:25)
#loc20 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_partition.py":261:27)
#loc21 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_partition.py":262:15)
#loc22 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_partition.py":262:36)
#loc23 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_partition.py":262:29)
#loc24 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_partition.py":262:8)
#loc25 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_partition.py":266:15)
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#loc26 = loc("a_q_ptr"(#loc))
#loc27 = loc("a_d_ptr"(#loc))
#loc28 = loc("w_q_ptr"(#loc))
#loc29 = loc("w_d_ptr"(#loc))
#loc30 = loc("c_ptr"(#loc))
#loc35 = loc("a_q3"(#loc7))
#loc36 = loc("w_q3"(#loc8))
#loc37 = loc("prod"(#loc9))
#loc38 = loc("prod"(#loc10))
#loc40 = loc("prod4"(#loc12))
#loc41 = loc("block_sum"(#loc13))
#loc42 = loc("block_sum"(#loc15))
#loc44 = loc("a_d"(#loc17))
#loc46 = loc("w_d"(#loc19))
#loc47 = loc("level_1"(#loc20))
#loc48 = loc("level_1"(#loc21))
#loc49 = loc("level_1"(#loc22))
#loc50 = loc("level_1"(#loc23))
#loc51 = loc("level_1"(#loc24))
#loc52 = loc("c"(#loc25))
#loc54 = loc(callsite(#loc14 at #loc42))
#loc55 = loc(callsite(#loc14 at #loc52))
module {
  func.func @q80_q80_partition_m(%arg0: memref<*xi8> {tt.divisibility = 16 : i32} loc("a_q_ptr"(#loc)), %arg1: memref<*xf16> {tt.divisibility = 16 : i32} loc("a_d_ptr"(#loc)), %arg2: memref<*xi8> {tt.divisibility = 16 : i32} loc("w_q_ptr"(#loc)), %arg3: memref<*xf16> {tt.divisibility = 16 : i32} loc("w_d_ptr"(#loc)), %arg4: memref<*xf32> {tt.divisibility = 16 : i32} loc("c_ptr"(#loc)), %arg5: i32 loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_partition.py":163:0), %arg6: i32 loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_partition.py":163:0), %arg7: i32 loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_partition.py":163:0), %arg8: i32 loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_partition.py":163:0), %arg9: i32 loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_partition.py":163:0), %arg10: i32 loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_partition.py":163:0)) {
    %c16_i32 = arith.constant 16 : i32 loc(#loc1)
    %c256 = arith.constant 256 : index loc(#loc1)
    %c8 = arith.constant 8 : index loc(#loc1)
    %c128 = arith.constant 128 : index loc(#loc1)
    %c0_i32 = arith.constant 0 : i32 loc(#loc1)
    %cst = arith.constant 0.000000e+00 : f32 loc(#loc1)
    %0 = arith.muli %arg8, %c16_i32 : i32 loc(#loc31)
    %1 = arith.index_cast %0 : i32 to index loc(#loc53)
    %2 = arith.muli %1, %c256 : index loc(#loc33)
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%2], sizes: [16, 256], strides: [256, 1] : memref<*xi8> to memref<16x256xi8, strided<[256, 1], offset: ?>> loc(#loc33)
    %alloc = memref.alloc() : memref<16x256xi8> loc(#loc33)
    memref.copy %reinterpret_cast, %alloc : memref<16x256xi8, strided<[256, 1], offset: ?>> to memref<16x256xi8> loc(#loc33)
    %3 = bufferization.to_tensor %alloc restrict writable : memref<16x256xi8> to tensor<16x256xi8> loc(#loc33)
    %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [128, 256], strides: [256, 1] : memref<*xi8> to memref<128x256xi8, strided<[256, 1]>> loc(#loc34)
    %alloc_1 = memref.alloc() : memref<128x256xi8> loc(#loc34)
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<128x256xi8, strided<[256, 1]>> to memref<128x256xi8> loc(#loc34)
    %4 = bufferization.to_tensor %alloc_1 restrict writable : memref<128x256xi8> to tensor<128x256xi8> loc(#loc34)
    %expanded = tensor.expand_shape %3 [[0], [1, 2]] output_shape [16, 1, 256] : tensor<16x256xi8> into tensor<16x1x256xi8> loc(#loc35)
    %expanded_2 = tensor.expand_shape %4 [[0, 1], [2]] output_shape [1, 128, 256] : tensor<128x256xi8> into tensor<1x128x256xi8> loc(#loc36)
    %5 = tensor.empty() : tensor<16x1x256xi16> loc(#loc37)
    %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded : tensor<16x1x256xi8>) outs(%5 : tensor<16x1x256xi16>) {
    ^bb0(%in: i8 loc("a_q3"(#loc7)), %out: i16 loc("prod"(#loc9))):
      %33 = arith.extsi %in : i8 to i16 loc(#loc37)
      linalg.yield %33 : i16 loc(#loc37)
    } -> tensor<16x1x256xi16> loc(#loc37)
    %7 = tensor.empty() : tensor<16x128x256xi16> loc(#loc37)
    %8 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6 : tensor<16x1x256xi16>) outs(%7 : tensor<16x128x256xi16>) attrs =  {broadcastDims = array<i64: 1>} {
    ^bb0(%in: i16 loc("prod"(#loc9)), %out: i16 loc("prod"(#loc9))):
      linalg.yield %in : i16 loc(#loc37)
    } -> tensor<16x128x256xi16> loc(#loc37)
    %9 = tensor.empty() : tensor<1x128x256xi16> loc(#loc38)
    %10 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_2 : tensor<1x128x256xi8>) outs(%9 : tensor<1x128x256xi16>) {
    ^bb0(%in: i8 loc("w_q3"(#loc8)), %out: i16 loc("prod"(#loc10))):
      %33 = arith.extsi %in : i8 to i16 loc(#loc38)
      linalg.yield %33 : i16 loc(#loc38)
    } -> tensor<1x128x256xi16> loc(#loc38)
    %11 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10 : tensor<1x128x256xi16>) outs(%7 : tensor<16x128x256xi16>) attrs =  {broadcastDims = array<i64: 0>} {
    ^bb0(%in: i16 loc("prod"(#loc10)), %out: i16 loc("prod"(#loc10))):
      linalg.yield %in : i16 loc(#loc38)
    } -> tensor<16x128x256xi16> loc(#loc38)
    %12 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8, %11 : tensor<16x128x256xi16>, tensor<16x128x256xi16>) outs(%8 : tensor<16x128x256xi16>) {
    ^bb0(%in: i16 loc("prod"(#loc9)), %in_12: i16 loc("prod"(#loc10)), %out: i16 loc("prod"(#loc9))):
      %33 = arith.muli %in, %in_12 : i16 loc(#loc39)
      linalg.yield %33 : i16 loc(#loc39)
    } -> tensor<16x128x256xi16> loc(#loc39)
    %expanded_3 = tensor.expand_shape %12 [[0], [1], [2, 3]] output_shape [16, 128, 8, 32] : tensor<16x128x256xi16> into tensor<16x128x8x32xi16> loc(#loc40)
    %13 = tensor.empty() : tensor<16x128x8x32xi32> loc(#loc41)
    %14 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_3 : tensor<16x128x8x32xi16>) outs(%13 : tensor<16x128x8x32xi32>) {
    ^bb0(%in: i16 loc("prod4"(#loc12)), %out: i32 loc("block_sum"(#loc13))):
      %33 = arith.extsi %in : i16 to i32 loc(#loc41)
      linalg.yield %33 : i32 loc(#loc41)
    } -> tensor<16x128x8x32xi32> loc(#loc41)
    %15 = tensor.empty() : tensor<16x128x8xi32> loc(#loc54)
    %16 = linalg.fill ins(%c0_i32 : i32) outs(%15 : tensor<16x128x8xi32>) -> tensor<16x128x8xi32> loc(#loc54)
    %reduced = linalg.reduce ins(%14 : tensor<16x128x8x32xi32>) outs(%16 : tensor<16x128x8xi32>) dimensions = [3] 
      (%in: i32 loc("block_sum"(#loc13)), %init: i32 loc(callsite(#loc14 at #loc42))) {
        %33 = arith.addi %in, %init : i32 loc(#loc54)
        linalg.yield %33 : i32 loc(#loc54)
      } loc(#loc54)
    %17 = arith.muli %1, %c8 : index loc(#loc43)
    %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%17], sizes: [16, 8], strides: [8, 1] : memref<*xf16> to memref<16x8xf16, strided<[8, 1], offset: ?>> loc(#loc43)
    %alloc_5 = memref.alloc() : memref<16x8xf16> loc(#loc43)
    memref.copy %reinterpret_cast_4, %alloc_5 : memref<16x8xf16, strided<[8, 1], offset: ?>> to memref<16x8xf16> loc(#loc43)
    %18 = bufferization.to_tensor %alloc_5 restrict writable : memref<16x8xf16> to tensor<16x8xf16> loc(#loc43)
    %expanded_6 = tensor.expand_shape %18 [[0], [1, 2]] output_shape [16, 1, 8] : tensor<16x8xf16> into tensor<16x1x8xf16> loc(#loc44)
    %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [128, 8], strides: [8, 1] : memref<*xf16> to memref<128x8xf16, strided<[8, 1]>> loc(#loc45)
    %alloc_8 = memref.alloc() : memref<128x8xf16> loc(#loc45)
    memref.copy %reinterpret_cast_7, %alloc_8 : memref<128x8xf16, strided<[8, 1]>> to memref<128x8xf16> loc(#loc45)
    %19 = bufferization.to_tensor %alloc_8 restrict writable : memref<128x8xf16> to tensor<128x8xf16> loc(#loc45)
    %expanded_9 = tensor.expand_shape %19 [[0, 1], [2]] output_shape [1, 128, 8] : tensor<128x8xf16> into tensor<1x128x8xf16> loc(#loc46)
    %20 = tensor.empty() : tensor<16x128x8xf32> loc(#loc47)
    %21 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%reduced : tensor<16x128x8xi32>) outs(%20 : tensor<16x128x8xf32>) {
    ^bb0(%in: i32 loc(callsite(#loc14 at #loc42)), %out: f32 loc("level_1"(#loc20))):
      %33 = arith.sitofp %in : i32 to f32 loc(#loc47)
      linalg.yield %33 : f32 loc(#loc47)
    } -> tensor<16x128x8xf32> loc(#loc47)
    %22 = tensor.empty() : tensor<16x1x8xf32> loc(#loc48)
    %23 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_6 : tensor<16x1x8xf16>) outs(%22 : tensor<16x1x8xf32>) {
    ^bb0(%in: f16 loc("a_d"(#loc17)), %out: f32 loc("level_1"(#loc21))):
      %33 = arith.extf %in : f16 to f32 loc(#loc48)
      linalg.yield %33 : f32 loc(#loc48)
    } -> tensor<16x1x8xf32> loc(#loc48)
    %24 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23 : tensor<16x1x8xf32>) outs(%20 : tensor<16x128x8xf32>) attrs =  {broadcastDims = array<i64: 1>} {
    ^bb0(%in: f32 loc("level_1"(#loc21)), %out: f32 loc("level_1"(#loc21))):
      linalg.yield %in : f32 loc(#loc48)
    } -> tensor<16x128x8xf32> loc(#loc48)
    %25 = tensor.empty() : tensor<1x128x8xf32> loc(#loc49)
    %26 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_9 : tensor<1x128x8xf16>) outs(%25 : tensor<1x128x8xf32>) {
    ^bb0(%in: f16 loc("w_d"(#loc19)), %out: f32 loc("level_1"(#loc22))):
      %33 = arith.extf %in : f16 to f32 loc(#loc49)
      linalg.yield %33 : f32 loc(#loc49)
    } -> tensor<1x128x8xf32> loc(#loc49)
    %27 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26 : tensor<1x128x8xf32>) outs(%20 : tensor<16x128x8xf32>) attrs =  {broadcastDims = array<i64: 0>} {
    ^bb0(%in: f32 loc("level_1"(#loc22)), %out: f32 loc("level_1"(#loc22))):
      linalg.yield %in : f32 loc(#loc49)
    } -> tensor<16x128x8xf32> loc(#loc49)
    %28 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %27 : tensor<16x128x8xf32>, tensor<16x128x8xf32>) outs(%24 : tensor<16x128x8xf32>) {
    ^bb0(%in: f32 loc("level_1"(#loc21)), %in_12: f32 loc("level_1"(#loc22)), %out: f32 loc("level_1"(#loc21))):
      %33 = arith.mulf %in, %in_12 : f32 loc(#loc50)
      linalg.yield %33 : f32 loc(#loc50)
    } -> tensor<16x128x8xf32> loc(#loc50)
    %29 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%21, %28 : tensor<16x128x8xf32>, tensor<16x128x8xf32>) outs(%21 : tensor<16x128x8xf32>) {
    ^bb0(%in: f32 loc("level_1"(#loc20)), %in_12: f32 loc("level_1"(#loc23)), %out: f32 loc("level_1"(#loc20))):
      %33 = arith.mulf %in, %in_12 : f32 loc(#loc51)
      linalg.yield %33 : f32 loc(#loc51)
    } -> tensor<16x128x8xf32> loc(#loc51)
    %30 = tensor.empty() : tensor<16x128xf32> loc(#loc55)
    %31 = linalg.fill ins(%cst : f32) outs(%30 : tensor<16x128xf32>) -> tensor<16x128xf32> loc(#loc55)
    %reduced_10 = linalg.reduce ins(%29 : tensor<16x128x8xf32>) outs(%31 : tensor<16x128xf32>) dimensions = [2] 
      (%in: f32 loc("level_1"(#loc24)), %init: f32 loc(callsite(#loc14 at #loc52))) {
        %33 = arith.addf %in, %init : f32 loc(#loc55)
        linalg.yield %33 : f32 loc(#loc55)
      } loc(#loc55)
    %32 = arith.muli %1, %c128 : index loc(#loc3)
    %reinterpret_cast_11 = memref.reinterpret_cast %arg4 to offset: [%32], sizes: [16, 128], strides: [128, 1] : memref<*xf32> to memref<16x128xf32, strided<[128, 1], offset: ?>> loc(#loc3)
    bufferization.materialize_in_destination %reduced_10 in writable %reinterpret_cast_11 : (tensor<16x128xf32>, memref<16x128xf32, strided<[128, 1], offset: ?>>) -> () loc(#loc3)
    return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_partition.py":185:22)
#loc3 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_partition.py":267:26)
#loc4 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_partition.py":202:8)
#loc5 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_partition.py":238:18)
#loc6 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_partition.py":239:18)
#loc11 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_partition.py":248:31)
#loc16 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_partition.py":255:18)
#loc18 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_partition.py":258:18)
#loc31 = loc("m_start"(#loc2))
#loc32 = loc("a_q_block_ptr"(#loc4))
#loc33 = loc("a_q"(#loc5))
#loc34 = loc("w_q"(#loc6))
#loc39 = loc("prod"(#loc11))
#loc43 = loc("a_d"(#loc16))
#loc45 = loc("w_d"(#loc18))
#loc53 = loc(fused[#loc3, #loc32])

