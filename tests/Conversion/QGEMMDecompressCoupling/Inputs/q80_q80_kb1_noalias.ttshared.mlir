#loc = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_strict.py":48:0)
#loc7 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_strict.py":121:26)
#loc8 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_strict.py":123:26)
#loc9 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_strict.py":127:19)
#loc10 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_strict.py":127:39)
#loc12 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_strict.py":130:29)
#loc13 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_strict.py":133:26)
#loc14 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/triton-riscv/.venv/lib/python3.10/site-packages/triton/language/standard.py":291:36)
#loc15 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_strict.py":133:17)
#loc17 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_strict.py":138:26)
#loc19 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_strict.py":141:26)
#loc20 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_strict.py":144:16)
#loc21 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_strict.py":144:38)
#loc22 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_strict.py":144:30)
#loc23 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_strict.py":145:18)
#loc24 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_strict.py":145:32)
#loc25 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_strict.py":148:15)
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2) -> (d0, 0, 0)>
#map5 = affine_map<(d0, d1, d2) -> (0, d1, 0)>
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
#loc41 = loc("blk"(#loc13))
#loc42 = loc("blk"(#loc15))
#loc44 = loc("a_d3"(#loc17))
#loc46 = loc("w_d3"(#loc19))
#loc47 = loc("d"(#loc20))
#loc48 = loc("d"(#loc21))
#loc49 = loc("d"(#loc22))
#loc50 = loc("lvl1"(#loc23))
#loc51 = loc("lvl1"(#loc24))
#loc52 = loc("c"(#loc25))
#loc54 = loc(callsite(#loc14 at #loc42))
#loc55 = loc(callsite(#loc14 at #loc52))
module {
  func.func @q80_q80_strict(%arg0: memref<*xi8> {llvm.noalias, tt.divisibility = 16 : i32} loc("a_q_ptr"(#loc)), %arg1: memref<*xf16> {llvm.noalias, tt.divisibility = 16 : i32} loc("a_d_ptr"(#loc)), %arg2: memref<*xi8> {llvm.noalias, tt.divisibility = 16 : i32} loc("w_q_ptr"(#loc)), %arg3: memref<*xf16> {llvm.noalias, tt.divisibility = 16 : i32} loc("w_d_ptr"(#loc)), %arg4: memref<*xf32> {llvm.noalias, tt.divisibility = 16 : i32} loc("c_ptr"(#loc)), %arg5: i32 loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_strict.py":48:0), %arg6: i32 loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_strict.py":48:0), %arg7: i32 loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_strict.py":48:0), %arg8: i32 loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_strict.py":48:0), %arg9: i32 loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_strict.py":48:0), %arg10: i32 loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_strict.py":48:0)) {
    %c32_i32 = arith.constant 32 : i32 loc(#loc1)
    %c32 = arith.constant 32 : index loc(#loc1)
    %c0_i32 = arith.constant 0 : i32 loc(#loc1)
    %cst = arith.constant 0.000000e+00 : f32 loc(#loc1)
    %0 = arith.muli %arg8, %c32_i32 : i32 loc(#loc31)
    %1 = arith.index_cast %0 : i32 to index loc(#loc53)
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [64, 32], strides: [32, 1] : memref<*xi8> to memref<64x32xi8, strided<[32, 1]>> loc(#loc33)
    %alloc = memref.alloc() : memref<64x32xi8> loc(#loc33)
    memref.copy %reinterpret_cast, %alloc : memref<64x32xi8, strided<[32, 1]>> to memref<64x32xi8> loc(#loc33)
    %2 = bufferization.to_tensor %alloc restrict writable : memref<64x32xi8> to tensor<64x32xi8> loc(#loc33)
    %3 = arith.muli %1, %c32 : index loc(#loc34)
    %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%3], sizes: [32, 32], strides: [32, 1] : memref<*xi8> to memref<32x32xi8, strided<[32, 1], offset: ?>> loc(#loc34)
    %alloc_1 = memref.alloc() : memref<32x32xi8> loc(#loc34)
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<32x32xi8, strided<[32, 1], offset: ?>> to memref<32x32xi8> loc(#loc34)
    %4 = bufferization.to_tensor %alloc_1 restrict writable : memref<32x32xi8> to tensor<32x32xi8> loc(#loc34)
    %expanded = tensor.expand_shape %2 [[0], [1, 2]] output_shape [64, 1, 32] : tensor<64x32xi8> into tensor<64x1x32xi8> loc(#loc35)
    %expanded_2 = tensor.expand_shape %4 [[0, 1], [2]] output_shape [1, 32, 32] : tensor<32x32xi8> into tensor<1x32x32xi8> loc(#loc36)
    %5 = tensor.empty() : tensor<64x1x32xi16> loc(#loc37)
    %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded : tensor<64x1x32xi8>) outs(%5 : tensor<64x1x32xi16>) {
    ^bb0(%in: i8 loc("a_q3"(#loc7)), %out: i16 loc("prod"(#loc9))):
      %31 = arith.extsi %in : i8 to i16 loc(#loc37)
      linalg.yield %31 : i16 loc(#loc37)
    } -> tensor<64x1x32xi16> loc(#loc37)
    %7 = tensor.empty() : tensor<64x32x32xi16> loc(#loc37)
    %8 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6 : tensor<64x1x32xi16>) outs(%7 : tensor<64x32x32xi16>) attrs =  {broadcastDims = array<i64: 1>} {
    ^bb0(%in: i16 loc("prod"(#loc9)), %out: i16 loc("prod"(#loc9))):
      linalg.yield %in : i16 loc(#loc37)
    } -> tensor<64x32x32xi16> loc(#loc37)
    %9 = tensor.empty() : tensor<1x32x32xi16> loc(#loc38)
    %10 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_2 : tensor<1x32x32xi8>) outs(%9 : tensor<1x32x32xi16>) {
    ^bb0(%in: i8 loc("w_q3"(#loc8)), %out: i16 loc("prod"(#loc10))):
      %31 = arith.extsi %in : i8 to i16 loc(#loc38)
      linalg.yield %31 : i16 loc(#loc38)
    } -> tensor<1x32x32xi16> loc(#loc38)
    %11 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10 : tensor<1x32x32xi16>) outs(%7 : tensor<64x32x32xi16>) attrs =  {broadcastDims = array<i64: 0>} {
    ^bb0(%in: i16 loc("prod"(#loc10)), %out: i16 loc("prod"(#loc10))):
      linalg.yield %in : i16 loc(#loc38)
    } -> tensor<64x32x32xi16> loc(#loc38)
    %12 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8, %11 : tensor<64x32x32xi16>, tensor<64x32x32xi16>) outs(%8 : tensor<64x32x32xi16>) {
    ^bb0(%in: i16 loc("prod"(#loc9)), %in_12: i16 loc("prod"(#loc10)), %out: i16 loc("prod"(#loc9))):
      %31 = arith.muli %in, %in_12 : i16 loc(#loc39)
      linalg.yield %31 : i16 loc(#loc39)
    } -> tensor<64x32x32xi16> loc(#loc39)
    %expanded_3 = tensor.expand_shape %12 [[0], [1], [2, 3]] output_shape [64, 32, 1, 32] : tensor<64x32x32xi16> into tensor<64x32x1x32xi16> loc(#loc40)
    %13 = tensor.empty() : tensor<64x32x1x32xi32> loc(#loc41)
    %14 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_3 : tensor<64x32x1x32xi16>) outs(%13 : tensor<64x32x1x32xi32>) {
    ^bb0(%in: i16 loc("prod4"(#loc12)), %out: i32 loc("blk"(#loc13))):
      %31 = arith.extsi %in : i16 to i32 loc(#loc41)
      linalg.yield %31 : i32 loc(#loc41)
    } -> tensor<64x32x1x32xi32> loc(#loc41)
    %15 = tensor.empty() : tensor<64x32x1xi32> loc(#loc54)
    %16 = linalg.fill ins(%c0_i32 : i32) outs(%15 : tensor<64x32x1xi32>) -> tensor<64x32x1xi32> loc(#loc54)
    %reduced = linalg.reduce ins(%14 : tensor<64x32x1x32xi32>) outs(%16 : tensor<64x32x1xi32>) dimensions = [3] 
      (%in: i32 loc("blk"(#loc13)), %init: i32 loc(callsite(#loc14 at #loc42))) {
        %31 = arith.addi %in, %init : i32 loc(#loc54)
        linalg.yield %31 : i32 loc(#loc54)
      } loc(#loc54)
    %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [64, 1], strides: [1, 1] : memref<*xf16> to memref<64x1xf16, strided<[1, 1]>> loc(#loc43)
    %alloc_5 = memref.alloc() : memref<64x1xf16> loc(#loc43)
    memref.copy %reinterpret_cast_4, %alloc_5 : memref<64x1xf16, strided<[1, 1]>> to memref<64x1xf16> loc(#loc43)
    %17 = bufferization.to_tensor %alloc_5 restrict writable : memref<64x1xf16> to tensor<64x1xf16> loc(#loc43)
    %expanded_6 = tensor.expand_shape %17 [[0], [1, 2]] output_shape [64, 1, 1] : tensor<64x1xf16> into tensor<64x1x1xf16> loc(#loc44)
    %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%1], sizes: [32, 1], strides: [1, 1] : memref<*xf16> to memref<32x1xf16, strided<[1, 1], offset: ?>> loc(#loc45)
    %alloc_8 = memref.alloc() : memref<32x1xf16> loc(#loc45)
    memref.copy %reinterpret_cast_7, %alloc_8 : memref<32x1xf16, strided<[1, 1], offset: ?>> to memref<32x1xf16> loc(#loc45)
    %18 = bufferization.to_tensor %alloc_8 restrict writable : memref<32x1xf16> to tensor<32x1xf16> loc(#loc45)
    %expanded_9 = tensor.expand_shape %18 [[0, 1], [2]] output_shape [1, 32, 1] : tensor<32x1xf16> into tensor<1x32x1xf16> loc(#loc46)
    %19 = tensor.empty() : tensor<64x1x1xf32> loc(#loc47)
    %20 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_6 : tensor<64x1x1xf16>) outs(%19 : tensor<64x1x1xf32>) {
    ^bb0(%in: f16 loc("a_d3"(#loc17)), %out: f32 loc("d"(#loc20))):
      %31 = arith.extf %in : f16 to f32 loc(#loc47)
      linalg.yield %31 : f32 loc(#loc47)
    } -> tensor<64x1x1xf32> loc(#loc47)
    %21 = tensor.empty() : tensor<64x32x1xf32> loc(#loc47)
    %22 = linalg.generic {indexing_maps = [#map4, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%20 : tensor<64x1x1xf32>) outs(%21 : tensor<64x32x1xf32>) attrs =  {broadcastDims = array<i64: 1>} {
    ^bb0(%in: f32 loc("d"(#loc20)), %out: f32 loc("d"(#loc20))):
      linalg.yield %in : f32 loc(#loc47)
    } -> tensor<64x32x1xf32> loc(#loc47)
    %23 = tensor.empty() : tensor<1x32x1xf32> loc(#loc48)
    %24 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_9 : tensor<1x32x1xf16>) outs(%23 : tensor<1x32x1xf32>) {
    ^bb0(%in: f16 loc("w_d3"(#loc19)), %out: f32 loc("d"(#loc21))):
      %31 = arith.extf %in : f16 to f32 loc(#loc48)
      linalg.yield %31 : f32 loc(#loc48)
    } -> tensor<1x32x1xf32> loc(#loc48)
    %25 = linalg.generic {indexing_maps = [#map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24 : tensor<1x32x1xf32>) outs(%21 : tensor<64x32x1xf32>) attrs =  {broadcastDims = array<i64: 0>} {
    ^bb0(%in: f32 loc("d"(#loc21)), %out: f32 loc("d"(#loc21))):
      linalg.yield %in : f32 loc(#loc48)
    } -> tensor<64x32x1xf32> loc(#loc48)
    %26 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%22, %25 : tensor<64x32x1xf32>, tensor<64x32x1xf32>) outs(%22 : tensor<64x32x1xf32>) {
    ^bb0(%in: f32 loc("d"(#loc20)), %in_12: f32 loc("d"(#loc21)), %out: f32 loc("d"(#loc20))):
      %31 = arith.mulf %in, %in_12 : f32 loc(#loc49)
      linalg.yield %31 : f32 loc(#loc49)
    } -> tensor<64x32x1xf32> loc(#loc49)
    %27 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%reduced : tensor<64x32x1xi32>) outs(%21 : tensor<64x32x1xf32>) {
    ^bb0(%in: i32 loc(callsite(#loc14 at #loc42)), %out: f32 loc("lvl1"(#loc23))):
      %31 = arith.sitofp %in : i32 to f32 loc(#loc50)
      linalg.yield %31 : f32 loc(#loc50)
    } -> tensor<64x32x1xf32> loc(#loc50)
    %28 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%27, %26 : tensor<64x32x1xf32>, tensor<64x32x1xf32>) outs(%27 : tensor<64x32x1xf32>) {
    ^bb0(%in: f32 loc("lvl1"(#loc23)), %in_12: f32 loc("d"(#loc22)), %out: f32 loc("lvl1"(#loc23))):
      %31 = arith.mulf %in, %in_12 : f32 loc(#loc51)
      linalg.yield %31 : f32 loc(#loc51)
    } -> tensor<64x32x1xf32> loc(#loc51)
    %29 = tensor.empty() : tensor<64x32xf32> loc(#loc55)
    %30 = linalg.fill ins(%cst : f32) outs(%29 : tensor<64x32xf32>) -> tensor<64x32xf32> loc(#loc55)
    %reduced_10 = linalg.reduce ins(%28 : tensor<64x32x1xf32>) outs(%30 : tensor<64x32xf32>) dimensions = [2] 
      (%in: f32 loc("lvl1"(#loc24)), %init: f32 loc(callsite(#loc14 at #loc52))) {
        %31 = arith.addf %in, %init : f32 loc(#loc55)
        linalg.yield %31 : f32 loc(#loc55)
      } loc(#loc55)
    %reinterpret_cast_11 = memref.reinterpret_cast %arg4 to offset: [%1], sizes: [64, 32], strides: [32, 1] : memref<*xf32> to memref<64x32xf32, strided<[32, 1], offset: ?>> loc(#loc3)
    bufferization.materialize_in_destination %reduced_10 in writable %reinterpret_cast_11 : (tensor<64x32xf32>, memref<64x32xf32, strided<[32, 1], offset: ?>>) -> () loc(#loc3)
    return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_strict.py":65:20)
#loc3 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_strict.py":150:26)
#loc4 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_strict.py":96:8)
#loc5 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_strict.py":117:18)
#loc6 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_strict.py":118:18)
#loc11 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_strict.py":127:31)
#loc16 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_strict.py":137:18)
#loc18 = loc("/home/linkai/projects/triton-riscv/triton-shared/RuyiAI/benchmark/kernels/q80_q80_strict.py":140:18)
#loc31 = loc("n_start"(#loc2))
#loc32 = loc("w_q_block_ptr"(#loc4))
#loc33 = loc("a_q"(#loc5))
#loc34 = loc("w_q"(#loc6))
#loc39 = loc("prod"(#loc11))
#loc43 = loc("a_d"(#loc16))
#loc45 = loc("w_d"(#loc18))
#loc53 = loc(fused[#loc3, #loc32])

