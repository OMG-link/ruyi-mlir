// RUN: buddy-opt %s -split-input-file -verify-diagnostics

func.func @mixed(%a: tensor<4xf32>, %b: vector<4xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{must be all tensors or all vectors}}
  %r = cpu.shuffle %a, %b [0, 1, 2, 3] : tensor<4xf32>, vector<4xf32> -> tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----

func.func @trailing(%a: tensor<4x8xf32>, %b: tensor<4x4xf32>) -> tensor<2x8xf32> {
  // expected-error @+1 {{agree in every dimension but the first}}
  %r = cpu.shuffle %a, %b [0, 1] : tensor<4x8xf32>, tensor<4x4xf32> -> tensor<2x8xf32>
  return %r : tensor<2x8xf32>
}

// -----

func.func @mask_size(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{first dimension must equal the mask size}}
  %r = cpu.shuffle %a, %b [0, 1] : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----

func.func @range(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<2xf32> {
  // expected-error @+1 {{mask element 8 is out of range [-1, 8)}}
  %r = cpu.shuffle %a, %b [0, 8] : tensor<4xf32>, tensor<4xf32> -> tensor<2xf32>
  return %r : tensor<2xf32>
}
