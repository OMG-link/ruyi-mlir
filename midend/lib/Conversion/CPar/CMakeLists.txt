add_mlir_library(CParPasses
  CParTransforms.cpp
  DistributeLoopsPass.cpp
  OutlineToRuntimePass.cpp

  LINK_LIBS PUBLIC
  BuddyCPar
  MLIRArithDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTransformUtils
)
