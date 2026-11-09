add_mlir_library(CParPasses
  CParTransforms.cpp
  DistributeLoopsPass.cpp
  OutlineToRuntimePass.cpp
  TcmAssignPass.cpp

  LINK_LIBS PUBLIC
  BuddyCPar
  MLIRArithDialect
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTransformUtils
)
