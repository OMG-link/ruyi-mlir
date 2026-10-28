add_mlir_dialect_library(BuddyCpu
  CpuDialect.cpp
  DEPENDS
  MLIRCpuIncGen
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)
