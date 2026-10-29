add_mlir_library(LowerCpuSchedBarrierPass
  LowerCpuSchedBarrier.cpp
  LINK_LIBS PUBLIC
    BuddyCpu
    BuddyTargetInfo
    MLIRFuncDialect
    MLIRIR
    MLIRLLVMDialect
    MLIRPass
    MLIRVectorDialect
)
