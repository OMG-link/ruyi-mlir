add_mlir_library(PromoteIMETileKPass
  PromoteIMETileK.cpp

  LINK_LIBS PUBLIC
    BuddyIMETileOperands
    BuddyTargetInfo
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRPass
    MLIRVectorDialect
)
