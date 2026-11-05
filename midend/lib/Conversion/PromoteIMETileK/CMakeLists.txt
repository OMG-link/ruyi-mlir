add_mlir_library(PromoteIMETileKPass
  PromoteIMETileK.cpp

  LINK_LIBS PUBLIC
    BuddyCpu
    BuddyIMETileOperands
    BuddyTargetInfo
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRPass
    MLIRSCFDialect
    MLIRVectorDialect
)
