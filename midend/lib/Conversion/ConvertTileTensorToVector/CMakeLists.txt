add_mlir_library(ConvertTileTensorToVectorPass
  ConvertTileTensorToVector.cpp
  LINK_LIBS PUBLIC
    BuddyCpu
    BuddyTargetInfo
    MLIRArithDialect
    MLIRBufferizationDialect
    MLIRFuncDialect
    MLIRIR
    MLIRLinalgDialect
    MLIRMemRefDialect
    MLIRMemRefUtils
    MLIRPass
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRVectorDialect
)
