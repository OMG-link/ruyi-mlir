add_mlir_library(LowerVectorContractToIMETilePass
  LowerVectorContractToIMETile.cpp
  LINK_LIBS PUBLIC
    BuddyIME
    BuddyTargetInfo
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRPass
    MLIRTransformUtils
    MLIRVectorDialect
)
