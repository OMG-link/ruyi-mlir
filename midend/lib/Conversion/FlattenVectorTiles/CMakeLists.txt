add_mlir_library(FlattenVectorTilesPass
  FlattenVectorTiles.cpp
  LINK_LIBS PUBLIC
    BuddyCpu
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRPass
    MLIRSCFTransforms
    MLIRTransforms
    MLIRVectorDialect
    MLIRVectorTransforms
    MLIRVectorToSCF
)
