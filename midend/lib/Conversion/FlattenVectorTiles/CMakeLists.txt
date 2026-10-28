add_mlir_library(FlattenVectorTilesPass
  FlattenVectorTiles.cpp
  LINK_LIBS PUBLIC
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
