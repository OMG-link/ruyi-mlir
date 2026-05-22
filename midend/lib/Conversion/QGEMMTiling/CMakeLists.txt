add_mlir_library(QGEMMTilingPass
  QGEMMTilingPass.cpp
  ReduceChainAnalysis.cpp
  RootFilter.cpp
  ContractionSubgraph.cpp
  DimensionClassifier.cpp
  TileSizeComputer.cpp
  TilingExecutor.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTensorDialect
  MLIRFuncDialect
  MLIRIR
  MLIRTilingInterface
  MLIRTransformUtils
)
