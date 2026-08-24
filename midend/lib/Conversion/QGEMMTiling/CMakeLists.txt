add_mlir_library(QGEMMTilingPass
  QGEMMTilingPass.cpp
  TileSizeComputer.cpp
  TilingExecutor.cpp

  LINK_LIBS PUBLIC
  QGEMMRecognizePass
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
