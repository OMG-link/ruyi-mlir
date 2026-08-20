add_mlir_library(QGEMMDimTransformPass
  QGEMMDimTransformPass.cpp

  LINK_LIBS PUBLIC
  QGEMMRecognizePass
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMemRefDialect
  MLIRTensorDialect
  MLIRTensorTransforms
  MLIRTransforms
)
