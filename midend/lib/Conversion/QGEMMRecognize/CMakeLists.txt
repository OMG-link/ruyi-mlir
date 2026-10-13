add_mlir_library(QGEMMRecognizePass
  QGEMMRecognizePass.cpp
  Boundary.cpp
  CandidateRoots.cpp
  LoadChain.cpp
  OperatorWalk.cpp
  Recognition.cpp
  Annotation.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRTensorDialect
)
