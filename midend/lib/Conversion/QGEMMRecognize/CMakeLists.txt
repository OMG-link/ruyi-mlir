add_mlir_library(QGEMMRecognizePass
  QGEMMRecognizePass.cpp
  Boundary.cpp
  CandidateRoots.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRTensorDialect
)
