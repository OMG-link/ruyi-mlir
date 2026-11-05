add_mlir_library(BuddyIMETileOperands
  TileOperands.cpp

  LINK_LIBS PUBLIC
  BuddyCpu
  BuddyTargetInfo
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRPass
  MLIRSCFDialect
  MLIRVectorDialect
  )
