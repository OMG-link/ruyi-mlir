add_mlir_library(BuddyIMETileOperands
  TileOperands.cpp

  LINK_LIBS PUBLIC
  BuddyTargetInfo
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRPass
  MLIRVectorDialect
  )
