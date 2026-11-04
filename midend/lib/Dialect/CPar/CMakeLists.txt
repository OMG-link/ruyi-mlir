add_mlir_dialect_library(BuddyCPar
  CParDialect.cpp

  DEPENDS
  MLIRCParIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
  )
