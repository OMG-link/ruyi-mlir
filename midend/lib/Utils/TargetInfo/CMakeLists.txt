add_mlir_library(BuddyTargetInfo
  TargetInfo.cpp
  LINK_LIBS PUBLIC
  LLVMSupport
  LLVMTargetParser
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
  )
