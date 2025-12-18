# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

file(GLOB globbed *.cpp)


add_mlir_library(MLIR_StreamHLS_Support
  ${globbed}
  
  DEPENDS
  MLIRDataflowIncGen
  MLIRDataflowAttributesIncGen
  MLIRDataflowInterfacesIncGen
  
  LINK_LIBS PUBLIC
  MLIRAffineUtils
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRVDialect
  MLIRTranslateLib
  MLIRMemRefDialect
  MLIRAnalysis
  MLIRAffineAnalysis
)

# target_include_directories(MLIR_StreamHLS_Support_Gurobi PRIVATE ${GUROBI_INCLUDE_DIRS})
# target_link_libraries(MLIR_StreamHLS_Support_Gurobi optimized ${GUROBI_CXX_LIBRARY}
#                         debug ${GUROBI_CXX_DEBUG_LIBRARY})
# target_link_libraries(MLIR_StreamHLS_Support_Gurobi PRIVATE ${GUROBI_LIBRARY})
