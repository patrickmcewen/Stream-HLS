/*
 * Copyright (c) 2024 Suhail Basalama
 *
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 */
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/IR/IntegerSet.h"


#include "streamhls/Transforms/Passes.h"
#include "streamhls/Support/Utils.h"
#include "streamhls/Support/AffineMemAccess.h"
#include "streamhls/Support/DFG.h"
#include "streamhls/Support/TechConfig.h"

#include <regex>

using namespace mlir;
using namespace streamhls;
using namespace dataflow;

#define DEBUG_TYPE "streamhls-node-level-parallelization"

namespace {
struct NodeParallelization : public NodeParallelizationBase<NodeParallelization> {
  NodeParallelization() = default;
  NodeParallelization(
    std::string argReportFile,
    bool argParallelizeNodes,
    uint argDSPs,
    uint argTilingLimit,
    uint argTimeLimitMinutes,
    std::string argTechConfigFile
  ) {
    reportFile = argReportFile;
    parallelizeNodes = argParallelizeNodes;
    DSPs = argDSPs;
    tilingLimit = argTilingLimit;
    timeLimitMinutes = argTimeLimitMinutes;
    techConfigFile = argTechConfigFile;
  }
  void runOnOperation() override {
    // Load technology config if specified
    if (!techConfigFile.empty()) {
      initTechConfig(techConfigFile);
    }

    func::FuncOp func = getOperation();
    auto context = func.getContext();
    OpBuilder builder(context);
    auto block = &func.front();
    DFG graph(*block);
    if (!graph.init(false)) {
      LLVM_DEBUG(llvm::dbgs() << "DFG init failed\n");
      return;
    }
    graph.createParallelizationPerformanceModel(reportFile, DSPs, tilingLimit, timeLimitMinutes);

    graph.callParallelizationSolver(reportFile);
    // graph.createParallelizationPythonModel(reportFile);
    // graph.findUniformParallelSolution();

    graph.applyNodeParallelization();
    // SmallVector<TypedValue<mlir::MemRefType>> memRefs;
    // for(auto arg : func.getArguments()){
    //   // dynamic cast to TypedValue<MemRefType>
    //   if(auto memref = dyn_cast<TypedValue<mlir::MemRefType>>(arg)){
    //     memRefs.push_back(memref);
    //   }
    // }
    // block->walk([&](memref::AllocOp allocOp){
    //   auto memref = allocOp.getResult();
    //   memRefs.push_back(memref);
    // });
  }
};
}

std::unique_ptr<Pass> streamhls::createNodeParallelizationPass(
  std::string reportFile,
  bool parallelizeNodes,
  uint DSPs,
  uint tilingLimit,
  uint timeLimitMinutes,
  std::string techConfigFile
) {
  return std::make_unique<NodeParallelization>(
    reportFile,
    parallelizeNodes,
    DSPs,
    tilingLimit,
    timeLimitMinutes,
    techConfigFile
  );
}