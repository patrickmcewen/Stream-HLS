/*
 * Copyright (c) 2024 Suhail Basalama
 *
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 */
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "streamhls/Transforms/Passes.h"
#include "streamhls/Support/Utils.h"
#include "streamhls/Support/DFG.h"
#include "streamhls/Support/TechConfig.h"

// include fstream for file IO
#include <fstream>
// include string
#include <string>

using namespace mlir;
using namespace streamhls;
using namespace dataflow;

#define DEBUG_TYPE "streamhls-node-graph-pipelining"

namespace {
struct NodeGraphPipelining
    : public NodeGraphPipeliningBase<NodeGraphPipelining> {
  NodeGraphPipelining() = default;
  NodeGraphPipelining(
    std::string argReportFile,
    std::string argLoopPermutationType,
    bool argOptimizeSchedule,
    uint argTimeLimitMinutes,
    std::string argTechConfigFile
  ) {
    reportFile = argReportFile;
    loopPermutationType = argLoopPermutationType;
    optimizeSchedule = argOptimizeSchedule;
    timeLimitMinutes = argTimeLimitMinutes;
    techConfigFile = argTechConfigFile;
  }
  void runOnOperation() override {
    // Load technology config if specified
    if (!techConfigFile.empty()) {
      initTechConfig(techConfigFile);
    }

    LLVM_DEBUG(llvm::dbgs() << "NodeGraphPipelining\n");
    auto func = getOperation();
    auto context = func.getContext();
    Block* block = &func.front();
    DFG graph(*block);
    if (!graph.init(/*mergeNodes*/ false)) {
      LLVM_DEBUG(llvm::dbgs() << "DFG init failed\n");
      return;
    }

    graph.createPermutationPerformanceModel(reportFile, timeLimitMinutes);

    graph.createPermutationPythonModel(reportFile, DFG::Default, /*optimizeOverlap*/ true);
    // graph.createPermutationPythonModel(reportFile, DFG::Default, /*optimizeOverlap*/ false);

    if(optimizeSchedule){
      graph.callPermutationSolver(reportFile, /*isMinimize*/ true);
      // graph.createPermutationPythonModel(reportFile, DFG::Minimize);
      graph.applyNodePermutations(DFG::Minimize);
    }
  }
};
} // namespace

std::unique_ptr<Pass> streamhls::createNodeGraphPipeliningPass(
  std::string reportFile,
  std::string loopPermutationType,
  bool optimizeSchedule,
  uint timeLimitMinutes,
  std::string techConfigFile
) {
  return std::make_unique<NodeGraphPipelining>(reportFile, loopPermutationType, optimizeSchedule, timeLimitMinutes, techConfigFile);
}