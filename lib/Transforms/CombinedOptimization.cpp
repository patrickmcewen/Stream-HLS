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
struct CombinedOptimization : public CombinedOptimizationBase<CombinedOptimization> {
  CombinedOptimization() = default;
  CombinedOptimization(
    std::string argReportFile,
    bool argParallelizeNodes,
    uint argDSPs,
    uint argTilingLimit,
    uint argTimeLimitMinutes,
    std::string argTechConfigFile,
    std::string argSolutionFile
  ) {
    reportFile = argReportFile;
    parallelizeNodes = argParallelizeNodes;
    DSPs = argDSPs;
    tilingLimit = argTilingLimit;
    timeLimitMinutes = argTimeLimitMinutes;
    techConfigFile = argTechConfigFile;
    solutionFile = argSolutionFile;
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

    if (!solutionFile.empty()) {
      // REPLAY MODE: load pre-computed decisions, skip GUROBI solve
      llvm::dbgs() << "Replay mode: loading solution from " << solutionFile << "\n";
      graph.loadSolutionFromFile(solutionFile);
    } else {
      // NORMAL MODE: solve with GUROBI
      graph.createCombinedOptimizationPerformanceModel(reportFile, DSPs, tilingLimit, timeLimitMinutes);
      graph.callCombinedOptimizationSolver(reportFile);
      graph.createCombinedOptimizationPythonModel(reportFile);
      // Save solution for future replay
      graph.saveSolutionToFile(reportFile + "_solution.json");
    }

    graph.applyCombinedOptimization();

  }
};
}

std::unique_ptr<Pass> streamhls::createCombinedOptimizationPass(
  std::string reportFile,
  bool parallelizeNodes,
  uint DSPs,
  uint tilingLimit,
  uint timeLimitMinutes,
  std::string techConfigFile,
  std::string solutionFile
) {
  return std::make_unique<CombinedOptimization>(
    reportFile,
    parallelizeNodes,
    DSPs,
    tilingLimit,
    timeLimitMinutes,
    techConfigFile,
    solutionFile
  );
}
