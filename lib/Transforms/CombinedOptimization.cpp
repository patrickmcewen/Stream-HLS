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
    graph.createCombinedOptimizationPerformanceModel(reportFile, DSPs, tilingLimit, timeLimitMinutes);

    graph.callCombinedOptimizationSolver(reportFile);

    graph.writeSolutionJSON(reportFile);

    graph.createCombinedOptimizationPythonModel(reportFile);
    
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
  std::string techConfigFile
) {
  return std::make_unique<CombinedOptimization>(
    reportFile,
    parallelizeNodes,
    DSPs,
    tilingLimit,
    timeLimitMinutes,
    techConfigFile
  );
}

namespace {
struct EmitTransformSpace : public EmitTransformSpaceBase<EmitTransformSpace> {
  EmitTransformSpace() = default;
  EmitTransformSpace(std::string argReportFile, uint argTilingLimit, std::string argTechConfigFile) {
    reportFile = argReportFile;
    tilingLimit = argTilingLimit;
    techConfigFile = argTechConfigFile;
  }
  void runOnOperation() override {
    if (!techConfigFile.empty())
      initTechConfig(techConfigFile);
    func::FuncOp func = getOperation();
    auto block = &func.front();
    DFG graph(*block);
    if (!graph.init(false)) {
      LLVM_DEBUG(llvm::dbgs() << "DFG init failed\n");
      return;
    }
    graph.emitTransformSpaceJSON(reportFile, tilingLimit);
  }
};
}

std::unique_ptr<Pass> streamhls::createEmitTransformSpacePass(
  std::string reportFile,
  uint tilingLimit,
  std::string techConfigFile
) {
  return std::make_unique<EmitTransformSpace>(reportFile, tilingLimit, techConfigFile);
}

namespace {
struct ApplyTransformSolution : public ApplyTransformSolutionBase<ApplyTransformSolution> {
  ApplyTransformSolution() = default;
  ApplyTransformSolution(std::string argSolutionFile, std::string argTechConfigFile) {
    solutionFile = argSolutionFile;
    techConfigFile = argTechConfigFile;
  }
  void runOnOperation() override {
    if (!techConfigFile.empty())
      initTechConfig(techConfigFile);
    func::FuncOp func = getOperation();
    auto block = &func.front();
    DFG graph(*block);
    if (!graph.init(false)) {
      LLVM_DEBUG(llvm::dbgs() << "DFG init failed\n");
      return;
    }
    graph.applyTransformSolutionFromJSON(solutionFile);
  }
};
}

std::unique_ptr<Pass> streamhls::createApplyTransformSolutionPass(
  std::string solutionFile,
  std::string techConfigFile
) {
  return std::make_unique<ApplyTransformSolution>(solutionFile, techConfigFile);
}