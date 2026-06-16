/*
 * Copyright (c) 2024 Suhail Basalama
 *
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 */
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "streamhls/Transforms/Passes.h"
#include "streamhls/Support/Utils.h"
#include "streamhls/Support/DFG.h"
#include <fstream>
#include <string>
using namespace mlir;
using namespace streamhls;
using namespace dataflow;

#define DEBUG_TYPE "streamhls-create-tasks"


namespace {
struct CreateTasks : public CreateTasksBase<CreateTasks> {
  CreateTasks() = default;
  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "CreateTasks\n");
    auto func = getOperation();
    auto context = func.getContext();
    auto dispatchOp = dispatchBlock(&func.front());
    OpBuilder builder(context);
    auto loc = builder.getUnknownLoc();
    Block* block = &dispatchOp.getBody().front();

    DFG graph(*block);
    if (!graph.init(false)) {
      LLVM_DEBUG(llvm::dbgs() << "DFG init failed\n");
      return;
    }
  //   // for(auto nodePair : graph.nodes){
  //   for(auto id : graph.topologicalSort()){
  //     DFG::Node* node = graph.getNode(id);
  //     auto loc = builder.getUnknownLoc();
  //     builder.setInsertionPoint(node->op);
  //     auto task = builder.create<TaskOp>(loc, ValueRange({}));
  //     auto taskBlock = builder.createBlock(&task.getBody());
  //     auto yield = builder.create<YieldOp>(loc, ValueRange({}));
  //     builder.setInsertionPointToStart(taskBlock);
  //     for(auto op : node->allocOps){
  //       op->moveBefore(yield);
  //     }
  //     // reverse order of forOps
  //     for(auto op : llvm::reverse(node->forOps)){
  //       op->moveBefore(yield);
  //     }
  //     // for(auto op : node.forOps){
  //     //   op->moveBefore(yield);
  //     // }
  //   }
  //   // llvm::dbgs() << "After creating tasks\n";
  //   for (auto &op : llvm::make_early_inc_range(*block)) {
  //     // op.dump();
  //     if (isa<StreamOp, ArrayOfStreamsOp, memref::AllocOp>(op)) {
  //       // Memory allocs are moved to the begining and skipped.
  //       op.moveBefore(block, block->begin());
  //     }
  //   }
  //   // llvm::dbgs() << "After moving ops\n";
  // }
    for(auto nodePair : graph.nodes){
      auto& node = nodePair.second;
      auto loc = builder.getUnknownLoc();
      builder.setInsertionPoint(node.op);
      auto task = builder.create<TaskOp>(loc, ValueRange({}));
      // Stamp the DFG node id so the emitted function name matches the id used
      // in the design-space/solution JSON files.
      task->setAttr("dataflow.node_id", builder.getI64IntegerAttr(node.id));
      auto taskBlock = builder.createBlock(&task.getBody());
      auto yield = builder.create<YieldOp>(loc, ValueRange({}));
      builder.setInsertionPointToStart(taskBlock);
      for(auto op : node.allocOps){
        op->moveBefore(yield);
      }
      // reverse order of forOps
      for(auto op : llvm::reverse(node.forOps)){
        op->moveBefore(yield);
      }
      // for(auto op : node.forOps){
      //   op->moveBefore(yield);
      // }
    }
    // llvm::dbgs() << "After creating tasks\n";
    for (auto &op : llvm::make_early_inc_range(*block)) {
      // op.dump();
      if (isa<StreamOp, ArrayOfStreamsOp, memref::AllocOp>(op)) {
        // Memory allocs are moved to the begining and skipped.
        op.moveBefore(block, block->begin());
      }
    }
    // llvm::dbgs() << "After moving ops\n";
  }
};
} // namespace

std::unique_ptr<Pass> streamhls::createCreateTasksPass() {
  return std::make_unique<CreateTasks>();
}