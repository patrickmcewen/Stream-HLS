//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//
/*
 * Modified by Suhail Basalama in 2024.
 *
 * This software is also released under the MIT License:
 * https://opensource.org/licenses/MIT
 */
#include "mlir/Analysis/Liveness.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "streamhls/Transforms/Passes.h"
#include "streamhls/Support/Utils.h"
#include "streamhls/Support/DataflowGraph.h"

using namespace mlir;
using namespace streamhls;
using namespace dataflow;

#define DEBUG_TYPE "streamhls-create-dataflow-from-affine"

namespace {
struct TaskPartition : public OpRewritePattern<DispatchOp> {
  using OpRewritePattern<DispatchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DispatchOp dispatch,
                                PatternRewriter &rewriter) const override {
    if (llvm::any_of(dispatch.getOps(), [](Operation &op) {
          return isa<bufferization::BufferizationDialect, tosa::TosaDialect,
                     tensor::TensorDialect, linalg::LinalgDialect>(
                     op.getDialect()) ||
                 isa<func::CallOp, DispatchOp, TaskOp>(op);
        }))
      return failure();
    auto &block = dispatch.getRegion().front();

    // Fuse operations into dataflow tasks. TODO: We need more case study to
    // figure out any other operations need to be separately handled. For
    // example, how to handle AffineIfOp?
    SmallVector<Operation *, 4> opsToFuse;
    unsigned taskIdx = 0;
    for (auto &op : llvm::make_early_inc_range(block)) {
      if (isa<StreamOp, ArrayOfStreamsOp>(op)) {
        // Memory allocs are moved to the begining and skipped.
        op.moveBefore(&block, block.begin());
      } else if (isa<AffineForOp>(op)){
        // op.dump();
        SmallVector<Value, 4> inputs;
        op.walk([&](Operation *op) {
          if(auto storeOp = dyn_cast<AffineStoreOp>(op)){
            auto memref = storeOp.getMemRef();
            if(!isa<BlockArgument>(memref)){
              auto defOp = memref.getDefiningOp();
              if(std::find(opsToFuse.begin(), opsToFuse.end(), defOp) == opsToFuse.end()){
                opsToFuse.push_back(defOp);
              }
            }
          }
          if(auto loadOp = dyn_cast<AffineLoadOp>(op)){
            auto memref = loadOp.getMemRef();
            if(!isa<BlockArgument>(memref)){
              auto defOp = memref.getDefiningOp();
              if(std::find(opsToFuse.begin(), opsToFuse.end(), defOp) == opsToFuse.end()){
                opsToFuse.push_back(defOp);
              }
            }
          }
        });
        // for(auto op : opsToFuse){
        //   op->dump();
        // }
        // size of opsToFuse
        // LLVM_DEBUG(llvm::dbgs() << "size of opsToFuse: " << opsToFuse.size() << "\n");
        opsToFuse.push_back(&op);

        // fuseOpsIntoTask(opsToFuse, rewriter);
        auto loc = rewriter.getUnknownLoc();
        rewriter.setInsertionPoint(&op);
        auto task = rewriter.create<TaskOp>(loc, ValueRange({}));
        auto taskBlock = rewriter.createBlock(&task.getBody());
        auto yield = rewriter.create<YieldOp>(loc, ValueRange({}));
        rewriter.setInsertionPointToStart(taskBlock);
        for(auto op : opsToFuse){
          op->moveBefore(yield);
        }
        opsToFuse.clear();
        taskIdx++;
      } 
      // if (isa<StreamOp>(op)) {
      //   // Memory allocs are moved to the begining and skipped.
      //   op.moveBefore(&block, block.begin());

      // } else if (isa<AffineForOp, scf::ForOp>(op)) {
      //   // We always take loop as root operation and fuse all the collected
      //   // operations so far.
      //   opsToFuse.push_back(&op);
      //   fuseOpsIntoTask(opsToFuse, rewriter);
      //   opsToFuse.clear();
      //   taskIdx++;

      // } else if (&op == block.getTerminator()) {
      //   // If the block will only generate one task, stop it.
      //   if (opsToFuse.empty() || taskIdx == 0)
      //     continue;
      //   fuseOpsIntoTask(opsToFuse, rewriter);
      //   opsToFuse.clear();
      //   taskIdx++;

      // } else {
      //   // Otherwise, we push back the current operation to the list.
      //   opsToFuse.push_back(&op);
      // }
    }
    return success();
  }
};
} // namespace

namespace {
static bool isWritten(OpOperand &use) {
  // For ScheduleOp, we don't rely on memory effect interface. Instead, we delve
  // into its region to figure out the effect. However, for NodeOp, we don't
  // need this recursive approach any more.
  if (auto node = dyn_cast<NodeOp>(use.getOwner()))
    return node.getOperandKind(use) == OperandKind::OUTPUT;
  // else if (auto schedule = dyn_cast<ScheduleOp>(use.getOwner()))
  //   return llvm::any_of(
  //       schedule.getBody().getArgument(use.getOperandNumber()).getUses(),
  //       [](OpOperand &argUse) { return isWritten(argUse); });
  else if (auto view = dyn_cast<ViewLikeOpInterface>(use.getOwner()))
    return llvm::any_of(view->getUses(),
                        [](OpOperand &viewUse) { return isWritten(viewUse); });
  return hasEffect<MemoryEffects::Write>(use.getOwner(), use.get()) ||
         isa<StreamWriteOp, ArrayOfStreamsWriteOp>(use.getOwner());
}

struct TasksToFuncs : public OpRewritePattern<TaskOp> {
  using OpRewritePattern<TaskOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TaskOp task,
    PatternRewriter &rewriter) const override {
    if (task.getNumResults())
      return task.emitOpError("should not yield any results");

    auto isInTask = [&](OpOperand &use) {
      return task->isAncestor(use.getOwner());
    };

    SmallVector<Value, 8> inputs;
    SmallVector<Location, 8> inputLocs;
    SmallVector<Value, 8> outputs;
    SmallVector<Location, 8> outputLocs;
    SmallVector<Value, 8> params;
    SmallVector<Location, 8> paramLocs;

    auto liveins = Liveness(task).getLiveIn(&task.getBody().front());
    for (auto livein : liveins) {
      if (task.getBody().isAncestor(livein.getParentRegion()))
        continue;

      if (livein.getType().isa<MemRefType, StreamType, RankedTensorType>()) {
        LLVM_DEBUG(llvm::dbgs() << "livein: " << livein << "\n");
        auto uses = llvm::make_filter_range(livein.getUses(), isInTask);
        if (llvm::any_of(uses, [](OpOperand &use) { return isWritten(use); })) {
          outputs.push_back(livein);
          outputLocs.push_back(livein.getLoc());
        } else {
          inputs.push_back(livein);
          inputLocs.push_back(livein.getLoc());
        }
      } else {
        params.push_back(livein);
        paramLocs.push_back(livein.getLoc());
      }
    }

    rewriter.setInsertionPoint(task);
    auto node = rewriter.create<NodeOp>(rewriter.getUnknownLoc(), inputs,
                                        outputs, params);
    // Carry the DFG node id stamped by CreateTasks onto the node so the emitted
    // function name aligns with the id used in the JSON files.
    if (auto nodeId = task->getAttr("dataflow.node_id"))
      node->setAttr("dataflow.node_id", nodeId);
    auto nodeBlock = rewriter.createBlock(&node.getBody());

    auto inputArgs = nodeBlock->addArguments(ValueRange(inputs), inputLocs);
    for (auto t : llvm::zip(inputs, inputArgs))
      std::get<0>(t).replaceUsesWithIf(std::get<1>(t), isInTask);

    auto outputArgs =
        node.getBody().addArguments(ValueRange(outputs), outputLocs);
    for (auto t : llvm::zip(outputs, outputArgs))
      std::get<0>(t).replaceUsesWithIf(std::get<1>(t), isInTask);

    auto paramArgs = nodeBlock->addArguments(ValueRange(params), paramLocs);
    for (auto t : llvm::zip(params, paramArgs))
      std::get<0>(t).replaceUsesWithIf(std::get<1>(t), isInTask);

    auto &nodeOps = nodeBlock->getOperations();
    auto &taskOps = task.getBody().front().getOperations();
    nodeOps.splice(nodeOps.begin(), taskOps, taskOps.begin(),
                   std::prev(taskOps.end()));

    rewriter.eraseOp(task);
    return success();
  }
};
} // namespace

namespace {
struct ConvertNodeToFunc : public OpRewritePattern<NodeOp> {
  ConvertNodeToFunc(MLIRContext *context, StringRef prefix, unsigned &nodeIdx)
      : OpRewritePattern<NodeOp>(context), prefix(prefix), nodeIdx(nodeIdx) {}

  LogicalResult matchAndRewrite(NodeOp node,
                                PatternRewriter &rewriter) const override {
    // Create a new sub-function. Name it using the DFG node id stamped by
    // CreateTasks so the function name matches the id in the JSON files. Fall
    // back to a running counter only if the id is missing.
    auto nodeIdAttr = node->getAttrOfType<IntegerAttr>("dataflow.node_id");
    unsigned funcId = nodeIdAttr ? nodeIdAttr.getInt() : nodeIdx++;
    rewriter.setInsertionPoint(node->getParentOfType<func::FuncOp>());
    auto subFunc = rewriter.create<func::FuncOp>(
        node.getLoc(), prefix.str() + "node" + std::to_string(funcId),
        rewriter.getFunctionType(node.getOperandTypes(), TypeRange()));


    // Inline the contents of the dataflow node.
    rewriter.inlineRegionBefore(node.getBodyRegion(), subFunc.getBody(),
                                subFunc.end());
    rewriter.setInsertionPointToEnd(&subFunc.front());
    rewriter.create<func::ReturnOp>(rewriter.getUnknownLoc());

    // Replace original with a function call.
    rewriter.setInsertionPoint(node);
    rewriter.replaceOpWithNewOp<func::CallOp>(node, subFunc,
                                              node.getOperands());
    return success();
  }

private:
  StringRef prefix;
  unsigned &nodeIdx;
};
} // namespace


namespace {
struct RemoveDispatch : public OpRewritePattern<DispatchOp> {
  using OpRewritePattern<DispatchOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(DispatchOp dispatch,
                                PatternRewriter &rewriter) const override {
    auto &block = dispatch.getRegion().front();
    auto topFunc = dispatch->getParentOfType<func::FuncOp>();
    auto topFuncBlock = &topFunc.front();
    auto terminator = topFuncBlock->getTerminator();
    block.walk([&](Operation* op) {
      if(!isa<YieldOp>(op))
        op->moveBefore(terminator);
    });
    rewriter.eraseOp(dispatch);
    return success();
  }
};
} // namespace

namespace {
struct CreateDataflowFromAffine
    : public CreateDataflowFromAffineBase<CreateDataflowFromAffine> {
  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();

    dispatchBlock(&func.front());
    AffineLoopBands targetBands;
    getLoopBands(func.front(), targetBands, /*allowHavingChilds=*/true);
    // for (auto &band : llvm::reverse(targetBands))
    //   dispatchBlock(band.back().getBody());

    mlir::RewritePatternSet patterns(context);
    // patterns.add<TaskPartition>(context);
    // (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
    // patterns.clear();
    patterns.add<TasksToFuncs>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
    patterns.clear();
    unsigned nodeIdx = 0;
    patterns.add<ConvertNodeToFunc>(context, "", nodeIdx);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
    patterns.clear();
    patterns.add<RemoveDispatch>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass> streamhls::createCreateDataflowFromAffinePass() {
  return std::make_unique<CreateDataflowFromAffine>();
}
