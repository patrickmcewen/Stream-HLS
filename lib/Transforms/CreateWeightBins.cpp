/*
 * Copyright (c) 2024 Suhail Basalama
 *
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 */
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "streamhls/Transforms/Passes.h"

#include "streamhls/Support/Utils.h"
#include "streamhls/Support/AffineMemAccess.h"

#include "mlir/IR/IntegerSet.h"
using namespace mlir;
using namespace streamhls;
using namespace dataflow;

#define DEBUG_TYPE "streamhls-create-weight-bins"

namespace {
  struct ConvertWeightsToBins : public OpRewritePattern<arith::ConstantOp> {
    ConvertWeightsToBins(MLIRContext *context, bool keepWeights = true) 
      : OpRewritePattern(context), keepWeights(keepWeights) {}

    using OpRewritePattern<arith::ConstantOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(
      arith::ConstantOp op,
      PatternRewriter &rewriter
    ) const override {
      LLVM_DEBUG(llvm::dbgs() << "ConvertWeightsToBins\n");
      LLVM_DEBUG(
        llvm::dbgs() << "Running on constant op: ";
        op.print(llvm::dbgs());
        llvm::dbgs() << "\n";
      );
      auto tensor = op.getValueAttr().dyn_cast_or_null<DenseElementsAttr>();
      
      if(!tensor) return failure();
      // if splat, replace with a single value
      if(tensor.isSplat()) {
        // // get element type
        // auto elementType = tensor.getType().getElementType();
        // if(auto float32Type = elementType.dyn_cast<Float32Type>()){
        //   auto value = tensor.getSplatValue<FloatAttr>().getValueAsDouble();
        //   // llvm::dbgs() << "value: " << value << "\n";
        //   auto attr = rewriter.getF32FloatAttr(value);
        //   // attr.dump();
        //   auto constOp = rewriter.create<arith::ConstantOp>(op.getLoc(), attr);
        //   // constOp.dump();
        //   rewriter.replaceOp(op, constOp.getResult());
        // } else if(auto float64Type = elementType.dyn_cast<Float64Type>()){
        //   auto value = tensor.getSplatValue<FloatAttr>().getValueAsDouble();
        //   auto attr = rewriter.getF64FloatAttr(value);
        //   auto constOp = rewriter.create<arith::ConstantOp>(op.getLoc(), attr);
        //   rewriter.replaceOp(op, constOp.getResult());
        // } else if(auto intType = elementType.dyn_cast<IntegerType>()){
        //   auto value = tensor.getSplatValue<IntegerAttr>().getInt();
        //   auto attr = rewriter.getIntegerAttr(intType, value);
        //   auto constOp = rewriter.create<arith::ConstantOp>(op.getLoc(), attr);
        //   rewriter.replaceOp(op, constOp.getResult());
        // } else {
        //   llvm::errs() << "Unsupported element type\n";
        //   return failure();
        // }
        return failure();
      }
      // TODO: create a heuristic for when to keep weights
      // if(tensor.size() < 1024) return failure();
      auto func = op->getParentOfType<func::FuncOp>();
      auto loc = rewriter.getUnknownLoc();
      auto argNum = func.getNumArguments();
      if(keepWeights){
        // AsmResourceBlob blob = UnmanagedAsmResourceBlob::allocateInferAlign(tensor.getRawData());
        StringRef nameForBlob = "weight";
        ShapedType type = tensor.getType().cast<ShapedType>();
        // cast tensor to memref
        auto memrefType = MemRefType::get(type.getShape(), type.getElementType());
        // auto attr = DenseElementsAttr::getFromRawBuffer(memrefType, tensor.getRawData());
        auto attr = DenseResourceElementsAttr::get(memrefType, nameForBlob, UnmanagedAsmResourceBlob::allocateInferAlign(tensor.getRawData()));
        op.setValueAttr(attr);
        func.insertArgument(argNum, op.getType(), {}, loc);
        func.setArgAttr(argNum, "builtin.dense_resource", attr);
      } else {
        func.insertArgument(argNum, op.getType(), {}, loc);
      }
      rewriter.replaceAllUsesWith(op, func.getArgument(argNum));
      return success();
    }
    private:
      bool keepWeights;
  };
}

// This pass takes creates the weight bins for the host function
namespace {
struct CreateWeightBins : public CreateWeightBinsBase<CreateWeightBins> {
  CreateWeightBins() = default;
  CreateWeightBins(bool argKeepWeights, std::string argTopFuncName) {
    keepWeights = argKeepWeights;
    topFuncName = argTopFuncName;
  }
  void runOnOperation() override {
    auto module = getOperation();
    auto context = module.getContext();
    auto func = module.lookupSymbol<func::FuncOp>(topFuncName);
    mlir::RewritePatternSet patterns(context);
    patterns.add<ConvertWeightsToBins>(context, keepWeights);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> streamhls::createCreateWeightBinsPass(bool keepWeights, std::string topFuncName) {
  return std::make_unique<CreateWeightBins>(
    keepWeights,
    topFuncName
  );
}