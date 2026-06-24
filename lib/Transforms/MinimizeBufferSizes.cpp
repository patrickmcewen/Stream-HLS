/*
 * Copyright (c) 2024 Suhail Basalama
 *
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 * parts of this pass are adobted from ScaleHLS
 */


#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/IR/IntegerSet.h"


#include "streamhls/Transforms/Passes.h"
#include "streamhls/Support/Utils.h"
#include "streamhls/Support/AffineMemAccess.h"

using namespace mlir;
using namespace streamhls;
using namespace dataflow;

#define DEBUG_TYPE "streamhls-minimize-buffer-sizes"


namespace {
  // SCALEHLS code
  static LogicalResult collapseMemref(
    Value memref, 
    SmallVectorImpl<int64_t>& newShape,
    SmallVectorImpl<unsigned>& remainDims
  ) {
    auto type = memref.getType().dyn_cast<MemRefType>();
    // if (!type)
    //   return failure();

    // // TODO: Support non-identity buffers.
    // if (!type.getLayout().getAffineMap().isIdentity() ||
    //     !llvm::any_of(type.getShape(),
    //                   [](int64_t dimSize) { return dimSize == 1; }))
    //   return failure();

    // // TODO: Support non-affine read/write interfaces.
    // if (llvm::any_of(memref.getUsers(), [](Operation *op) {
    //       return !isa<AffineReadOpInterface, AffineWriteOpInterface>(op);
    //     }))
    //   return failure();

    // // Construct new shape.
    // SmallVector<int64_t> newShape;
    // SmallVector<unsigned> remainDims;
    // for (auto dimSize : llvm::enumerate(type.getShape())){
    //   if (dimSize.value() != 1) {
    //     newShape.push_back(dimSize.value());
    //     remainDims.push_back(dimSize.index());
    //   }
    // }
    // type.dump();
    // Set the buffer to a new type.
    auto newType = MemRefType::get(newShape, type.getElementType(), AffineMap(),
                                  type.getMemorySpace());
    // newType.dump();
    memref.setType(newType);

    // Update buffer users.
    for (auto user : memref.getUsers()) {
      AffineMap map;
      if (auto read = dyn_cast<affine::AffineReadOpInterface>(user))
        map = read.getAffineMap();
      else if (auto write = dyn_cast<affine::AffineWriteOpInterface>(user))
        map = write.getAffineMap();
      // map.dump();
      SmallVector<AffineExpr> newResults;
      for (auto dim : remainDims)
        newResults.push_back(map.getResult(dim));
      auto newMap = AffineMap::get(map.getNumDims(), map.getNumSymbols(),
                                  newResults, map.getContext());
      // newMap.dump();
      
      user->setAttr("map", AffineMapAttr::get(newMap));
    }
    return success();
  }
  struct MinimizeBufferPattern : public OpRewritePattern<memref::AllocOp> {
    using OpRewritePattern<memref::AllocOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(
      memref::AllocOp allocOp,
      PatternRewriter &rewriter
    ) const override {
      auto memRef = allocOp.getResult();
      auto context = rewriter.getContext();
      SmallVector<Operation*, 4> loadAndStoreOps;
      SmallVector<Operation::operand_range, 4> mapOperands;
      LLVM_DEBUG(
        llvm::dbgs() << memRef << "\n";
      );
      AffineForOp forOp;
      for(auto user : memRef.getUsers()){
        if(isa<AffineLoadOp, AffineStoreOp>(user)){
          forOp = dyn_cast<AffineForOp>(user->getParentOfType<AffineForOp>());
          loadAndStoreOps.push_back(user);
          if(auto loadOp = dyn_cast<AffineLoadOp>(user)){
            mapOperands.push_back(loadOp.getMapOperands());
            if(llvm::any_of(loadOp.getAffineMap().getResults(), [](AffineExpr expr){
              return expr.getKind() == AffineExprKind::FloorDiv || expr.getKind() == AffineExprKind::Mod;
            })){
              return failure();
            }
          }
          if(auto storeOp = dyn_cast<AffineStoreOp>(user)){
            mapOperands.push_back(storeOp.getMapOperands());
            if(llvm::any_of(storeOp.getAffineMap().getResults(), [](AffineExpr expr){
              return expr.getKind() == AffineExprKind::FloorDiv || expr.getKind() == AffineExprKind::Mod;
            })){
              return failure();
            }
          }
        }
      }
      for(auto op : loadAndStoreOps){
        if(op->getParentOfType<AffineForOp>() != forOp){
          return failure();
        }
      }
      for(auto operands1 : mapOperands){
        for(auto operands2 : mapOperands){
          if(operands1.size() != operands2.size()){
            return failure();
          }
          for(unsigned i = 0; i < operands1.size(); i++){
            if(operands1[i] != operands2[i]){
              return failure();
            }
          }
        }
      }

      // AffineLoopBand band;
      // getLoopBandFromInnermost(forOp, band);
      // SmallVector<Value, 4> loopIVs;
      // for(auto loop : band){
      //   loopIVs.push_back(loop.getInductionVar());
      // }
      // SmallVector<Value, 4> bufferIndices;
      // for(auto val : mapOperands[0]){
      //   bufferIndices.push_back(val);
      // }
      // Value irrelevantIV;
      // for(auto iv : loopIVs){
      //   bool found = false;
      //   for(auto val : bufferIndices){
      //     if(val == iv){
      //       found = true;
      //       break;
      //     }
      //   }
      //   if(!found){
      //     irrelevantIV = iv;
      //     break;
      //   }
      // }
      // if(!irrelevantIV){
      //   return failure();
      // }
      // SmallVector<Value, 4> dimValsToKeep;
      // unsigned endIdx = 0;
      // for(auto iv : loopIVs){
      //   if(iv == irrelevantIV){
      //     break;
      //   }
      //   endIdx++;
      // }
      // for(unsigned i = endIdx; i < loopIVs.size(); i++){
      //   dimValsToKeep.push_back(loopIVs[i]);
      // }
      // SmallVector<unsigned, 4> dimIdxsToKeep;
      // for(auto val : dimValsToKeep){
      //   for(unsigned i = 0; i < bufferIndices.size(); i++){
      //     if(bufferIndices[i] == val){
      //       dimIdxsToKeep.push_back(i);
      //     }
      //   }
      // }
      // auto oldShape = allocOp.getType().getShape();

      // if(dimIdxsToKeep.size() == oldShape.size()){
      //   return failure();
      // }
      // // LLVM_DEBUG(
      // //   llvm::dbgs() << "Irrelevant IV: " << irrelevantIV << "\n";
      // // );
      // // LLVM_DEBUG(
      // //   llvm::dbgs() << "Found " << loadAndStoreOps.size() << " users\n";
      // // );
      // SmallVector<int64_t, 4> newShape; 
      // SmallVector<unsigned, 4> remainDims;
      // for(auto idx : dimIdxsToKeep){
      //   LLVM_DEBUG(
      //     llvm::dbgs() << "Removing dimension " << idx << "\n";
      //   );
      //   newShape.push_back(oldShape[idx]);
      //   remainDims.push_back(idx);
      // }
      // for(auto shp : oldShape){
      //   LLVM_DEBUG(
      //     llvm::dbgs() << shp << " ";
      //   );
      // }
      // LLVM_DEBUG(
      //   llvm::dbgs() << "\n";
      // );
      // for(auto shp : newShape){
      //   LLVM_DEBUG(
      //     llvm::dbgs() << shp << " ";
      //   );
      // }
      // if(oldShape.size() == newShape.size()){
      //   return failure();
      // }
      // return collapseMemref(memRef, newShape, remainDims);
      return success();
    }
  };
}

namespace {
  struct MinimizeLocalBuffersNoPartition : public OpRewritePattern<memref::AllocOp> {
    using OpRewritePattern<memref::AllocOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(
      memref::AllocOp allocOp,
      PatternRewriter &rewriter
    ) const override {
      AffineForOp forOp;
      for(auto user : allocOp.getResult().getUsers()){
        if(auto op = user->getParentOfType<AffineForOp>()){
          forOp = op;
          break;
        }
      }
      if(!forOp){
        return failure();
      }
      AffineLoopBand band;
      getLoopBandFromInnermost(forOp, band);
      if(band.size() != 6){
        return failure();
      }
      if(!isPerfectlyNested(band)){
        return failure();
      }
      AffineLoopBand outerLoops;
      for(unsigned i = 0; i < band.size()/2; i++){
        outerLoops.push_back(band[i]);
      }
      AffineLoopBand innerLoops;
      for(unsigned i = band.size()/2; i < band.size(); i++){
        innerLoops.push_back(band[i]);
      }
      // SmallVector<std::pair<unsigned, bool>> isLoopUsed;
      SmallVector<AffineStoreOp> storeOps;
      SmallVector<AffineLoadOp> loadOps;
      for(auto user : allocOp.getResult().getUsers()){
        if(auto loadOp = dyn_cast<AffineLoadOp>(user)){
          loadOps.push_back(loadOp);
        }else if(auto storeOp = dyn_cast<AffineStoreOp>(user)){
          storeOps.push_back(storeOp);
        }
      }
      auto reuseIdx = 0;
      for(int i = (int)innerLoops.size() - 1; i >= 0; i--){  // signed: size()-1 underflows unsigned
        auto forOp = innerLoops[i];
        auto iv = forOp.getInductionVar();
        bool isUsed = false;
        for(auto storeOp : storeOps){
          for(auto operandPair : llvm::enumerate(storeOp.getMapOperands())){
            auto operand = operandPair.value();
            auto operandIdx = operandPair.index();
            if(operand == iv){
              isUsed = true;
              break;
            }
          }
        }
        for(auto loadOp : loadOps){
          for(auto operandPair : llvm::enumerate(loadOp.getMapOperands())){
            auto operand = operandPair.value();
            auto operandIdx = operandPair.index();
            if(operand == iv){
              isUsed = true;
              break;
            }
          }
        }
        if(!isUsed){
          reuseIdx = i;
          break;
        }
      }
      // allocOp.dump();
      // llvm::dbgs() << "Reuse Index: " << reuseIdx << "\n";
      auto oldShape = allocOp.getMemref().getType().cast<MemRefType>().getShape();
      DenseMap<unsigned, unsigned> newShapeMap; // dim, size
      for(int i = 0; i < reuseIdx; i++){
        auto outerForOp = outerLoops[i];
        auto outerIV = outerForOp.getInductionVar();
        auto innerForOp = innerLoops[i];
        auto innerIV = innerForOp.getInductionVar();
        auto innerUB = innerForOp.getConstantUpperBound();
        // update affine maps
        for(auto storeOp : storeOps){
          int mapOperandIdx = -1;
          for(auto operandPair : llvm::enumerate(storeOp.getOperands())){
            auto operand = operandPair.value();
            auto operandIdx = operandPair.index();
            if(operand == outerIV){
              mapOperandIdx = operandIdx;
              break;
            }
          }
          // set map operand to 0
          auto constOp = rewriter.create<arith::ConstantOp>(
            allocOp.getLoc(),
            rewriter.getIndexType(),
            rewriter.getIndexAttr(0)
          );
          if(mapOperandIdx != -1){
            auto affineMap = storeOp.getAffineMap();
            auto dimExpr = rewriter.getAffineDimExpr(mapOperandIdx - 2);
            int targetDim = -1;
            for(auto resultPair : llvm::enumerate(affineMap.getResults())){
              auto result = resultPair.value();
              auto resultPos = resultPair.index();
              result.walk([&](AffineExpr expr){
                if(expr == dimExpr){
                  targetDim = resultPos;
                  return;
                }
              });
            }
            newShapeMap[targetDim] = innerUB;
            storeOp.setOperand(mapOperandIdx, constOp.getResult());
          }
        }
        for(auto loadOp : loadOps){
          int mapOperandIdx = -1;
          for(auto operandPair : llvm::enumerate(loadOp.getOperands())){
            auto operand = operandPair.value();
            auto operandIdx = operandPair.index();
            if(operand == outerIV){
              mapOperandIdx = operandIdx;
              break;
            }
          }
          // set map operand to 0
          auto constOp = rewriter.create<arith::ConstantOp>(
            allocOp.getLoc(),
            rewriter.getIndexType(),
            rewriter.getIndexAttr(0)
          );
          // loadOp.getAffineMap().dump();
          if(mapOperandIdx != -1){
            auto affineMap = loadOp.getAffineMap();
            auto dimExpr = rewriter.getAffineDimExpr(mapOperandIdx - 1);
            int targetDim = -1;
            for(auto resultPair : llvm::enumerate(affineMap.getResults())){
              auto result = resultPair.value();
              auto resultPos = resultPair.index();
              result.walk([&](AffineExpr expr){
                if(expr == dimExpr){
                  targetDim = resultPos;
                  return;
                }
              });
            }
            newShapeMap[targetDim] = innerUB;
            loadOp.setOperand(mapOperandIdx, constOp.getResult());
          }
        }
      }
      SmallVector<int64_t> newShape;
      for(auto dimSize : llvm::enumerate(oldShape)){
        auto dim = dimSize.index();
        auto size = dimSize.value();
        if(newShapeMap.count(dim)){
          newShape.push_back(newShapeMap[dim]);
        }else{
          newShape.push_back(size);
        }
      }
      // llvm::dbgs() << "old shape: ";
      // for(auto size : oldShape){
      //   llvm::dbgs() << size << " ";
      // }
      // llvm::dbgs() << "\n";
      // llvm::dbgs() << "new shape: ";
      // for(auto size : newShape){
      //   llvm::dbgs() << size << " ";
      // }
      // llvm::dbgs() << "\n";
      // update memref type
      auto oldMemref = allocOp.getMemref();
      auto oldMemSpace = oldMemref.getType().cast<MemRefType>().getMemorySpace();
      auto oldLayout = oldMemref.getType().cast<MemRefType>().getLayout();
      auto newType = MemRefType::get(newShape, oldMemref.getType().cast<MemRefType>().getElementType(), oldLayout, oldMemSpace);
      oldMemref.setType(newType);
      return success();
    }
  };
}


namespace {
struct MinimalBufferSizes : public MinimalBufferSizesBase<MinimalBufferSizes> {

  MinimalBufferSizes() = default;

  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add<MinimizeLocalBuffersNoPartition>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass> streamhls::createMinimalBufferSizesPass() {
  return std::make_unique<MinimalBufferSizes>();
}