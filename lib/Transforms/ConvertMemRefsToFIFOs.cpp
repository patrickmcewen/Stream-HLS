/*
 * Copyright (c) 2024 Suhail Basalama
 *
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 */
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "streamhls/Transforms/Passes.h"
// #include "streamhls/Transforms/Utils.h"

// #include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
// #include "mlir/Analysis/Presburger/PresburgerSpace.h"

#include "streamhls/Support/Utils.h"
#include "streamhls/Support/AffineMemAccess.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/IR/AffineMap.h"

#include "mlir/IR/IntegerSet.h"


using namespace mlir;
using namespace streamhls;
using namespace dataflow;
using namespace presburger;

#define DEBUG_TYPE "streamhls-convert-memrefs-to-fifos"

namespace{
  static LogicalResult storeLoadRewriter(
    PatternRewriter &rewriter,
    memref::AllocOp allocOp,
    AffineStoreOp storeOp,
    AffineLoadOp loadOp,
    AccessInfo storeInfo,
    AccessInfo loadInfo,
    bool isStoreLoadStoreLoad = false
  ){
    auto memrefType = allocOp.getMemref().getType().getElementType();
    auto memrefShape = allocOp.getMemref().getType().cast<MemRefType>().getShape();
    auto memrefSize = 1;
    for(auto dim : memrefShape){
      memrefSize *= dim;
    }
    auto loc = rewriter.getUnknownLoc();
    // create fifo
    rewriter.setInsertionPoint(allocOp);
    auto streamOp = rewriter.create<dataflow::StreamOp>(
      loc,
      dataflow::StreamType::get(rewriter.getContext(), memrefType, memrefSize),
      memrefSize
    );

    rewriter.setInsertionPoint(storeOp);
    if(!storeInfo.ifExprs.empty() && !isa<AffineIfOp>(storeOp->getParentOp())){
      // create write fifo operation
      auto ifCondition = IntegerSet::get(storeInfo.ifExprs.size(), 0, storeInfo.ifExprs, storeInfo.ifEqFlags);
      auto ifOp = rewriter.create<AffineIfOp>(
            loc, ifCondition, storeInfo.ifOperands, false);
      rewriter.setInsertionPointToStart(ifOp.getThenBlock());
    }
    // create write fifo operation
    rewriter.create<dataflow::StreamWriteOp>(
      loc, streamOp.getChannel(), storeOp.getValueToStore());

    // if(!storeInfo.ifExprs.empty()){
    //   // create affine load operation
    //   rewriter.create<AffineLoadOp>(
    //     loc, loadOp.getMemRef(), loadOp.getAffineMap(), loadOp.getMapOperands()
    //   );
    // }
    // create read fifo
    rewriter.setInsertionPoint(loadOp);
    if (!loadInfo.ifExprs.empty() && !isa<AffineIfOp>(loadOp->getParentOp())) {
      auto ifCondition = IntegerSet::get(loadInfo.ifExprs.size(), 0, loadInfo.ifExprs, loadInfo.ifEqFlags);
      auto ifOp = rewriter.create<AffineIfOp>(
            loc, ifCondition, loadInfo.ifOperands, false);
      rewriter.setInsertionPointToStart(ifOp.getThenBlock());
    }
    // create read fifo operation
    auto streamReadOp = rewriter.create<dataflow::StreamReadOp>(
      loc, memrefType, streamOp.getChannel()
    );
    // if (!loadInfo.ifExprs.empty() && !isa<AffineIfOp>(loadOp->getParentOp())) {
    //   // create affine store operation
    //   rewriter.create<AffineStoreOp>(
    //     loc, streamReadOp.getResult(), loadOp.getMemRef(), loadOp.getAffineMap(), loadOp.getMapOperands()
    //   );
    // } 
    rewriter.replaceOp(loadOp, streamReadOp.getResult());
    rewriter.eraseOp(storeOp);
    rewriter.eraseOp(allocOp);
    return success();
    // if(storeInfo.ifExprs.empty() && loadInfo.ifExprs.empty()){
    //   rewriter.replaceOp(loadOp, streamReadOp.getResult());
    //   if(!isStoreLoadStoreLoad){
    //     rewriter.eraseOp(storeOp);
    //     rewriter.eraseOp(allocOp);
    //   }
    //   return success();
    // } else if(storeInfo.ifExprs.empty() && !loadInfo.ifExprs.empty()){
    //   rewriter.replaceOp(loadOp, streamReadOp.getResult());
    //   if(!isStoreLoadStoreLoad){
    //     rewriter.eraseOp(storeOp);
    //     // rewriter.eraseOp(allocOp);
    //   }
    //   return success();
    // } else if(!storeInfo.ifExprs.empty() && loadInfo.ifExprs.empty()){
    //   rewriter.replaceOp(loadOp, streamReadOp.getResult());
    //   if(!isStoreLoadStoreLoad){
    //     rewriter.eraseOp(storeOp);
    //     rewriter.eraseOp(allocOp);
    //   }
    //   return success();
    // } else {
    //   rewriter.replaceOp(loadOp, streamReadOp.getResult());
    //   if(!isStoreLoadStoreLoad)
    //     rewriter.eraseOp(storeOp);
    //   return success();
    // }
  }
  static LogicalResult storeLoadParallelRewriter(
    PatternRewriter &rewriter,
    memref::AllocOp allocOp,
    AffineStoreOp storeOp,
    AffineLoadOp loadOp,
    AccessInfo storeInfo,
    AccessInfo loadInfo,
    bool isStoreLoadStoreLoad = false
  ){
    auto memrefType = allocOp.getMemref().getType().getElementType();
    auto memrefShape = allocOp.getMemref().getType().cast<MemRefType>().getShape();
    auto memrefSize = 1;
    for(auto dim : memrefShape){
      memrefSize *= dim;
    }
    AffineForOp storeOpFor = storeOp->getParentOfType<AffineForOp>();
    AffineLoopBand storeOpBand;
    getLoopBandFromInnermost(storeOpFor, storeOpBand);
    AffineForOp loadOpFor = loadOp->getParentOfType<AffineForOp>();
    AffineLoopBand loadOpBand;
    getLoopBandFromInnermost(loadOpFor, loadOpBand);
    auto storeNumLoops = storeOpBand.size();
    auto loadNumLoops = loadOpBand.size();
    LLVM_DEBUG(
      llvm::dbgs() << "storeNumLoops: " << storeNumLoops << " loadNumLoops: " << loadNumLoops << "\n";
    );
    SmallVector<Value, 4> storeIvs;
    SmallVector<int64_t, 4> storeUbs;
    // we iterate and get every 2n element of the store and load affine maps since we are doing 1-level tiling
    for(auto iter : llvm::enumerate(storeOp.getMapOperands())){
      auto loopIV = iter.value();
      auto loopIdx = iter.index();
      for(auto forOpPair : enumerate(storeOpBand)){
        auto forOp = forOpPair.value();
        auto idx = forOpPair.index();
        if (loopIV == forOp.getInductionVar()) {
          loopIdx = idx;
        }
      }
      if(loopIdx < storeNumLoops/2)
        continue;
      // auto forOp = loopIV.getDefiningOp();
      if(auto forOp = dyn_cast_if_present<AffineForOp>(loopIV.getParentBlock()->getParentOp())){
        if(forOp.hasConstantBounds()){
          auto ub = forOp.getConstantUpperBound();
          storeIvs.push_back(loopIV);
          storeUbs.push_back(ub);
          LLVM_DEBUG(
            llvm::dbgs() << ub << " ";
          );
        }else{
          assert(false && "store forOp does not have constant bounds");
        }
      }else{
        assert(false && "store affine map operand is not an affine for op");
      }
    }
    LLVM_DEBUG(
      llvm::dbgs() << "\n";
    );
    SmallVector<Value, 4> loadIvs;
    SmallVector<int64_t, 4> loadUbs;
    for(auto iter : llvm::enumerate(loadOp.getMapOperands())){
      auto loopIV = iter.value();
      auto loopIdx = iter.index();
      for(auto forOpPair : enumerate(loadOpBand)){
        auto forOp = forOpPair.value();
        auto idx = forOpPair.index();
        if (loopIV == forOp.getInductionVar()) {
          loopIdx = idx;
        }
      }
      if(loopIdx < loadNumLoops/2)
        continue;
      if(auto forOp = dyn_cast_if_present<AffineForOp>(loopIV.getParentBlock()->getParentOp())){
        if(forOp.hasConstantBounds()){
          auto ub = forOp.getConstantUpperBound();
          loadIvs.push_back(loopIV);
          loadUbs.push_back(ub);
        }else{
          assert(false && "load forOp does not have constant bounds");
        }
      }else{
        assert(false && "load affine map operand is not an affine for op");
      }
    }
    // // print the store and load ivs and ubs
    // for(auto iter : llvm::enumerate(storeIvsUbs)){
    //   auto iv = iter.value().first;
    //   auto ub = iter.value().second;
    //   llvm::dbgs() << "store iv: " << iv << " ub: " << ub << "\n";
    // }
    // for(auto iter : llvm::enumerate(loadIvsUbs)){
    //   auto iv = iter.value().first;
    //   auto ub = iter.value().second;
    //   llvm::dbgs() << "load iv: " << iv << " ub: " << ub << "\n";
    // }
    for(auto storeLoadUbs : llvm::zip(storeUbs, loadUbs)){
      auto storeUb = std::get<0>(storeLoadUbs);
      auto loadUb = std::get<1>(storeLoadUbs);
      if(storeUb != loadUb){
        LLVM_DEBUG(
          llvm::dbgs() << "storeUb != loadUb\n"
        );
        return failure();
      }
    }
    auto unrollFactor = 1;
    for(auto storeUb : storeUbs){
      unrollFactor *= storeUb;
    }

    auto loc = rewriter.getUnknownLoc();
    // create fifo
    rewriter.setInsertionPoint(allocOp);

    auto streamType = StreamType::get(rewriter.getContext(), memrefType, memrefSize/unrollFactor);
    auto rankedTensorType = RankedTensorType::get(storeUbs, streamType);
    // auto tensorOfStreamsOp = rewriter.create<tensor::EmptyOp>(
    //   loc,
    //   storeUbs,
    //   streamType
    // );
    // auto streamOp = rewriter.create<dataflow::StreamOp>(
    //   loc,
    //   dataflow::StreamType::get(rewriter.getContext(), memrefType, memrefSize),
    //   memrefSize
    // );
    auto arrayOfStreamsOp = rewriter.create<ArrayOfStreamsOp>(
      loc,
      rankedTensorType
    );

    rewriter.setInsertionPoint(storeOp);
    if(!storeInfo.ifExprs.empty() && !isa<AffineIfOp>(storeOp->getParentOp())){
      // create write fifo operation
      auto ifCondition = IntegerSet::get(storeInfo.ifExprs.size(), 0, storeInfo.ifExprs, storeInfo.ifEqFlags);
      auto ifOp = rewriter.create<AffineIfOp>(
            loc, ifCondition, storeInfo.ifOperands, false);
      rewriter.setInsertionPointToStart(ifOp.getThenBlock());
    }
    // // create write fifo operation
    // rewriter.create<dataflow::StreamWriteOp>(
    //   loc, streamOp.getChannel(), storeOp.getValueToStore());

    // // create write fifo tensor operation
    // auto writeFifo = rewriter.create<tensor::ExtractOp>(
    //   loc,
    //   streamType,
    //   tensorOfStreamsOp.getResult(),
    //   storeIvs
    // );
    // rewriter.create<dataflow::StreamWriteOp>(
    //   loc, writeFifo.getResult(), storeOp.getValueToStore()
    // );
    auto arrayOfStreamsWrite = rewriter.create<ArrayOfStreamsWriteOp>(
      loc,
      arrayOfStreamsOp.getResult(),
      storeOp.getValueToStore(),
      storeIvs,
      AffineMap::getMultiDimIdentityMap(storeIvs.size(), rewriter.getContext())
    );

    // create read fifo
    rewriter.setInsertionPoint(loadOp);
    if (!loadInfo.ifExprs.empty() && !isa<AffineIfOp>(loadOp->getParentOp())) {
      auto ifCondition = IntegerSet::get(loadInfo.ifExprs.size(), 0, loadInfo.ifExprs, loadInfo.ifEqFlags);
      auto ifOp = rewriter.create<AffineIfOp>(
            loc, ifCondition, loadInfo.ifOperands, false);
      rewriter.setInsertionPointToStart(ifOp.getThenBlock());
    }
    // // create read fifo operation
    // auto streamReadOp = rewriter.create<dataflow::StreamReadOp>(
    //   loc, memrefType, streamOp.getChannel()
    // );

    // // create read fifo tensor operation
    // auto readFifo = rewriter.create<tensor::ExtractOp>(
    //   loc,
    //   streamType,
    //   tensorOfStreamsOp.getResult(),
    //   loadIvs
    // );
    // auto readFifoResult = rewriter.create<dataflow::StreamReadOp>(
    //   loc, memrefType, readFifo.getResult()
    // );
    auto arrayOfStreamsRead = rewriter.create<ArrayOfStreamsReadOp>(
      loc,
      arrayOfStreamsOp.getElementType(),
      arrayOfStreamsOp.getResult(),
      loadIvs,
      AffineMap::getMultiDimIdentityMap(loadIvs.size(), rewriter.getContext())
    );

    rewriter.replaceOp(loadOp, arrayOfStreamsRead.getResult());
    rewriter.eraseOp(storeOp);
    rewriter.eraseOp(allocOp);
    return success();
  }
  // static IntegerRelation getRel(memref::AllocOp allocOp){
  //   auto memRef = allocOp.getResult();
  //   auto context = allocOp.getContext();
  //   auto memRefType = memRef.getType().cast<MemRefType>();
  //   auto memRefShape = memRefType.getShape();
  //   // lbs and ubs
  //   SmallVector<int64_t, 4> lbs;
  //   SmallVector<int64_t, 4> ubs;
  //   for(auto dim : memRefShape){
  //     lbs.push_back(0);
  //     ubs.push_back(dim);
  //   }

  //   // creating timestamp equation
  //   SmallVector<int64_t, 4> timeEq;
  //   auto idx = 0;
  //   for(auto currDim : memRefShape){
  //     auto var = 1;
  //     // iterate over ivs starting from current iv to the end of ivs
  //     for(auto nextDim : llvm::drop_begin(memRefShape, idx + 1)){
  //       var *= nextDim;
  //     }
  //     idx++;
  //     timeEq.push_back(var);
  //   }
  //   timeEq.push_back(-1);
  //   timeEq.push_back(0);

  //   // create integer relation
  //   auto rel = presburger::IntegerRelation(
  //     presburger::PresburgerSpace::getRelationSpace(
  //       memRefShape.size(), /* domain size */ 
  //       1, /* range size */
  //       0, /* symbols */
  //       0  /* locals */
  //     )
  //   );

  //   // add equalities
  //   rel.addEquality(timeEq);

  //   SmallVector<SmallVector<int64_t, 4>, 4> constraints;
  //   auto colIdx = 0;
  //   for(auto idx = 0; idx < memRefShape.size(); idx++){
  //     SmallVector<int64_t, 4> lbConstraint;
  //     SmallVector<int64_t, 4> ubConstraint;
  //     for(auto i = 0; i < memRefShape.size(); i++){
  //       if(i == idx){
  //         lbConstraint.push_back(1);
  //         ubConstraint.push_back(-1);
  //       }else{
  //         lbConstraint.push_back(0);
  //         ubConstraint.push_back(0);
  //       }
  //     }
  //     lbConstraint.push_back(0); // for time stamp
  //     ubConstraint.push_back(0); // for time stamp
  //     lbConstraint.push_back(lbs[idx]);
  //     ubConstraint.push_back(ubs[idx] - 1);
  //     constraints.push_back(lbConstraint);
  //     constraints.push_back(ubConstraint);
  //   }
  //   // add constraints
  //   for(auto constraint : constraints){
  //     rel.addInequality(constraint);
  //   }
  //   return rel;
    
  // }
  // static AffineMap getAccessPerm(Operation *op){
  //   if(!isa<AffineLoadOp, AffineStoreOp>(op)){
  //     assert(false && "op is not a load or store operation");
  //   }
  //   SmallVector<Operation*, 4> relevantOps;
  //   SmallVector<Operation*, 4> irrelevantOps;
  //   getRelevantAndIrrelevantEnclosingAffineOps(*op, &relevantOps, &irrelevantOps);

  //   // handle load or store
  //   SmallVector<Value, 4> operands;
  //   if(auto loadOp = dyn_cast<AffineLoadOp>(op)){
  //     operands = loadOp.getMapOperands();
  //   }else if(auto storeOp = dyn_cast<AffineStoreOp>(op)){
  //     operands = storeOp.getMapOperands();
  //   }

  //   SmallVector<unsigned, 4> perm;
  //   for(auto operandPair : llvm::enumerate(operands)){
  //     auto operand = operandPair.value();
  //     for(auto relevantOpPair : llvm::enumerate(relevantOps)){
  //       auto loopIdx = relevantOpPair.index();
  //       auto relevantOp = relevantOpPair.value();
  //       auto forOp = dyn_cast<AffineForOp>(relevantOp);
  //       if(forOp){
  //         if(operand == forOp.getInductionVar()){
  //           perm.push_back(loopIdx);
  //         }
  //       }
  //     }
  //   }
  //   return AffineMap::getPermutationMap(perm, op->getContext());
  // }
  struct StoreLoadPattern : public OpRewritePattern<memref::AllocOp> {
    StoreLoadPattern(MLIRContext *context, bool parallelizeNodes) 
      : OpRewritePattern(context), parallelizeNodes(parallelizeNodes) {}
    using OpRewritePattern<memref::AllocOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(
      memref::AllocOp allocOp,
      PatternRewriter &rewriter
    ) const override {
      auto memRef = allocOp.getResult();
      auto context = rewriter.getContext();
      SmallVector<AffineLoadOp, 4> loadOps;
      SmallVector<AffineStoreOp, 4> storeOps;
      for(auto user : memRef.getUsers()){
        if(auto loadOp = dyn_cast<AffineLoadOp>(user)){
          // loadOp.getAffineMap().dump();
          loadOps.push_back(loadOp);
        }
        if(auto storeOp = dyn_cast<AffineStoreOp>(user)){
          // storeOp.getAffineMap().dump();
          storeOps.push_back(storeOp);
        }
      }
      // return failure();
      // if there is only one load and one store
      if(loadOps.size() == 1 && storeOps.size() == 1){
        auto loadOp = loadOps[0];
        auto storeOp = storeOps[0];
        auto loadOpParentFor = loadOp->getParentOfType<AffineForOp>();
        auto storeOpParentFor = storeOp->getParentOfType<AffineForOp>();
        if(loadOpParentFor && storeOpParentFor){
          if(loadOpParentFor == storeOpParentFor){
            LLVM_DEBUG(
              llvm::dbgs() << "StoreLoadPattern: loadOpParentFor == storeOpParentFor\n"
            );
            return failure();
          }
        }
        LLVM_DEBUG(
          allocOp.dump();
        );
        AccessInfo loadInfo(loadOp);
        AccessInfo storeInfo(storeOp);
        if(failed(loadInfo.getAllInfo(context))){
          LLVM_DEBUG(
            llvm::dbgs() << "loadInfo.getAllInfo failed\n"
          );
          return failure();
        }
        if(failed(storeInfo.getAllInfo(context))){
          LLVM_DEBUG(
            llvm::dbgs() << "storeInfo.getAllInfo failed\n"
          );
          return failure();
        }
        // auto storeTimeIndexRelation = storeInfo.getTimeIndexRelation();
        // auto loadTimeIndexRelation = loadInfo.getTimeIndexRelation();
        // if(storeTimeIndexRelation.getNumConstraints() && loadTimeIndexRelation.getNumConstraints()){
        //   if(storeTimeIndexRelation.isEqual(loadTimeIndexRelation)){
        // auto loadSpace = loadInfo.getRelevantSpace();
        // auto storeSpace = storeInfo.getRelevantSpace();
        // auto loadRelevantRel = loadInfo.relevantRelation;
        // auto storeRelevantRel = storeInfo.relevantRelation;
        // auto loadRelevantTimeIndexRelation = loadInfo.getTimeIndexRelevantRelation();
        // auto storeRelevantTimeIndexRelation = storeInfo.getTimeIndexRelevantRelation();
        // AffineLoopBand storeBand;
        // auto storeInnerMostForOp = storeOp->getParentOfType<AffineForOp>();
        // getLoopBandFromInnermost(storeInnerMostForOp, storeBand);

        // AffineLoopBand loadBand;
        // auto loadInnerMostForOp = loadOp->getParentOfType<AffineForOp>();
        // getLoopBandFromInnermost(loadInnerMostForOp, loadBand);
        // auto storeAP = getAccessPattern(storeOp);
        // auto storeMinAP = getMinimalAccessPattern(storeOp);

        // if(storeAP != storeMinAP && storeAP.isPermutation()){
        //   llvm::dbgs() << "======================================================================================================================================\n";
        //   llvm::dbgs() << "storeBand: \n"; 
        //   storeAP.dump();
        //   storeMinAP.dump(); 
        //   llvm::dbgs() << "-------------------------------------------------------------------\n";
        //   // storeBand[0].dump();
        //   for(auto operand : storeOp.getMapOperands()){
        //     operand.getParentBlock()->getParentOp()->dump();
        //   }
        //   storeOp.dump();
        //   storeOp.getAffineMap().dump();
        //   // getAccessPattern(storeOp);
        // }
        // // llvm::dbgs() << "-------------------------------------------------------------------\n";

        // auto loadAP = getAccessPattern(loadOp);
        // auto loadMinAP = getMinimalAccessPattern(loadOp);
        // if(loadAP != loadMinAP && loadAP.isPermutation()){
        //   llvm::dbgs() << "======================================================================================================================================\n";
        //   llvm::dbgs() << "loadBand: \n";
        //   loadAP.dump();
        //   loadMinAP.dump();
        //   llvm::dbgs() << "-------------------------------------------------------------------\n";
        //   // loadBand[0].dump();
        //   for(auto operand : loadOp.getMapOperands()){
        //     operand.getParentBlock()->getParentOp()->dump();
        //   }
        //   loadOp.dump();
        //   loadOp.getAffineMap().dump();
        //   // getAccessPattern(loadOp);
        // }
        // llvm::dbgs() << "-------------------------------------------------------------------\n";
        // llvm::dbgs() << "loadBand: \n";
        // // loadBand[0].dump();
        // for(auto operand : loadOp.getMapOperands()){
        //   operand.getParentBlock()->getParentOp()->dump();
        // }
        // loadOp.dump();
        // // loadOp.getAffineMap().dump();
        // getAccessPattern(loadOp);
        // llvm::dbgs() << "======================================================================================================================================\n";
        // return failure();
        // return failure();
        // llvm::dbgs() << "\n\n";
        // return failure();
        // getAccessPerm(storeOp).dump();
        // getAccessPerm(loadOp).dump();
        // storeRelevantRel.dump();
        // storeRelevantTimeIndexRelation.dump();
        // loadRelevantRel.dump();
        // loadRelevantTimeIndexRelation.dump();

        // llvm::dbgs() << "is equal: " << storeRelevantRel.isEqual(loadRelevantRel) << "\n";
        // llvm::dbgs() << "is equal: " << storeRelevantTimeIndexRelation.isEqual(loadRelevantTimeIndexRelation) << "\n";
        // if(/*loadSpace.isCompatible(storeSpace)*/ true){
        //   if(/*loadInfo.relevantRelation.isEqual(storeInfo.relevantRelation)*/ true){
        // if(loadSpace.isCompatible(storeSpace)){
        //   if(loadInfo.relevantRelation.isEqual(storeInfo.relevantRelation)){
        // allocOp.dump();
        // storeOp.dump();
        // loadOp.dump();
        // getAccessPattern(storeOp).dump();
        // getAccessPattern(loadOp).dump();
        // llvm::dbgs() << parallelizeNodes << "\n\n";
        // auto storeFlag = true;
        // auto loadFlag = true;
        auto storeMap = storeOp.getAffineMap();
        auto loadMap = loadOp.getAffineMap();
        // llvm::dbgs() << "store equals load map: " << (getMinimalAccessPattern(storeOp) == getMinimalAccessPattern(loadOp)) << " : " << (storeMap == loadMap)  << "\n";
        // storeOp.dump();
        // storeMap.dump();
        // loadMap.dump();
        // for(auto expr : storeMap.getResults()){
        //   expr.walk([&](AffineExpr expr){
        //     if(auto dimExpr = dyn_cast<AffineDimExpr>(expr)){
        //       auto kind = expr.getKind();
        //       if(kind == AffineExprKind::Mod || kind == AffineExprKind::FloorDiv){
        //         storeFlag = false;
        //       }
        //     }
        //   });
        // }
        // for(auto expr : loadMap.getResults()){
        //   expr.walk([&](AffineExpr expr){
        //     if(auto dimExpr = dyn_cast<AffineDimExpr>(expr)){
        //       auto kind = expr.getKind();
        //       if(kind == AffineExprKind::Mod || kind == AffineExprKind::FloorDiv){
        //         loadFlag = false;
        //       }
        //     }
        //   });
        // }
        // if(storeFlag == false || loadFlag == false){
        //   return failure();
        // }

        if(getMinimalAccessPattern(storeOp) == getMinimalAccessPattern(loadOp) && (storeMap == loadMap)){ // TODO: tmp fix for multi head self attention
          if(true){
        // auto loadSpace = loadInfo.getSpace();
        // auto storeSpace = storeInfo.getSpace();
        // if (loadSpace.isCompatible(storeSpace)) {
        //   if (loadInfo.relation.isEqual(storeInfo.relation)) {
          if(parallelizeNodes){
            if(failed(storeLoadParallelRewriter(
              rewriter,
              allocOp,
              storeOp,
              loadOp,
              storeInfo,
              loadInfo
            ))) {
              LLVM_DEBUG(
                llvm::dbgs() << "StoreLoadPattern: storeLoadRewriter failed\n"
              );
              return failure();
            }
          }else{
            if(failed(storeLoadRewriter(
              rewriter,
              allocOp,
              storeOp,
              loadOp,
              storeInfo,
              loadInfo
            ))) {
              LLVM_DEBUG(
                llvm::dbgs() << "StoreLoadPattern: storeLoadRewriter failed\n"
              );
              return failure();
            }
          }
          } else{
            LLVM_DEBUG(
              llvm::dbgs() << "StoreLoadPattern: loadRelevantRelation is not equal to storeRelevantRelation\n"
            );
            return failure();
          }
        } else {
          LLVM_DEBUG(
            llvm::dbgs() << "StoreLoadPattern: loadSpace is not compatible with storeSpace\n"
          );
          return failure();
        }
      }
      return success();
    }
    private:
      bool parallelizeNodes;
  };
} // namespace

namespace {
struct ConvertMemRefsToFIFOs
    : public ConvertMemRefsToFIFOsBase<ConvertMemRefsToFIFOs> {
  ConvertMemRefsToFIFOs() = default;
  ConvertMemRefsToFIFOs(bool argParallelizeNodes){
    parallelizeNodes = argParallelizeNodes;
  }
  void runOnOperation() override {
    // llvm::dbgs() << "ConvertMemRefsToFIFOs\n";
    auto func = getOperation();
    auto context = func.getContext();
    mlir::RewritePatternSet patterns(context);
    // patterns.add<StoreLoadStoreLoadPattern>(context);
    // (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
    // patterns.clear();
    patterns.add<StoreLoadPattern>(context, parallelizeNodes);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  
  }
};
} // namespace

std::unique_ptr<Pass> streamhls::createConvertMemRefsToFIFOsPass(bool parallelizeNodes) {
  return std::make_unique<ConvertMemRefsToFIFOs>(parallelizeNodes);
}