/*
 * Copyright (c) 2024 Suhail Basalama
 *
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 */
#include <optional>
#include <string>
#include <regex>
#include <iostream>
#include <memory>
#include <array>
#include <stdexcept>
#include <random>
#include <chrono>
#include <ctime>
#include "streamhls/Support/DFG.h"
#include "streamhls/Support/AffineMemAccess.h"
#include "streamhls/Support/Utils.h"
// #include "gurobi_c++.h"

using namespace mlir;
using namespace streamhls;
using namespace dataflow;
using namespace affine;

using Node = DFG::Node;

#define DEBUG_TYPE "streamhls-dfg"

// LoopNestStateCollector walks loop nests and collects load and store
// operations, and whether or not a region holding op other than ForOp and IfOp
// was encountered in the loop nest.
void LoopNestInfo::collect(Operation *opToWalk) {
  opToWalk->walk([&](Operation *op) {
    if (isa<AffineForOp>(op))
      forOps.push_back(cast<AffineForOp>(op));
    else if (op->getNumRegions() != 0 && !isa<AffineIfOp>(op))
      hasNonAffineRegionOp = true;
    else if (isa<AffineReadOpInterface>(op)){
      loadOpInsts.push_back(op);
      auto memref = cast<AffineReadOpInterface>(op).getMemRef();
      if(auto allocOp = memref.getDefiningOp<memref::AllocOp>()){
        // if memref is not in allocOps, add it
        // TODO: mayber we need to check if the memref is the func argument
        if(std::find(allocOps.begin(), allocOps.end(), allocOp) == allocOps.end()){
          allocOps.push_back(allocOp);
        }
      }
    }
    else if (isa<AffineWriteOpInterface>(op)){
      storeOpInsts.push_back(op);
      auto memref = cast<AffineWriteOpInterface>(op).getMemRef();
      if(auto allocOp = memref.getDefiningOp<memref::AllocOp>()){
        // if memref is not in allocOps, add it
        // TODO: mayber we need to check if the memref is the func argument
        if(std::find(allocOps.begin(), allocOps.end(), allocOp) == allocOps.end()){
          allocOps.push_back(allocOp);
        }
      }
    }
    else if (isa<StreamReadOp>(op))
      fifoReadOps.push_back(op);
    else if (isa<StreamWriteOp>(op))
      fifoWriteOps.push_back(op);
  });
}

// // Returns the load op count for 'memref'.
// unsigned Node::getLoadOpCount(Value memref) const {
//   unsigned loadOpCount = 0;
//   for (Operation *loadOp : loads) {
//     if (memref == cast<AffineReadOpInterface>(loadOp).getMemRef())
//       ++loadOpCount;
//   }
//   return loadOpCount;
// }

// // Returns the store op count for 'memref'.
// unsigned Node::getStoreOpCount(Value memref) const {
//   unsigned storeOpCount = 0;
//   for (Operation *storeOp : stores) {
//     if (memref == cast<AffineWriteOpInterface>(storeOp).getMemRef())
//       ++storeOpCount;
//   }
//   return storeOpCount;
// }

// // Returns the FIFO read op count for 'fifo'.
// unsigned Node::getFifoReadOpCount(Value fifo) const {
//   unsigned fifoReadOpCount = 0;
//   for (Operation *fifoReadOp : fifoReads) {
//     if (fifo == cast<StreamReadOp>(fifoReadOp).getChannel())
//       ++fifoReadOpCount;
//   }
//   return fifoReadOpCount;
// }

// // Returns the FIFO write op count for 'fifo'.
// unsigned Node::getFifoWriteOpCount(Value fifo) const {
//   unsigned fifoWriteOpCount = 0;
//   for (Operation *fifoWriteOp : fifoWrites) {
//     if (fifo == cast<StreamWriteOp>(fifoWriteOp).getChannel())
//       ++fifoWriteOpCount;
//   }
//   return fifoWriteOpCount;
// }

// // Returns all store ops in 'storeOps' which access 'memref'.
// void Node::getStoreOpsForMemref(Value memref,
//                                 SmallVectorImpl<Operation *> *storeOps) const {
//   for (Operation *storeOp : stores) {
//     if (memref == cast<AffineWriteOpInterface>(storeOp).getMemRef())
//       storeOps->push_back(storeOp);
//   }
// }

// // Returns all load ops in 'loadOps' which access 'memref'.
// void Node::getLoadOpsForMemref(Value memref,
//                                SmallVectorImpl<Operation *> *loadOps) const {
//   for (Operation *loadOp : loads) {
//     if (memref == cast<AffineReadOpInterface>(loadOp).getMemRef())
//       loadOps->push_back(loadOp);
//   }
// }



// // Returns all memrefs in 'loadAndStoreMemrefSet' for which this node
// // has at least one load and store operation.
// void Node::getLoadAndStoreMemrefSet(
//     DenseSet<Value> *loadAndStoreMemrefSet) const {
//   llvm::SmallDenseSet<Value, 2> loadMemrefs;
//   for (Operation *loadOp : loads) {
//     loadMemrefs.insert(cast<AffineReadOpInterface>(loadOp).getMemRef());
//   }
//   for (Operation *storeOp : stores) {
//     auto memref = cast<AffineWriteOpInterface>(storeOp).getMemRef();
//     if (loadMemrefs.count(memref) > 0)
//       loadAndStoreMemrefSet->insert(memref);
//   }
// }

// Initializes the data dependence graph by walking operations in `block`.
// Assigns each node in the graph a node id based on program order in 'f'.
bool DFG::init(bool mergeAllocNodes) {
  LLVM_DEBUG(llvm::dbgs() << "--- Initializing MDG ---\n");
  DenseMap<AffineForOp, unsigned> visitedLoops;
  unsigned id = 0;
  for (Operation &op : block) {
    if (auto streamOp = dyn_cast<StreamOp> (&op)) {
      auto users = streamOp.getResult().getUsers();
      unsigned srcId;
      unsigned dstId;
      for(auto user : users){
        // LLVM_DEBUG(llvm::dbgs() << "User: \n");
        if(auto streamWriteOp = dyn_cast<StreamWriteOp>(user)){
          AffineLoopBand band;
          auto innerMostForOp = streamWriteOp.getOperation()->getParentOfType<AffineForOp>();
          getLoopBandFromInnermost(innerMostForOp, band);
          auto outerMostForOp = band[0];
          // outerMostForOp->dump();
          // if outerMostForOp is not visited, create new node, else get the current node
          // if(std::find(visitedLoops.begin(), visitedLoops.end(), outerMostForOp) == visitedLoops.end()){
          if(visitedLoops.count(outerMostForOp) == 0){
            Node node(id, outerMostForOp);
            // get loopband trip count
            node.tripCount = getLoopNestIterations(band);
            visitedLoops.insert({outerMostForOp, node.id});
            node.forOps.push_back(outerMostForOp);
            nodes.insert({node.id, node});
            srcId = node.id;
            id++;
          } else {
            auto nodeId = visitedLoops[outerMostForOp];
            srcId = nodeId;
          }
          // LLVM_DEBUG(llvm::dbgs() << "Adding node " << srcNode->id << " with op: \n");
        }
        if(auto streamReadOp = dyn_cast<StreamReadOp>(user)){
          AffineLoopBand band;
          auto innerMostForOp = streamReadOp.getOperation()->getParentOfType<AffineForOp>();
          getLoopBandFromInnermost(innerMostForOp, band);
          auto outerMostForOp = band[0];
          // outerMostForOp->dump();
          // if outerMostForOp is not visited, create new node, else get the current node
          // if(std::find(visitedLoops.begin(), visitedLoops.end(), outerMostForOp) == visitedLoops.end()){
          if(visitedLoops.count(outerMostForOp) == 0){
            Node node(id, outerMostForOp);
            // get loopband trip count
            node.tripCount = getLoopNestIterations(band);
            visitedLoops.insert({outerMostForOp, node.id});
            node.forOps.push_back(outerMostForOp);
            nodes.insert({node.id, node});
            dstId = node.id;
            id++;
          } else {
            auto nodeId = visitedLoops[outerMostForOp];
            dstId = nodeId;
          }
          // LLVM_DEBUG(llvm::dbgs() << "Adding node " << dstNode->id << " with op: \n");
        }
      }
      auto srcNode = getNode(srcId);
      auto dstNode = getNode(dstId);
      // LLVM_DEBUG(llvm::dbgs() << "Adding edge from " << srcNode->id << " to " << dstNode->id << "\n");
      addEdge(srcNode->id, dstNode->id, streamOp.getResult());
      // edgeMap[streamOp.getResult()] = {-1};
      // edges.push_back({srcNode->id, dstNode->id, streamOp.getResult()});

    } 
    if (auto aofsOp = dyn_cast<ArrayOfStreamsOp> (&op)) {
      auto users = aofsOp.getResult().getUsers();
      unsigned srcId;
      unsigned dstId;
      for(auto user : users){
        // LLVM_DEBUG(llvm::dbgs() << "User: \n");
        if(auto aofsWriteOp = dyn_cast<ArrayOfStreamsWriteOp>(user)){
          AffineLoopBand band;
          auto innerMostForOp = aofsWriteOp.getOperation()->getParentOfType<AffineForOp>();
          getLoopBandFromInnermost(innerMostForOp, band);
          auto outerMostForOp = band[0];
          // outerMostForOp->dump();
          // if outerMostForOp is not visited, create new node, else get the current node
          // if(std::find(visitedLoops.begin(), visitedLoops.end(), outerMostForOp) == visitedLoops.end()){
          if(visitedLoops.count(outerMostForOp) == 0){
            Node node(id, outerMostForOp);
            // get loopband trip count
            node.tripCount = getLoopNestIterations(band);
            visitedLoops.insert({outerMostForOp, node.id});
            node.forOps.push_back(outerMostForOp);
            nodes.insert({node.id, node});
            srcId = node.id;
            id++;
          } else {
            auto nodeId = visitedLoops[outerMostForOp];
            srcId = nodeId;
          }
          // LLVM_DEBUG(llvm::dbgs() << "Adding node " << srcNode->id << " with op: \n");
        }
        if(auto aofsReadOp = dyn_cast<ArrayOfStreamsReadOp>(user)){
          AffineLoopBand band;
          auto innerMostForOp = aofsReadOp.getOperation()->getParentOfType<AffineForOp>();
          getLoopBandFromInnermost(innerMostForOp, band);
          auto outerMostForOp = band[0];
          // outerMostForOp->dump();
          // if outerMostForOp is not visited, create new node, else get the current node
          // if(std::find(visitedLoops.begin(), visitedLoops.end(), outerMostForOp) == visitedLoops.end()){
          if(visitedLoops.count(outerMostForOp) == 0){
            Node node(id, outerMostForOp);
            // get loopband trip count
            node.tripCount = getLoopNestIterations(band);
            visitedLoops.insert({outerMostForOp, node.id});
            node.forOps.push_back(outerMostForOp);
            nodes.insert({node.id, node});
            dstId = node.id;
            id++;
          } else {
            auto nodeId = visitedLoops[outerMostForOp];
            dstId = nodeId;
          }
          // LLVM_DEBUG(llvm::dbgs() << "Adding node " << dstNode->id << " with op: \n");
        }
      }
      auto srcNode = getNode(srcId);
      auto dstNode = getNode(dstId);
      // LLVM_DEBUG(llvm::dbgs() << "Adding edge from " << srcNode->id << " to " << dstNode->id << "\n");
      addEdge(srcNode->id, dstNode->id, aofsOp.getResult());
      // edgeMap[streamOp.getResult()] = {-1};
      // edges.push_back({srcNode->id, dstNode->id, streamOp.getResult()});

    } 
    if (auto allocOp = dyn_cast<memref::AllocOp>(&op)){
      auto users = allocOp.getResult().getUsers();
      SmallVector<AffineStoreOp, 2> srcOps;
      SmallVector<AffineLoadOp, 2> dstOps;
      SmallVector<unsigned, 2> srcIds;
      SmallVector<unsigned, 2> dstIds;
      for(auto user : users){
        if(auto storeOp = dyn_cast<AffineStoreOp>(user)){
          auto innerMostForOp = storeOp.getOperation()->getParentOfType<AffineForOp>();
          AffineLoopBand band;
          getLoopBandFromInnermost(innerMostForOp, band);
          auto outerMostForOp = band[0];
          // if outerMostForOp is not visited, create new node, else get the current node
          if(visitedLoops.count(outerMostForOp) == 0){
            Node node(id, outerMostForOp);
            // get loopband trip count
            node.tripCount = getLoopNestIterations(band);
            visitedLoops.insert({outerMostForOp, node.id});
            node.forOps.push_back(outerMostForOp);
            nodes.insert({node.id, node});
            srcIds.push_back(node.id);
            srcOps.push_back(storeOp);
            id++;
          } else {
            auto nodeId = visitedLoops[outerMostForOp];
            srcIds.push_back(nodeId);
            srcOps.push_back(storeOp);
          }
        }
        if(auto loadOp = dyn_cast<AffineLoadOp>(user)){
          auto innerMostForOp = loadOp.getOperation()->getParentOfType<AffineForOp>();
          AffineLoopBand band;
          getLoopBandFromInnermost(innerMostForOp, band);
          auto outerMostForOp = band[0];
          // if outerMostForOp is not visited, create new node, else get the current node
          if(visitedLoops.count(outerMostForOp) == 0){
            Node node(id, outerMostForOp);
            // get loopband trip count
            node.tripCount = getLoopNestIterations(band);
            visitedLoops.insert({outerMostForOp, node.id});
            node.forOps.push_back(outerMostForOp);
            nodes.insert({node.id, node});
            dstIds.push_back(node.id);
            dstOps.push_back(loadOp);
            id++;
          } else {
            auto nodeId = visitedLoops[outerMostForOp];
            dstIds.push_back(nodeId);
            dstOps.push_back(loadOp);
          }
        }
      }
      for(auto src : llvm::zip(srcIds, srcOps)){
        for(auto dst : llvm::zip(dstIds, dstOps)){
          auto srcId = std::get<0>(src);
          auto dstId = std::get<0>(dst);
          auto srcOp = std::get<1>(src);
          auto dstOp = std::get<1>(dst);
          auto srcNode = getNode(srcId);
          auto dstNode = getNode(dstId);
          if(srcNode->id != dstNode->id){
            addEdge(srcNode->id, dstNode->id, srcOp, dstOp, allocOp.getResult());
            edges.push_back({srcNode->id, dstNode->id, allocOp.getResult()});
            // edgeMap[allocOp.getResult()] = {-1};
          }else{
            srcNode->allocOps.insert(srcNode->allocOps.begin(), allocOp);
          }
        }
      }
      
    }
  }
  if(mergeAllocNodes){
    SmallVector<std::pair<unsigned, unsigned>, 4> mergedNodes;
    for(auto edge : edges){
      LLVM_DEBUG(llvm::dbgs() << "Merging " << edge.srcId << " to " << edge.dstId << "\n");
      // LLVM_DEBUG(llvm::dbgs() << "Num nodes: " << nodes.size() << "\n");
      // for(auto node : nodes){
      //   LLVM_DEBUG(llvm::dbgs() << "Node: " << node.first << " " << node.second.id << "\n");
      // }
      // if the edge is already merged, skip
      if(std::find(mergedNodes.begin(), mergedNodes.end(), std::make_pair(edge.srcId, edge.dstId)) != mergedNodes.end()){
        continue;
      }
      for(auto &e : edges){
        LLVM_DEBUG(llvm::dbgs() << "Edge: " << e.srcId << " " << e.dstId << "\n");
      }
      LLVM_DEBUG(llvm::dbgs() << "Merging nodes...\n");
      mergeNodes(edge.srcId, edge.dstId);
      // update the edges
      for(auto &e : edges){
        if(e.dstId == edge.srcId){
          e.dstId = edge.dstId;
        }
        if(e.srcId == edge.srcId){
          e.srcId = edge.dstId;
        }
      }
      for(auto &e : edges){
        LLVM_DEBUG(llvm::dbgs() << "Edge: " << e.srcId << " " << e.dstId << "\n");
      }
      // LLVM_DEBUG(llvm::dbgs() << "Num nodes: " << nodes.size() << "\n");
      // for(auto node : nodes){
      //   LLVM_DEBUG(llvm::dbgs() << "Node: " << node.first << " " << node.second.id << "\n");
      // }
      mergedNodes.push_back({edge.srcId, edge.dstId});
    }
  }
  return true;
}

// // Initializes the data dependence graph by walking operations in `block`.
// // Assigns each node in the graph a node id based on program order in 'f'.
// bool DFG::init(bool mergeAllocNodes) {
//   LLVM_DEBUG(llvm::dbgs() << "--- Initializing Dataflow Graph ---\n");
//   // Map from a memref to the set of ids of the nodes that have ops accessing
//   // the memref.
//   DenseMap<Value, SetVector<unsigned>> memrefAccesses;
//   DenseMap<Value, SetVector<unsigned>> fifoAccesses;
//   DenseMap<Operation *, unsigned> forToNodeMap;
//   for (Operation &op : block) {
//     if (dyn_cast<AffineForOp>(op)) {
//       // Create graph node 'id' to represent top-level 'forOp' and record
//       // all loads and store accesses it contains.
//       LoopNestInfo collector;
//       collector.collect(&op);
//       // Return false if a region holding op other than 'affine.for' and
//       // 'affine.if' was found (not currently supported).
//       if (collector.hasNonAffineRegionOp)
//         return false;
//       Node node(nextNodeId++, &op);
//       for (auto *opInst : collector.loadOpInsts) {
//         // opInst->dump();
//         node.loads.push_back(opInst);
//         auto memref = cast<AffineReadOpInterface>(opInst).getMemRef();
//         // if(auto allocOp = memref.getDefiningOp<memref::AllocOp>()){
//         //   // if memref is not in allocOps, add it
//         //   if(std::find(node.allocOps.begin(), node.allocOps.end(), allocOp) == node.allocOps.end()){
//         //     node.allocOps.push_back(allocOp);
//         //   }
//         // }
//         memrefAccesses[memref].insert(node.id);
//       }
//       for (auto *opInst : collector.storeOpInsts) {
//         // opInst->dump();
//         node.stores.push_back(opInst);
//         auto memref = cast<AffineWriteOpInterface>(opInst).getMemRef();
//         // if(auto allocOp = memref.getDefiningOp<memref::AllocOp>()){
//         //   // if memref is not in allocOps, add it
//         //   if(std::find(node.allocOps.begin(), node.allocOps.end(), allocOp) == node.allocOps.end()){
//         //     node.allocOps.push_back(allocOp);
//         //   }
//         // }
//         memrefAccesses[memref].insert(node.id);
//       }
//       for (auto *opInst : collector.fifoReadOps) {
//         // opInst->dump();
//         node.fifoReads.push_back(opInst);
//         auto fifo = cast<StreamReadOp>(opInst).getChannel();
//         fifoAccesses[fifo].insert(node.id);
//       }
//       for (auto *opInst : collector.fifoWriteOps) {
//         // opInst->dump();
//         node.fifoWrites.push_back(opInst);
//         auto fifo = cast<StreamWriteOp>(opInst).getChannel();
//         fifoAccesses[fifo].insert(node.id);
//       }
//       forToNodeMap[&op] = node.id;
//       nodes.insert({node.id, node});
//       // TODO: insert alloc and for ops in the node
//       node.forOps.push_back(cast<AffineForOp>(&op));
//       op.dump();
//       for(auto allocOp : collector.allocOps){
//         allocOp.dump();
//         node.allocOps.push_back(allocOp);
//       }
//       LLVM_DEBUG(llvm::dbgs() <<"\n");
//     } 

//     else if (op.getNumRegions() != 0) {
//       // Return false if another region is found (not currently supported).
//       return false;
//     } 
//   }

//   // // Add dependence edges between nodes which produce SSA values and their
//   // // users. Load ops can be considered as the ones producing SSA values.
//   // for (auto &idAndNode : nodes) {
//   //   const Node &node = idAndNode.second;
//   //   // Stores don't define SSA values, skip them.
//   //   if (!node.stores.empty())
//   //     continue;
//   //   Operation *opInst = node.op;
//   //   for (Value value : opInst->getResults()) {
//   //     for (Operation *user : value.getUsers()) {
//   //       // Ignore users outside of the block.
//   //       if (block.getParent()->findAncestorOpInRegion(*user)->getBlock() !=
//   //           &block)
//   //         continue;
//   //       SmallVector<AffineForOp, 4> loops;
//   //       getAffineForIVs(*user, &loops);
//   //       if (loops.empty())
//   //         continue;
//   //       assert(forToNodeMap.count(loops[0]) > 0 && "missing mapping");
//   //       unsigned userLoopNestId = forToNodeMap[loops[0]];
//   //       addEdge(node.id, userLoopNestId, value);
//   //     }
//   //   }
//   // }

//   // Walk memref access lists and add graph edges between dependent nodes.
//   for (auto &memrefAndList : memrefAccesses) {
//     unsigned n = memrefAndList.second.size();
//     for (unsigned i = 0; i < n; ++i) {
//       unsigned srcId = memrefAndList.second[i];
//       bool srcHasStore =
//           getNode(srcId)->getStoreOpCount(memrefAndList.first) > 0;
//       for (unsigned j = i + 1; j < n; ++j) {
//         unsigned dstId = memrefAndList.second[j];
//         bool dstHasStore =
//             getNode(dstId)->getStoreOpCount(memrefAndList.first) > 0;
//         if (srcHasStore || dstHasStore){
//           addEdge(srcId, dstId, memrefAndList.first);
//           edges.push_back({srcId, dstId, memrefAndList.first});
//         }
//       }
//     }
//   }
//   // Walk fifo access lists and add graph edges between dependent nodes.
//   for (auto &fifoAndList : fifoAccesses) {
//     unsigned n = fifoAndList.second.size();
//     for (unsigned i = 0; i < n; ++i) {
//       unsigned srcId = fifoAndList.second[i];
//       bool srcHasStore =
//           getNode(srcId)->getFifoWriteOpCount(fifoAndList.first) > 0;
//       for (unsigned j = i + 1; j < n; ++j) {
//         unsigned dstId = fifoAndList.second[j];
//         bool dstHasStore =
//             getNode(dstId)->getFifoWriteOpCount(fifoAndList.first) > 0;
//         if (srcHasStore || dstHasStore){
//           addEdge(srcId, dstId, fifoAndList.first);
//         }
//       }
//     }
//   }
//   LLVM_DEBUG(llvm::dbgs() << "DFG edges: " << edges.size() << "\n");
//   if(mergeAllocNodes){
//     SmallVector<std::pair<unsigned, unsigned>, 4> mergedNodes;
//     for(auto edge : edges){
//       LLVM_DEBUG(llvm::dbgs() << "Merging " << edge.srcId << " to " << edge.dstId << "\n");
//       // if the edge is already merged, skip
//       if(std::find(mergedNodes.begin(), mergedNodes.end(), std::make_pair(edge.srcId, edge.dstId)) != mergedNodes.end()){
//         continue;
//       }
//       mergeNodes(edge.srcId, edge.dstId);
//       // update srcId to dstId
//       for(auto &e : edges){
//         if(e.dstId == edge.srcId){
//           e.dstId = edge.dstId;
//         }
//       }
//       mergedNodes.push_back({edge.srcId, edge.dstId});
//     }
//   }
//   return true;
// }

// Merge node 'srcId' into node 'dstId' by moving all edges from 'srcId' to
Node *DFG::mergeNodes(unsigned srcId, unsigned dstId) {
  // Move all edges from 'srcId' to 'dstId'.
  SmallVector<memref::AllocOp, 2> allocOps;
  if (outEdges.count(srcId) > 0) {
    SmallVector<Edge, 2> oldOutEdges = outEdges[srcId];
    for (auto &outEdge : oldOutEdges) {
      removeEdge(srcId, outEdge.id, outEdge.value);
      if(outEdge.value.getDefiningOp<memref::AllocOp>())
        allocOps.push_back(outEdge.value.getDefiningOp<memref::AllocOp>());
      else
        addEdge(dstId, outEdge.id, outEdge.value);
    }
  }
  if (inEdges.count(srcId) > 0) {
    SmallVector<Edge, 2> oldInEdges = inEdges[srcId];
    for (auto &inEdge : oldInEdges) {
      removeEdge(inEdge.id, srcId, inEdge.value);
      if(inEdge.value.getDefiningOp<memref::AllocOp>())
        allocOps.push_back(inEdge.value.getDefiningOp<memref::AllocOp>());
      else
        addEdge(inEdge.id, dstId, inEdge.value);
    }
  }
  // Move all ops from 'srcId' to 'dstId'.
  Node *srcNode = getNode(srcId);
  Node *dstNode = getNode(dstId);
  // append allocOps to dstNode
  dstNode->allocOps.append(srcNode->allocOps.begin(), srcNode->allocOps.end());
  dstNode->allocOps.append(allocOps.begin(), allocOps.end());
  dstNode->forOps.append(srcNode->forOps.begin(), srcNode->forOps.end());
  // reverse the order of ops in dstNode
  // std::reverse(dstNode->ops.begin(), dstNode->ops.end());
  // // alloc ops need to be always first
  // std::stable_sort(dstNode->ops.begin(), dstNode->ops.end(), [](Operation *a, Operation *b){
  //   return isa<memref::AllocOp>(a) && !isa<memref::AllocOp>(b);
  // });
  // Remove 'srcId' from the graph.
  removeNode(srcId);
  return dstNode;
}
static void printVector(SmallVectorImpl<unsigned> &v){
  for(auto i : v){
    LLVM_DEBUG(llvm::dbgs() << i << " ");
  }
  LLVM_DEBUG(llvm::dbgs() << "\n");
}
static void printVector(SmallVectorImpl<int64_t> &v){
  for(auto i : v){
    LLVM_DEBUG(llvm::dbgs() << i << " ");
  }
  LLVM_DEBUG(llvm::dbgs() << "\n");
}
// static LogicalResult getPermMap(
//   FlatAffineRelation baseRel, // relation to be matched
//   FlatAffineRelation permRel, // relation to be permuted
//   SmallVector<unsigned> &permMap
// ){
//   // LLVM_DEBUG(llvm::dbgs() << "domain dims " << storeAccess.getNumDomainDims() << "\n");
//   // LLVM_DEBUG(llvm::dbgs() << "range dims " << storeAccess.getNumRangeDims() << "\n");
//   // LLVM_DEBUG(llvm::dbgs() << "domain dims " << loadAccess.getNumDomainDims() << "\n");
//   // LLVM_DEBUG(llvm::dbgs() << "range dims " << loadAccess.getNumRangeDims() << "\n");
//   // LLVM_DEBUG(llvm::dbgs() << "loadRel.getNumEqualities() " << loadRel.getNumEqualities() << "\n");
//   // LLVM_DEBUG(llvm::dbgs() << "loadRel.getNumDomainVars() " << loadRel.getNumDomainDims() << "\n");

//   // for(unsigned col = 0; col < rel.getNumDomainDims(); col++){
//   //   for(unsigned row = 0; row < rel.getNumEqualities(); row++){
//   //     if(rel.atEq64(row, col)){
//   //       permMap.push_back(row);
//   //     }
//   //   }
//   // }
//   for(unsigned permCol = 0; permCol < permRel.getNumDomainDims(); permCol++){
//     for(unsigned permRow = 0; permRow < permRel.getNumEqualities(); permRow++){
//       if(permRel.atEq64(permRow, permCol)){
//         for(unsigned baseCol = 0; baseCol < baseRel.getNumDomainDims(); baseCol++){
//           if(baseRel.atEq64(permRow, baseCol)){
//             permMap.push_back(baseCol);
//           }
//         }
//       }
//     }
//   }

//   // for(unsigned i = 0; i < storeRel.getNumEqualities(); i++){
//   //   auto storeEq = storeRel.getEquality64(i);
//   //   for(unsigned j = 0; j < loadRel.getNumEqualities(); j++){
//   //     auto loadEq = loadRel.getEquality64(j);
//   //     printVector(storeEq);
//   //     printVector(loadEq);
//   //     // compare storeEq and loadEq, where storeEq and loadEq are SmallVector<int64_t, 4>
//   //     auto eq = true;
//   //     for(unsigned k = 0; k < storeEq.size(); k++){
//   //       if(storeEq[k] != loadEq[k]){
//   //         eq = false;
//   //         break;
//   //       }
//   //     }
//   //     if(eq){
//   //       permMap.push_back(j);
//   //     }
//   //   }
//   // }
//   // for(auto idx : permMap){
//   //   LLVM_DEBUG(llvm::dbgs() << idx << " ");
//   // }
//   // storeAccess.dump();
// }

// induction variable struct
struct IV{
  Value value;
  int64_t lb;
  int64_t ub;
  int64_t tripCount;
};

// static FlatAffineRelation createIVsIndexRelation(Operation* operation){
//   AffineStoreOp storeOp;
//   AffineLoadOp loadOp;
//   if(!isa<AffineStoreOp, AffineLoadOp>(operation)){
//     assert(false && "operation is not a store or load");
//   }else{
//     if(isa<AffineStoreOp>(operation)){
//       storeOp = cast<AffineStoreOp>(operation);
//     }
//     if(isa<AffineLoadOp>(operation)){
//       loadOp = cast<AffineLoadOp>(operation);
//     }
//   }
//   AffineLoopBand band;
//   auto innerMostForOp = operation->getParentOfType<AffineForOp>();
//   auto outerMostForOp = getLoopBandFromInnermost(innerMostForOp, band);
//   if(!isPerfectlyNested(band)) // TODO: add more checks
//     assert(false && "band is not perfectly nested");

//   SmallVector<IV, 4> ivs;
//   for(auto forOp : band){
//     if(!forOp.hasConstantBounds())
//       assert(false && "bounds are not constant");
//     auto lb = forOp.getConstantLowerBound();
//     if(lb != 0)
//       assert(false && "lower bound is not 0");
//     auto ub = forOp.getConstantUpperBound();
//     auto tripCount = ub - lb;
//     auto step = forOp.getStep();
//     auto iv = IV{forOp.getInductionVar(), lb, ub, tripCount};
//     ivs.push_back(iv);
//   }
// }

static FlatAffineRelation createTimeIVsRelation(Operation* operation){
  // Create set corresponding to domain of access.
  AffineStoreOp storeOp;
  AffineLoadOp loadOp;
  if(!isa<AffineStoreOp, AffineLoadOp>(operation)){
    assert(false && "operation is not a store or load");
  }else{
    if(isa<AffineStoreOp>(operation)){
      storeOp = cast<AffineStoreOp>(operation);
    }
    if(isa<AffineLoadOp>(operation)){
      loadOp = cast<AffineLoadOp>(operation);
    }
  }
  AffineLoopBand band;
  auto innerMostForOp = operation->getParentOfType<AffineForOp>();
  auto outerMostForOp = getLoopBandFromInnermost(innerMostForOp, band);
  if(!isPerfectlyNested(band)) // TODO: add more checks
    assert(false && "band is not perfectly nested");

  SmallVector<IV, 4> ivs;
  for(auto forOp : band){
    if(!forOp.hasConstantBounds())
      assert(false && "bounds are not constant");
    auto lb = forOp.getConstantLowerBound();
    if(lb != 0)
      assert(false && "lower bound is not 0");
    auto ub = forOp.getConstantUpperBound();
    auto tripCount = ub - lb;
    auto step = forOp.getStep();
    auto iv = IV{forOp.getInductionVar(), lb, ub, tripCount};
    ivs.push_back(iv);
  }
  // create relation
  // creating timestamp equation
  SmallVector<int64_t, 4> timeEq = {-1};
  auto idx = 0;
  for(auto iv : ivs){
    auto var = 1;
    // iterate over ivs starting from current iv to the end of ivs
    for(auto i = idx + 1; i < ivs.size(); i++){
      var *= ivs[i].tripCount;
    }
    idx++;
    timeEq.push_back(var);
  }
  timeEq.push_back(0);
  SmallVector<SmallVector<int64_t, 4>, 4> inequalities;

  // creating constraints
  auto colIdx = 0;
  for(auto iv : ivs){
    SmallVector<int64_t, 4> lbConstraint = {0}; // for time col
    SmallVector<int64_t, 4> ubConstraint = {0}; // for time col
    for(auto i = 0; i < ivs.size(); i++){
      if(i == colIdx){
        lbConstraint.push_back(1);
        ubConstraint.push_back(-1);
      }else{
        lbConstraint.push_back(0);
        ubConstraint.push_back(0);
      }
    }
    lbConstraint.push_back(iv.lb);
    ubConstraint.push_back(iv.ub-1);
    inequalities.push_back(lbConstraint);
    inequalities.push_back(ubConstraint);
    colIdx++;
  }
  // time bounds
  auto bandTripCount = 1;
  for(auto iv : ivs){
    bandTripCount *= iv.tripCount;
  }
  SmallVector<int64_t, 4> lbConstraint = {1}; // for time col
  SmallVector<int64_t, 4> ubConstraint = {-1}; // for time col
  for(auto i = 0; i < ivs.size(); i++){
    lbConstraint.push_back(0);
    ubConstraint.push_back(0);
  }
  lbConstraint.push_back(0);
  ubConstraint.push_back(bandTripCount-1);
  inequalities.push_back(lbConstraint);
  inequalities.push_back(ubConstraint);

  // create relation
  FlatAffineRelation relation(
    (unsigned) inequalities.size(), // numInequalities
    (unsigned) timeEq.size(), // numEqualities
    (unsigned) 1 + ivs.size() + 1, // cols (time + ivs + const)
    (unsigned) 1, // domain size (time)
    (unsigned) ivs.size(), // range size (ivs)
    (unsigned) 0, // numSymbols
    (unsigned) 0 // numLocals
  );
  relation.setSpace(presburger::PresburgerSpace::getRelationSpace(1,(unsigned) ivs.size(),0,0));
  
  colIdx = 1;
  for(auto iv : ivs){
    relation.setValue(colIdx, ivs[colIdx-1].value);
    colIdx++;
  }
  relation.addEquality(timeEq);
  for(auto ineq : inequalities){
    relation.addInequality(ineq);
  }
  return relation;
}

static FlatAffineRelation createRelevantTimeIVsRelation(Operation* operation){
  // Create set corresponding to domain of access.
  AffineStoreOp storeOp;
  AffineLoadOp loadOp;
  if(!isa<AffineStoreOp, AffineLoadOp>(operation)){
    assert(false && "operation is not a store or load");
  }else{
    if(isa<AffineStoreOp>(operation)){
      storeOp = cast<AffineStoreOp>(operation);
    }
    if(isa<AffineLoadOp>(operation)){
      loadOp = cast<AffineLoadOp>(operation);
    }
  }
  AffineLoopBand band;
  auto innerMostForOp = operation->getParentOfType<AffineForOp>();
  auto outerMostForOp = getLoopBandFromInnermost(innerMostForOp, band);
  if(!isPerfectlyNested(band)) // TODO: add more checks
    assert(false && "band is not perfectly nested");

  SmallVector<IV, 4> ivs;
  for(auto forOp : band){
    // check if the iv is relevant
    auto iter = forOp.getInductionVar();
    auto isRelevant = false;
    for(auto op : iter.getUsers()){
      if(op == operation){
        isRelevant = true;
        break;
      }
    }
    if(!isRelevant)
      continue;
    if(!forOp.hasConstantBounds())
      assert(false && "bounds are not constant");
    auto lb = forOp.getConstantLowerBound();
    if(lb != 0)
      assert(false && "lower bound is not 0");
    auto ub = forOp.getConstantUpperBound();
    auto tripCount = ub - lb;
    auto step = forOp.getStep();
    auto iv = IV{forOp.getInductionVar(), lb, ub, tripCount};
    ivs.push_back(iv);
  }
  // create relation
  // creating timestamp equation
  SmallVector<int64_t, 4> timeEq = {-1};
  auto idx = 0;
  for(auto iv : ivs){
    auto var = 1;
    // iterate over ivs starting from current iv to the end of ivs
    for(auto i = idx + 1; i < ivs.size(); i++){
      var *= ivs[i].tripCount;
    }
    idx++;
    timeEq.push_back(var);
  }
  timeEq.push_back(0);
  SmallVector<SmallVector<int64_t, 4>, 4> inequalities;

  // creating constraints
  auto colIdx = 0;
  for(auto iv : ivs){
    SmallVector<int64_t, 4> lbConstraint = {0}; // for time col
    SmallVector<int64_t, 4> ubConstraint = {0}; // for time col
    for(auto i = 0; i < ivs.size(); i++){
      if(i == colIdx){
        lbConstraint.push_back(1);
        ubConstraint.push_back(-1);
      }else{
        lbConstraint.push_back(0);
        ubConstraint.push_back(0);
      }
    }
    lbConstraint.push_back(iv.lb);
    ubConstraint.push_back(iv.ub-1);
    inequalities.push_back(lbConstraint);
    inequalities.push_back(ubConstraint);
    colIdx++;
  }
  // time bounds
  auto bandTripCount = 1;
  for(auto iv : ivs){
    bandTripCount *= iv.tripCount;
  }
  SmallVector<int64_t, 4> lbConstraint = {1}; // for time col
  SmallVector<int64_t, 4> ubConstraint = {-1}; // for time col
  for(auto i = 0; i < ivs.size(); i++){
    lbConstraint.push_back(0);
    ubConstraint.push_back(0);
  }
  lbConstraint.push_back(0);
  ubConstraint.push_back(bandTripCount-1);
  inequalities.push_back(lbConstraint);
  inequalities.push_back(ubConstraint);

  // create relation
  FlatAffineRelation relation(
    (unsigned) inequalities.size(), // numInequalities
    (unsigned) timeEq.size(), // numEqualities
    (unsigned) 1 + ivs.size() + 1, // cols (time + ivs + const)
    (unsigned) 1, // domain size (time)
    (unsigned) ivs.size(), // range size (ivs)
    (unsigned) 0, // numSymbols
    (unsigned) 0 // numLocals
  );
  relation.setSpace(presburger::PresburgerSpace::getRelationSpace(1,(unsigned) ivs.size(),0,0));
  
  colIdx = 1;
  for(auto iv : ivs){
    relation.setValue(colIdx, ivs[colIdx-1].value);
    colIdx++;
  }
  relation.addEquality(timeEq);
  for(auto ineq : inequalities){
    relation.addInequality(ineq);
  }
  return relation;
}

void DFG::topologicalSortUtil(unsigned currId, std::set<unsigned> &visited, std::stack<unsigned> &stack){
  // mark currId as visited
  visited.insert(currId);
  // for each edge from currId
  for(auto edge : outEdges[currId]){
    // if edge.dstId is not visited
    if(visited.find(edge.id) == visited.end()){
      // call recursive DFS
      topologicalSortUtil(edge.id, visited, stack);
    }
  }
  // push currId to stack
  stack.push(currId);
}

SmallVector<unsigned> DFG::topologicalSort(){
  // visited
  std::set<unsigned> visited;
  // stack
  std::stack<unsigned> stack;
  // for each node
  for(auto node : nodes){
    // if node is not visited
    if(visited.find(node.first) == visited.end()){
      // call recursive DFS
      topologicalSortUtil(node.first, visited, stack);
    }
  }
  // create vector from stack
  SmallVector<unsigned> sorted;
  while(!stack.empty()){
    sorted.push_back(stack.top());
    stack.pop();
  }
  return sorted;
}

static std::string recursiveNestedMax(SmallVector<std::string> &terms){
  if(terms.size() == 1){
    return terms[0];
  }else{
    auto first = terms[0];
    terms.erase(terms.begin());
    auto second = recursiveNestedMax(terms);
    return " m.max2(" + first + ", " + second + ")";
    // return " \\\nm.max2(" + first + ", " + second + ")";
  }
}

static presburger::SymbolicLexOpt findSymbolicIntegerLexOpt(const presburger::PresburgerRelation &rel,
                                                bool isMin) {
  presburger::SymbolicLexOpt result(rel.getSpace());
  presburger::PWMAFunction &lexopt = result.lexopt;
  presburger::PresburgerSet &unboundedDomain = result.unboundedDomain;
  for (const presburger::IntegerRelation &cs : rel.getAllDisjuncts()) {
    presburger::SymbolicLexOpt s(rel.getSpace());
    if (isMin) {
      s = cs.findSymbolicIntegerLexMin();
      lexopt = lexopt.unionLexMin(s.lexopt);
    } else {
      s = cs.findSymbolicIntegerLexMax();
      lexopt = lexopt.unionLexMax(s.lexopt);
    }
    unboundedDomain = unboundedDomain.intersect(s.unboundedDomain);
  }
  return result;
}

static uint64_t getAccessLexOpt(presburger::IntegerRelation rel, SmallVectorImpl<int64_t>& elementIndices, bool isMin = false){
  // presburger::PresburgerRelation newRel = rel.computeReprWithOnlyDivLocals();
  rel.inverse();
  // presburger::SymbolicLexOpt lexOpt = findSymbolicIntegerLexOpt(rel, isMin);
  auto lexOpt = rel.findSymbolicIntegerLexMin();
  // LLVM_DEBUG(
  //   llvm::dbgs() << "Rel.getNumDomainVars(): " << rel.getNumDomainVars() << "\n";
  //   llvm::dbgs() << "Rel.getNumRangeVars(): " << rel.getNumSymbolVars() << "\n";
  //   llvm::dbgs() << "elementIndices.size(): " << elementIndices.size() << "\n";
  // );
  std::optional<SmallVector<presburger::MPInt, 8>> output = lexOpt.lexopt.valueAt(elementIndices);
  if(output.has_value()){
    if(output.value().size() == 1){
      presburger::MPInt val = output.value()[0];
      return int64FromMPInt(val);
    } else {
      assert(false && "output.value().size() != 1");
    }
  }else{
    assert(false && "could't find lexmax");
  }
}

static SmallVector<unsigned> inversePerm(const SmallVectorImpl<unsigned> &permutation) {
  SmallVector<unsigned> inversePermutation(permutation.size());
  for (int i = 0; i < permutation.size(); ++i) {
    inversePermutation[permutation[i]] = i;
  }
  return inversePermutation;
}

// static int64_t getOverlap(
//   DFG::Edge edge, 
//   DFG::NodeInfo srcNodeInfo,
//   DFG::NodeInfo dstNodeInfo,
//   unsigned srcII,
//   unsigned dstII,
//   uint64_t srcTripCount,
//   uint64_t dstTripCount,
//   bool optimizeOverlap = true
// ){
//   // LLVM_DEBUG(
//   //   llvm::dbgs() << "Checking overlap between: " << edge.srcOp << " and " << edge.dstOp << "\n";
//   // );
//   // if(srcNodeInfo.stores[edge.srcOp].timeIndexRelation == nullptr){
//   //   assert(false && "srcNodeInfo.stores[edge.srcOp].timeIndexRelation == nullptr");
//   // }
//   // if(dstNodeInfo.loads[edge.dstOp].timeIndexRelation == nullptr){
//   //   assert(false && "dstNodeInfo.loads[edge.dstOp].timeIndexRelation == nullptr");
//   // }
//   // if(srcNodeInfo.stores[edge.srcOp].timeIndexRelevantRelation == nullptr){
//   //   assert(false && "srcNodeInfo.stores[edge.srcOp].timeIndexRelevantRelation == nullptr");
//   // }
//   // if(dstNodeInfo.loads[edge.dstOp].timeIndexRelevantRelation == nullptr){
//   //   assert(false && "dstNodeInfo.loads[edge.dstOp].timeIndexRelevantRelation == nullptr");
//   // }

//   // auto srcRel = optimizeOverlap? srcNodeInfo.stores[edge.srcOp].timeIndexRelevantRelation : 
//   //   srcNodeInfo.stores[edge.srcOp].timeIndexRelation;
//   auto srcAccessMap = optimizeOverlap? srcNodeInfo.stores[edge.srcOp].accessMap : 
//     srcNodeInfo.stores[edge.srcOp].accessMap;
//   auto srcFirstElementTime = srcNodeInfo.stores[edge.srcOp].firstElementTime*srcII;
//   auto srcLastElementTime = srcNodeInfo.stores[edge.srcOp].lastElementTime*srcII;
  
//   // auto dstRel = optimizeOverlap? dstNodeInfo.loads[edge.dstOp].timeIndexRelevantRelation : 
//   //   dstNodeInfo.loads[edge.dstOp].timeIndexRelation;
//   auto dstAccessMap = optimizeOverlap? dstNodeInfo.loads[edge.dstOp].accessMap : 
//     dstNodeInfo.loads[edge.dstOp].accessMap;
//   auto dstFirstElementTime = dstNodeInfo.loads[edge.dstOp].firstElementTime*dstII;
//   auto dstLastElementTime = dstNodeInfo.loads[edge.dstOp].lastElementTime*dstII;
//   auto overlap = -1;
//   // srcRel->dump();
//   // dstRel->dump();
//   if(dstAccessMap == srcAccessMap){
//     auto storeDiff = (srcLastElementTime - srcFirstElementTime);
//     auto loadDiff = (dstLastElementTime - dstFirstElementTime);
//     auto isSrcReduction = (storeDiff + srcII) != srcTripCount*srcII;
//     auto isDstReduction = (loadDiff + dstII) != dstTripCount*dstII;
//     // LLVM_DEBUG(
//     //   llvm::dbgs() << "storeDiff: " << storeDiff << "\n";
//     //   llvm::dbgs() << "loadDiff: " << loadDiff << "\n";
//     //   llvm::dbgs() << "srcTripCount: " << srcTripCount << "\n";
//     //   llvm::dbgs() << "dstTripCount: " << dstTripCount << "\n";
//     //   llvm::dbgs() << "isSrcReduction: " << isSrcReduction << "\n";
//     //   llvm::dbgs() << "isDstReduction: " << isDstReduction << "\n\n";
//     // );
//     // overlap = srcII*srcTripCount - srcFirstElementTime;
//     // if(srcTripCount < dstTripCount){
//     //   overlap = srcTripCount - srcFirstElementTime;
//     // } else if(srcTripCount > dstTripCount){
//     //   overlap = srcTripCount - srcFirstElementTime;
//     // } else {
//     auto latency = loadDiff >= storeDiff? 
//       dstTripCount + srcFirstElementTime : 
//       (dstTripCount - dstLastElementTime) + srcTripCount - 1;
//     overlap = srcTripCount + dstTripCount - latency;  
//     // if        (storeDiff < loadDiff && !isSrcReduction && isDstReduction) {
//     //   overlap = dstLastElementTime - dstFirstElementTime; // the problem is here
//     // } else if (storeDiff < loadDiff && isSrcReduction && isDstReduction){
//     //   overlap = srcTripCount - srcFirstElementTime;
//     // } else if (storeDiff > loadDiff && isSrcReduction && !isDstReduction){
//     //   overlap = dstLastElementTime + 1;
//     // } else if (storeDiff > loadDiff && isSrcReduction && isDstReduction){
//     //   overlap = dstLastElementTime + 1;
//     // } else if (storeDiff == loadDiff && isSrcReduction && !isDstReduction){
//     //   overlap = dstLastElementTime + 1;
//     // } else if (storeDiff == loadDiff && !isSrcReduction && !isDstReduction){
//     //   overlap = dstLastElementTime + 1;
//     // } else if (storeDiff == loadDiff && !isSrcReduction && isDstReduction){
//     //   overlap = dstLastElementTime + 1;
//     // } else if (storeDiff == loadDiff && isSrcReduction && isDstReduction){
//     //   overlap = dstLastElementTime + 1;
//     // } else {
//     //   LLVM_DEBUG(
//     //     llvm::dbgs() << "storeDiff: " << storeDiff << "\n";
//     //     llvm::dbgs() << "loadDiff: " << loadDiff << "\n";
//     //     llvm::dbgs() << "srcTripCount: " << srcTripCount << "\n";
//     //     llvm::dbgs() << "dstTripCount: " << dstTripCount << "\n";
//     //     llvm::dbgs() << "isSrcReduction: " << isSrcReduction << "\n";
//     //     llvm::dbgs() << "isDstReduction: " << isDstReduction << "\n\n";
//     //   );
//     //   assert(false && "getOverlap: unexpected case");
//     // }
//     // }
//   } else {
//     overlap = -1;
//   }
//   return overlap;
// }

// static int64_t getOverlap(
//   DFG::Edge edge, 
//   SmallVectorImpl<unsigned>& srcPerm, 
//   SmallVectorImpl<unsigned>& dstPerm,
//   unsigned srcII,
//   unsigned dstII
// ){
//   // SmallVector<AffineStoreOp> storeOps;
//   // SmallVector<AffineLoadOp> loadOps;
//   // for(auto user : edge.getUsers()){
//   //   if(auto storeOp = dyn_cast<AffineStoreOp>(user)){
//   //     storeOps.push_back(storeOp);
//   //   } else if(auto loadOp = dyn_cast<AffineLoadOp>(user)){
//   //     loadOps.push_back(loadOp);
//   //   }
//   // }
//   // if(storeOps.size() == 1 || loadOps.size() == 1){
//     // get store/load info
//     AffineStoreOp storeOp = edge.srcOp;
//     AffineLoadOp loadOp = edge.dstOp;
    
//     // auto shape = loadOp.getMemRefType().getShape();
//     // SmallVector<int64_t> firstElementIndices;
//     // for(auto i = 0; i < shape.size(); i++){
//     //   firstElementIndices.push_back(0);
//     // }
//     // SmallVector<int64_t> lastElementIndices;
//     // for(auto i = 0; i < shape.size(); i++){
//     //   lastElementIndices.push_back(shape[i] - 1);
//     // }
//     // // get loop bands of store/load before permutation
//     // AffineLoopBand storeBand;
//     // auto storeInnerLoop = storeOp.getOperation()->getParentOfType<AffineForOp>();
//     // getLoopBandFromInnermost(storeInnerLoop, storeBand);

//     // AffineLoopBand loadBand;
//     // auto loadInnerLoop = loadOp.getOperation()->getParentOfType<AffineForOp>();
//     // getLoopBandFromInnermost(loadInnerLoop, loadBand);

//     // auto storeTripCount = getLoopNestIterations(storeBand)*srcII;
//     // auto loadTripCount = getLoopNestIterations(loadBand)*dstII;

//     // // permute store loops
//     // auto newStoreRoot = storeBand[permuteLoops(storeBand, srcPerm)];
//     // storeBand.clear();
//     // getLoopBandFromOutermost(newStoreRoot, storeBand);

//     // // permute load loops
//     // auto newLoadRoot = loadBand[permuteLoops(loadBand, dstPerm)];
//     // loadBand.clear();
//     // getLoopBandFromOutermost(newLoadRoot, loadBand);

//     auto storeInfoPostPerm = AccessInfo(storeOp);
//     auto loadInfoPostPerm = AccessInfo(loadOp);
//     if(failed(storeInfoPostPerm.getAllInfo(storeOp.getContext()))){
//       assert(false && "storeInfoPostPerm.getAllInfo failed");
//     }
//     if(failed(loadInfoPostPerm.getAllInfo(loadOp.getContext()))){
//       assert(false && "loadInfoPostPerm.getAllInfo failed");
//     }
//     auto storeRel = storeInfoPostPerm.getTimeIndexRelevantRelation();
//     auto loadRel = loadInfoPostPerm.getTimeIndexRelevantRelation();
//     auto overlap = -1;
//     if(storeRel.isEqual(loadRel)){
//     //   auto storeTimeIdxRel = storeInfoPostPerm.getTimeIndexRelation();
//     //   auto loadTimeIdxRel = loadInfoPostPerm.getTimeIndexRelation();
//     //   auto storeFirstElement = getAccessLexOpt(storeTimeIdxRel, firstElementIndices, true)*srcII;
//     //   auto storeLastElement = getAccessLexOpt(storeTimeIdxRel, lastElementIndices, true)*srcII;
//     //   auto storeDiff = (storeLastElement - storeFirstElement);
//     //   auto loadFirstElement = getAccessLexOpt(loadTimeIdxRel, firstElementIndices, true)*dstII;
//     //   auto loadLastElement = getAccessLexOpt(loadTimeIdxRel, lastElementIndices, true)*dstII;
//     //   auto loadDiff = (loadLastElement - loadFirstElement);
//     //   auto latency = loadDiff >= storeDiff? 
//     //     loadTripCount + storeFirstElement : 
//     //     (loadTripCount - loadLastElement) + storeTripCount - 1;
//     //   overlap = storeTripCount + loadTripCount - latency;
//     //   LLVM_DEBUG(
//     //     llvm::dbgs() << "overlap: " << overlap << "\n";
//     //   );
//       overlap = 1;
//     } else {
//       overlap = 0;
//     }
//     // undo the permutation
//     // newLoadRoot = loadBand[permuteLoops(loadBand, inversePerm(dstPerm))];
//     // loadBand.clear();
//     // getLoopBandFromOutermost(newLoadRoot, loadBand);
//     // // undo the permutation
//     // newStoreRoot = storeBand[permuteLoops(storeBand, inversePerm(srcPerm))];
//     // storeBand.clear();
//     // getLoopBandFromOutermost(newStoreRoot, storeBand);
//     return overlap;
//   // }else{
//   //   assert(false && "getOverlap: edge has more than one store or load op");
//   // }
// }


/// Returns true if `v` is allocated locally to `enclosingOp` -- i.e., it is
/// allocated by an operation nested within `enclosingOp`.
static bool isLocallyDefined(Value v, Operation *enclosingOp) {
  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return false;

  if (hasSingleEffect<MemoryEffects::Allocate>(defOp, v) &&
      enclosingOp->isProperAncestor(defOp))
    return true;

  // Aliasing ops.
  auto viewOp = dyn_cast<ViewLikeOpInterface>(defOp);
  return viewOp && isLocallyDefined(viewOp.getViewSource(), enclosingOp);
}

// Returns a result string which represents the direction vector (if there was
// a dependence), returns the string "false" otherwise.
static std::string
getDirectionVectorStr(bool ret, unsigned numCommonLoops, unsigned loopNestDepth,
                      ArrayRef<DependenceComponent> dependenceComponents) {
  if (!ret)
    return "false";
  if (dependenceComponents.empty() || loopNestDepth > numCommonLoops)
    return "true";
  std::string result;
  for (const auto &dependenceComponent : dependenceComponents) {
    std::string lbStr = "-inf";
    if (dependenceComponent.lb.has_value() &&
        *dependenceComponent.lb != std::numeric_limits<int64_t>::min())
      lbStr = std::to_string(*dependenceComponent.lb);

    std::string ubStr = "+inf";
    if (dependenceComponent.ub.has_value() &&
        *dependenceComponent.ub != std::numeric_limits<int64_t>::max())
      ubStr = std::to_string(*dependenceComponent.ub);

    result += "[" + lbStr + ", " + ubStr + "]";
  }
  return result;
}

// For each access in 'loadsAndStores', runs a dependence check between this
// "source" access and all subsequent "destination" accesses in
// 'loadsAndStores'. Emits the result of the dependence check as a note with
// the source access.
static void checkDependences(ArrayRef<Operation *> loadsAndStores) {
  for (unsigned i = 0, e = loadsAndStores.size(); i < e; ++i) {
    auto *srcOpInst = loadsAndStores[i];
    affine::MemRefAccess srcAccess(srcOpInst);
    for (unsigned j = 0; j < e; ++j) {
      auto *dstOpInst = loadsAndStores[j];
      affine::MemRefAccess dstAccess(dstOpInst);

      unsigned numCommonLoops =
          getNumCommonSurroundingLoops(*srcOpInst, *dstOpInst);
      for (unsigned d = 1; d <= numCommonLoops + 1; ++d) {
        SmallVector<DependenceComponent, 2> dependenceComponents;
        DependenceResult result = checkMemrefAccessDependence(
            srcAccess, dstAccess, d, /*dependenceConstraints=*/nullptr,
            &dependenceComponents);
        if (result.value == DependenceResult::Failure) {
          srcOpInst->emitError("dependence check failed");
        } else {
          bool ret = hasDependence(result);
          // TODO: Print dependence type (i.e. RAW, etc) and print
          // distance vectors as: ([2, 3], [0, 10]). Also, shorten distance
          // vectors from ([1, 1], [3, 3]) to (1, 3).
          srcOpInst->emitRemark("dependence from ")
              << i << " to " << j << " at depth " << d << " = "
              << getDirectionVectorStr(ret, numCommonLoops, d,
                                       dependenceComponents);
        }
      }
    }
  }
}

static OperationName getOpsBetweenLoadAndStore(
  AffineLoadOp loadOp, AffineStoreOp storeOp) {
  auto currValue = loadOp.getResult();
  SmallVector<Operation *, 8> opsBetweenLoadAndStore;
  for(auto user : currValue.getUsers()){
    // user->dump();
    opsBetweenLoadAndStore.push_back(user);
  }
  if(opsBetweenLoadAndStore.size() == 1){
    return opsBetweenLoadAndStore[0]->getName();
  }else{
    assert(false && "opsBetweenLoadAndStore.size() != 1");
  }
}

static bool isReductionLoop(AffineForOp forOp){
  // Collect all load and store ops in loop nest rooted at 'forOp'.
  SmallVector<Operation *, 8> loadOps;
  SmallVector<Operation *, 8> storeOps;
  auto walkResult = forOp.walk([&](Operation *op) -> WalkResult {
    if (auto readOp = dyn_cast<AffineReadOpInterface>(op)) {
      // Memrefs that are allocated inside `forOp` need not be considered.
      if (!isLocallyDefined(readOp.getMemRef(), forOp))
        loadOps.push_back(op);
    } else if (auto writeOp = dyn_cast<AffineWriteOpInterface>(op)) {
      // Filter out stores the same way as above.
      if (!isLocallyDefined(writeOp.getMemRef(), forOp))
        storeOps.push_back(op);
    } else if (!isa<AffineForOp, AffineYieldOp, AffineIfOp>(op) &&
              !hasSingleEffect<MemoryEffects::Allocate>(op) &&
              !isMemoryEffectFree(op)) {
      // Alloc-like ops inside `forOp` are fine (they don't impact parallelism)
      // as long as they don't escape the loop (which has been checked above).
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  // Stop early if the loop has unknown ops with side effects.
  if (walkResult.wasInterrupted())
    return false;

  for(auto *storeOp : storeOps){
    auto storeMemref = cast<AffineWriteOpInterface>(storeOp).getMemRef();
    auto storeBlock = storeOp->getBlock();
    for(auto *loadOp : loadOps){
      auto loadMemref = cast<AffineReadOpInterface>(loadOp).getMemRef();
      auto loadBlock = loadOp->getBlock();
      if(storeMemref == loadMemref && storeBlock == loadBlock){
        return true;
      }
    }
  }
  return false;
}
static uint64_t getForLoopII(AffineForOp forOp){
  // Collect all load and store ops in loop nest rooted at 'forOp'.
  SmallVector<AffineStoreOp, 1> storeOps;
  SmallVector<AffineLoadOp, 1> loadOps;
  auto walkResult = forOp.walk([&](Operation *op) -> WalkResult {
    if (auto readOp = dyn_cast<AffineReadOpInterface>(op)) {
      loadOps.push_back(cast<AffineLoadOp>(op));
    } else if (auto writeOp = dyn_cast<AffineWriteOpInterface>(op)) {
      storeOps.push_back(cast<AffineStoreOp>(op));
    } 
    return WalkResult::advance();
  });
  auto II = 1;
  for(auto storeOp : storeOps){
    auto storeMemref = storeOp.getMemRef();
    auto storeBlock = storeOp->getBlock();
    auto storeAffineMap = storeOp.getAffineMap();
    auto storeMapOperands = storeOp.getMapOperands();
    for(auto loadOp : loadOps){
      auto loadMemref = loadOp.getMemRef();
      auto loadBlock = loadOp->getBlock();
      auto loadAffineMap = loadOp.getAffineMap();
      auto loadMapOperands = loadOp.getMapOperands();
      // to make sure the store and load are in the same block and have the same memref
      if(storeMemref == loadMemref && storeBlock == loadBlock){
        auto innerMostLoop = storeOp->getParentOfType<AffineForOp>();
        auto innerMostIV = innerMostLoop.getInductionVar();
        SmallVector<Operation *, 8> opsBetweenLoadAndStore;
        auto opName = getOpsBetweenLoadAndStore(cast<AffineLoadOp>(loadOp), cast<AffineStoreOp>(storeOp));
        auto latency = (opName.getStringRef() == "arith.addf") ? 4 : 2;
        II = latency;
        for(auto operand : storeMapOperands){
          if(operand == innerMostIV){
            II = 1;
          }
        }
      }
    }
  }
  if(II == -1) {
    assert(false && "II == -1");
  }
  return II;
}
// static uint64_t getForLoopII(AffineForOp forOp){
//   // Collect all load and store ops in loop nest rooted at 'forOp'.
//   SmallVector<Operation *, 8> loadOps;
//   SmallVector<Operation *, 8> storeOps;
//   auto walkResult = forOp.walk([&](Operation *op) -> WalkResult {
//     if (auto readOp = dyn_cast<AffineReadOpInterface>(op)) {
//       // Memrefs that are allocated inside `forOp` need not be considered.
//       if (!isLocallyDefined(readOp.getMemRef(), forOp))
//         loadOps.push_back(op);
//     } else if (auto writeOp = dyn_cast<AffineWriteOpInterface>(op)) {
//       // Filter out stores the same way as above.
//       if (!isLocallyDefined(writeOp.getMemRef(), forOp))
//         storeOps.push_back(op);
//     } else if (!isa<AffineForOp, AffineYieldOp, AffineIfOp>(op) &&
//               !hasSingleEffect<MemoryEffects::Allocate>(op) &&
//               !isMemoryEffectFree(op)) {
//       // Alloc-like ops inside `forOp` are fine (they don't impact parallelism)
//       // as long as they don't escape the loop (which has been checked above).
//       return WalkResult::interrupt();
//     }

//     return WalkResult::advance();
//   });

//   // Stop early if the loop has unknown ops with side effects.
//   if (walkResult.wasInterrupted())
//     return false;

//   for(auto *storeOp : storeOps){
//     auto storeMemref = cast<AffineWriteOpInterface>(storeOp).getMemRef();
//     auto storeBlock = storeOp->getBlock();
//     for(auto *loadOp : loadOps){
//       auto loadMemref = cast<AffineReadOpInterface>(loadOp).getMemRef();
//       auto loadBlock = loadOp->getBlock();
//       if(storeMemref == loadMemref && storeBlock == loadBlock){
//         SmallVector<Operation *, 8> opsBetweenLoadAndStore;
//         auto opName = getOpsBetweenLoadAndStore(cast<AffineLoadOp>(loadOp), cast<AffineStoreOp>(storeOp));
//         auto II = (opName.getStringRef() == "arith.addf") ? 4 : 2;
//         // LLVM_DEBUG(
//         //   llvm::dbgs() << "opsBetweenLoadAndStore: " << opName << "\n";
//         // );
//         auto storeInfo = AccessInfo(storeOp);
//         auto loadInfo = AccessInfo(loadOp);
//         if(failed(storeInfo.getAllInfo(storeOp->getContext()))){
//           assert(false && "storeInfo.getAllInfo failed");
//         }
//         if(failed(loadInfo.getAllInfo(loadOp->getContext()))){
//           assert(false && "loadInfo.getAllInfo failed");
//         }
//         if(storeInfo.relation.isEqual(loadInfo.relation)){
//           // LLVM_DEBUG(
//           //   llvm::dbgs() << "store and load are equal\n";
//           // );
//           auto reductionLoopIVs = storeInfo.irrelevantIVs;
//           // if reductionloops are inner most, then II = 4, else II = 1
//           auto innerMostLoop = storeOp->getParentOfType<AffineForOp>();
//           auto innerMostLoopIV = innerMostLoop.getInductionVar();
//           for(auto redIV : reductionLoopIVs){
//             if(redIV == innerMostLoopIV){
//               return II;
//             }
//           }
//         } else {
//           LLVM_DEBUG(
//             llvm::dbgs() << "Error: store and load are not equal\n";
//           );
//         }
//       }
//     }
//   }
//   return 1;
// }

static uint64_t getForLoopII(AffineForOp forOp, SmallVectorImpl<unsigned>&permMap){
  // permute loop 
  AffineLoopBand band;
  getLoopBandFromOutermost(forOp, band);
  auto newRoot = band[permuteLoops(band, permMap)];
  band.clear();
  getLoopBandFromOutermost(newRoot, band);
  auto II = getForLoopII(newRoot);
  // undo the permutation
  newRoot = band[permuteLoops(band, inversePerm(permMap))];
  band.clear();
  getLoopBandFromOutermost(newRoot, band);
  return II;
}
static uint64_t factorial(unsigned int n) {
    std::vector<unsigned int> numbers(n);
    std::iota(numbers.begin(), numbers.end(), 1);
    return std::accumulate(numbers.begin(), numbers.end(), 1ull, std::multiplies<unsigned long long>());
}

static AffineMap getAffineMapFromIntegerRelation(presburger::IntegerRelation rel, MLIRContext *context){
  presburger::IntegerRelation newRel = presburger::IntegerRelation(rel);
  if(rel.getNumSymbolVars() != 0){
    assert(false && "rel.getNumSymbolVars() != 0");
  }
  if(rel.getNumDomainVars() != 1){
    assert(false && "rel.getNumDomainVars() != 1");
  }
  auto numDimVars = rel.getNumDimVars();
  auto numLocals = rel.getNumLocalVars();
  rel.projectOut(numDimVars, numLocals);
  newRel.dump();
  rel.dump();
  LLVM_DEBUG(
    llvm::dbgs() << "newRel.isEqual(rel): " << newRel.isEqual(rel) << "\n";
    llvm::dbgs() << "newRel.computeVolume(): " << newRel.computeVolume() << "\n";
    llvm::dbgs() << "rel.computeVolume(): " << rel.computeVolume() << "\n";
  );
  AffineExpr expr = getAffineConstantExpr(0, context);
  for(unsigned i = 1; i < numDimVars; i++){
    expr = expr + (-rel.atEq64(0, i))*getAffineDimExpr(i-1, context);
  }
  expr = expr - rel.atEq64(0, numDimVars);
  return AffineMap::get(numDimVars - 1, 0, expr);
  // if(rel.getNumLocalVars() != rel.getNumDimVars()){
  //   assert(false && "rel.getNumLocalVars() != rel.getNumDimVars()");
  // }
  // time equation

  // DenseMap<unsigned, int64_t> vars;
  // DenseMap<unsigned, int64_t> consts;
  // auto numEqualities = rel.getNumEqualities();
  // auto numVars = rel.getNumVars();
  // auto numLocalVars = rel.getNumLocalVars();
  // auto numDomainVars = rel.getNumDomainVars();
  // auto numDimVars = rel.getNumDimVars();
  // auto numCols = rel.getNumCols();
  // auto numRangeVars = rel.getNumRangeVars();
  // // LLVM_DEBUG(
  // //   llvm::dbgs() << "numEqualities: " << numEqualities << "\n";
  // //   llvm::dbgs() << "numVars: " << numVars << "\n";
  // //   llvm::dbgs() << "numLocalVars: " << numLocalVars << "\n";
  // //   llvm::dbgs() << "numDomainVars: " << numDomainVars << "\n";
  // //   llvm::dbgs() << "numDimVars: " << numDimVars << "\n";
  // //   llvm::dbgs() << "numCols: " << numCols << "\n";
  // // );
  // auto timeEq = rel.getEquality64(0);
  // // get elements from numDimVars to numDimVars + numLocalVars
  // SmallVector<int64_t> localConsts;
  // for(unsigned i = numDimVars; i < numDimVars + numLocalVars; i++){
  //   localConsts.push_back(timeEq[i]);
  // }
  // for(unsigned row = 1; row < numEqualities; row++){
  //   auto eq = rel.getEquality64(row);
  //   for(unsigned col = numDimVars; col < numVars; col++){
  //     auto val = eq[col];
  //     if(val == 1){
  //       bool hasNoneZero = false;
  //       for(unsigned col2 = 0; col2 < numDimVars; col2++){
  //         if(eq[col2] == -1){
  //           vars[col - numDimVars] = col2-1;
  //           hasNoneZero = true;
  //         }
  //       }
  //       if(!hasNoneZero){
  //         consts[col - numDimVars] = eq[numVars];
  //       }
  //     }
  //   }
  // }
  // for(auto elem : vars){
  //   LLVM_DEBUG(
  //     llvm::dbgs() << "vars: " << elem.first << " " << elem.second << "\n";
  //   );
  // }
  // for(auto elem : consts){
  //   LLVM_DEBUG(
  //     llvm::dbgs() << "consts: " << elem.first << " " << elem.second << "\n";
  //   );
  // }
  // if(vars.size() + consts.size() != numLocalVars){
  //   assert(false && "vars.size() + consts.size() != numLocalVars");
  // }
  // for(auto elem : vars){
  //   LLVM_DEBUG(
  //     llvm::dbgs() << "vars: " << elem.first << " " << elem.second << "\n";
  //   );
  // }
  // for(auto elem : consts){
  //   LLVM_DEBUG(
  //     llvm::dbgs() << "consts: " << elem.first << " " << elem.second << "\n";
  //   );
  // }
  // AffineExpr expr = getAffineConstantExpr(0, context);
  // for(unsigned i = 0; i < numLocalVars; i++){
  //   if(vars.count(i) > 0){
  //     // exprs.push_back(getAffineDimExpr(vars[i], context));
  //     expr = expr + getAffineDimExpr(vars[i], context) * getAffineConstantExpr(localConsts[i], context);
  //   } else if(consts.count(i) > 0){
  //     // exprs.push_back(getAffineConstantExpr(consts[i], context));
  //     expr = expr + getAffineConstantExpr(-consts[i], context) * getAffineConstantExpr(localConsts[i], context);
  //   } else {
  //     assert(false && "vars.count(i) > 0 || consts.count(i) > 0");
  //   }
  // }
  // rel.dump();
  // AffineExpr expr = getAffineConstantExpr(0, context);
  // for(unsigned i = 1; i < numDimVars; i++){
  //   expr = expr + (-rel.atEq64(0, i))*getAffineDimExpr(i-1, context);
  // }
  // expr = expr + rel.atEq64(0, numDimVars);
  // return AffineMap::get(numDimVars - 1, 0, expr);
}

bool DFG::randomSearch(){
  auto seed = std::chrono::system_clock::now().time_since_epoch().count();
  srand(seed);
  unsigned varIdx = 0;
  for(auto& nodePair : nodes){
    auto id = nodePair.first;
    auto& node = nodePair.second;
    if(!node.op){
      continue;
    }
    auto forOp = dyn_cast<AffineForOp>(node.op);
    AffineLoopBand band;
    getLoopBandFromOutermost(forOp, band);
    Node* currNode = getNode(id);
    // llvm::dbgs() << "node: " << id << " trip count " << getLoopNestIterations(band) << "\n";
    // num permutations = factorial of the number of loops in the band
    auto numPerms = factorial(band.size());
    // choose random permutation between varIdx and varIdx + numPerms
    currNode->defaultPermIdx = varIdx + (rand() % numPerms);
    varIdx += numPerms;
  }
  // tmp file
  std::string reportFile = "/tmp/model";
  // createPythonModel2(reportFile, DFG::Default, /*optimizeOverlap*/ true);
  // run python script
  std::string cmd = "python " + reportFile + "_default.py";
  // get file name
  // call the command and direct its output to a file
  std::array<char, 128> buffer;
  std::string result;

  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
  if (!pipe) {
    return false;
  }
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }
  llvm::dbgs() << result;
  return true;
}

static bool isSubset(SmallVectorImpl<unsigned>& group, SmallVectorImpl<unsigned>& permMap){
  for(auto elem : group){
    if(std::find(permMap.begin(), permMap.end(), elem) == permMap.end()){
      return false;
    }
  }
  return true;
}

// the group is not ordered and the permMap is not ordered
static bool isContiguousSubset(SmallVectorImpl<unsigned>& group, SmallVectorImpl<unsigned>& permMap){
  if(group.size() == 0){
    return false;
  }
  if(permMap.size() == 0){
    return false;
  }
  auto firstElem = group[0];
  auto firstElemIdx = std::find(permMap.begin(), permMap.end(), firstElem);
  if(firstElemIdx == permMap.end()){
    return false;
  }
  for(unsigned i = 1; i < group.size(); i++){
    auto elem = group[i];
    auto elemIdx = std::find(permMap.begin(), permMap.end(), elem);
    if(elemIdx == permMap.end()){
      return false;
    }
    if(elemIdx - firstElemIdx != i){
      return false;
    }
  }
  return true;
}
static uint64_t getNumberOfDivisors(uint64_t num){
  uint64_t count = 0;
  for(uint64_t i = 1; i <= num; i++){
    if(num % i == 0){
      count++;
    }
  }
  return count;
}

static uint64_t getNumberOfDivisorTilingFactors(AffineLoopBand band){
  uint64_t numDivisors = 1;
  for(auto loop : band){
    auto tripCount = loop.getConstantUpperBound();
    numDivisors *= getNumberOfDivisors(tripCount);
  }
  return numDivisors;
}

LogicalResult DFG::populateNodeInfo(bool enablePermutations){
  unsigned varIdx = 0;
  uint64_t permDesignSpaceSize = 1;
  uint64_t tilingDesignSpaceSize = 1;
  for(auto& nodePair : nodes){
    auto id = nodePair.first;
    auto& node = nodePair.second;
    auto forOp = dyn_cast<AffineForOp>(node.op);

    node.isReduction = isReductionLoop(forOp);
    if(!forOp){
      assert(false && "node is not a for op");
    }
    forOp->walk([&](Operation *op) {
      if(isa<arith::AddFOp>(op)){
        node.DSP_factor += 2;
      }
      if(isa<arith::MulFOp>(op)){
        node.DSP_factor += 3;
      }
    });
    // collect stores and loads
    auto inputs = inEdges[id];
    auto outputs = outEdges[id];
    for(auto inEdge : inputs){
      auto value = inEdge.value;
      // LLVM_DEBUG(llvm::dbgs() << "inEdge: " << value << "\n");
      for(auto user : value.getUsers()){
        if(auto loadOp = dyn_cast<AffineLoadOp>(user)){
          node.loads.push_back(loadOp);
        }
      }
    }
    for(auto outEdge : outputs){
      auto value = outEdge.value;
      // LLVM_DEBUG(llvm::dbgs() << "outEdge: " << value << "\n");
      for(auto user : value.getUsers()){
        if(auto storeOp = dyn_cast<AffineStoreOp>(user)){
          node.stores.push_back(storeOp);
        }
      }
    }

    LLVM_DEBUG(
      llvm::dbgs() << "node: " << id << " has " << node.loads.size() << " loads and " << node.stores.size() << " stores\n";
    );

    AffineLoopBand band;
    getLoopBandFromOutermost(forOp, band);

    // original permutation map
    SmallVector<unsigned> permMap;
    for (unsigned i = 0; i < band.size(); i++) {
      permMap.push_back(i);
    }

    Node* currNode = getNode(id);
    auto numPerms = factorial(band.size());
    currNode->defaultPermIdx = 0;// + (rand() % numPerms);
    // get all permutations of the loops as a list
    auto allPerms = std::vector<SmallVector<unsigned>>();
    if(enablePermutations){
      do {
        allPerms.push_back(permMap);
      } while (std::next_permutation(permMap.begin(), permMap.end()));
    }else{
      allPerms.push_back(permMap);
    }

    permDesignSpaceSize *= allPerms.size();
    tilingDesignSpaceSize *= getNumberOfDivisorTilingFactors(band);

    int permIdx = 0;
    currNode->nodeInfo.resize(allPerms.size());

    for(auto permMap : allPerms){
      auto inversePermMap = inversePerm(permMap);
      LLVM_DEBUG(
        llvm::dbgs() << "Checking permutation #: " << varIdx << "\n";
      );
      // do the permutation
      auto newRoot = band[permuteLoops(band, permMap)];
      band.clear();
      getLoopBandFromOutermost(newRoot, band);

      // initialize info
      // currNode->permutationInfo.insert({varIdx, PermutationInfo()});
      currNode->nodeInfo[permIdx] = NodeInfo();
      // band[0].dump();

      // get store and load info
      for(auto store : currNode->stores){
        currNode->nodeInfo[permIdx].storesMap.insert({store, EdgeInfo()});
        currNode->nodeInfo[permIdx].storesMap[store].accessMap = getMinimalAccessPattern(store);
        auto firstElementTimeEq = getVariableTimeFunction(id, inversePermMap, store, true);
        auto lastElementTimeEq = getVariableTimeFunction(id, inversePermMap, store, false);
        currNode->nodeInfo[permIdx].storesMap[store].firstElementTimeEq = firstElementTimeEq;
        currNode->nodeInfo[permIdx].storesMap[store].lastElementTimeEq = lastElementTimeEq;
        auto firstElementTime = getTimeFunction(store, true);
        auto lastElementTime = getTimeFunction(store, false);
        // store->dump();
        // llvm::dbgs() << "assert " << firstElementTimeEq << " == " << firstElementTime << ", f\"{" << firstElementTimeEq << "} != " << firstElementTime << "\"\n";
        // llvm::dbgs() << "assert " << lastElementTimeEq << " == " << lastElementTime << ", f\"{" << lastElementTimeEq << "} != " << lastElementTime << "\"\n";
        currNode->nodeInfo[permIdx].storesMap[store].firstElementTime = firstElementTime;
        currNode->nodeInfo[permIdx].storesMap[store].lastElementTime = lastElementTime;
        // populate indexLoopMap
        for(auto memrefPair : llvm::enumerate(store.getMapOperands())){
          auto memref = memrefPair.value();
          for(auto forOpPair : llvm::enumerate(band)){
            auto loop = forOpPair.value();
            auto tripCount = loop.getConstantUpperBound();
            auto loopIdx = forOpPair.index();
            if(loop.getInductionVar() == memref){
              currNode->nodeInfo[permIdx].storesMap[store].indexLoopInfo.push_back({loopIdx, tripCount});
            }
          }
        }
      }
      for(auto load : currNode->loads){
        currNode->nodeInfo[permIdx].loadsMap.insert({load, EdgeInfo()});
        currNode->nodeInfo[permIdx].loadsMap[load].accessMap = getMinimalAccessPattern(load);
        auto firstElementTimeEq = getVariableTimeFunction(id, inversePermMap, load, true);
        auto lastElementTimeEq = getVariableTimeFunction(id, inversePermMap, load, false);
        currNode->nodeInfo[permIdx].loadsMap[load].firstElementTimeEq = firstElementTimeEq;
        currNode->nodeInfo[permIdx].loadsMap[load].lastElementTimeEq = lastElementTimeEq;
        auto firstElementTime = getTimeFunction(load, true);
        auto lastElementTime = getTimeFunction(load, false);
        // load->dump();
        // llvm::dbgs() << "assert " << firstElementTimeEq << " == " << firstElementTime << ", \"{" << firstElementTimeEq << "} != " << firstElementTime << "\"\n";
        // llvm::dbgs() << "assert " << lastElementTimeEq << " == " << lastElementTime << ", \"{" << lastElementTimeEq << "} != " << lastElementTime << "\"\n";
        currNode->nodeInfo[permIdx].loadsMap[load].firstElementTime = firstElementTime;
        currNode->nodeInfo[permIdx].loadsMap[load].lastElementTime = lastElementTime;
        // populate indexLoopMap
        for(auto memrefPair : llvm::enumerate(load.getMapOperands())){
          auto memref = memrefPair.value();
          for(auto forOpPair : llvm::enumerate(band)){
            auto loop = forOpPair.value();
            auto tripCount = loop.getConstantUpperBound();
            auto loopIdx = forOpPair.index();
            if(loop.getInductionVar() == memref){
              currNode->nodeInfo[permIdx].loadsMap[load].indexLoopInfo.push_back({loopIdx, tripCount});
            }
          }
        }
      }
      currNode->nodeInfo[permIdx].permutation = permMap;
      currNode->nodeInfo[permIdx].II = getForLoopII(newRoot);
      // undo the permutation
      newRoot = band[permuteLoops(band, inversePermMap)];
      band.clear();
      getLoopBandFromOutermost(newRoot, band);
      permIdx++;
    } 
    varIdx += allPerms.size();
  }
  // llvm::dbgs() << "NumberOfVariables: " << varIdx << "\n";
  llvm::dbgs() << "Permutation DesignSpaceSize: " << permDesignSpaceSize << "\n";
  llvm::dbgs() << "Parallelization DesignSpaceSize: " << tilingDesignSpaceSize << "\n";
  llvm::dbgs() << "Total DesignSpaceSize: " << permDesignSpaceSize * tilingDesignSpaceSize << "\n";
  return success();
}

bool DFG::createRootNode(){
  SmallVector<unsigned, 3> srcNodeIds;
  for(auto nodePair : nodes){
    auto id = nodePair.first;
    auto& node = nodePair.second;
    if(inEdges[id].size() == 0){
      srcNodeIds.push_back(id);
    }
  }
  Node root(10000, nullptr);
  root.nodeInfo.push_back(NodeInfo());
  nodes.insert({root.id, root});
  for(auto dstId : srcNodeIds){
    if (!hasEdge(root.id, dstId, nullptr)) {
      outEdges[root.id].push_back({dstId, nullptr});
      inEdges[dstId].push_back({root.id, nullptr});
    }
  }
  return true;
}

bool DFG::createSinkNode(){
  SmallVector<unsigned, 3> sinkNodeIds;
  for(auto nodePair : nodes){
    auto id = nodePair.first;
    auto& node = nodePair.second;
    if(outEdges[id].size() == 0){
      sinkNodeIds.push_back(id);
    }
  }
  Node sink(10001, nullptr);
  sink.tripCount = 0;
  nodes.insert({sink.id, sink});
  for(auto srcId : sinkNodeIds){
    if (!hasEdge(srcId, sink.id, nullptr)) {
      outEdges[srcId].push_back({sink.id, nullptr});
      inEdges[sink.id].push_back({srcId, nullptr});
    }
  }
  return true;
}

static void addVariable(std::stringstream& perfModel, std::string varName, std::string varType, int64_t varLB = -1, int64_t varUB = -1){
  if (varLB == -1 && varUB == -1)
    perfModel << "var " << varName << ", " << varType << ";\n";
  else
    perfModel << "var " << varName << " >= " << varLB << ", <= " << varUB << ", " << varType << ";\n";
}

static void addConstraint(std::stringstream& perfModel, std::string name, std::string lhs, std::string rhs, std::string op){
  perfModel << "subject to " << name << ":\n\t" << lhs << " " << op << " " << rhs << ";\n";
}

static void startTimeEquation(std::stringstream& perfModel, DFG& dfg, unsigned id){
  auto currNode = dfg.getNode(id);
  perfModel <<"# start time\n";
  // perfModel << "var st" << id << ", integer;\n";
  addVariable(perfModel, "st" + std::to_string(id), "integer");
  // perfModel << "subject to cst" << id << ":\n\tst" << id << " == ";
  std::stringstream lhs;
  lhs << "st" << id;
  std::stringstream rhs;
  if(dfg.inEdges[id].size() > 1)
    rhs << "max(";
  else
    rhs << "(";
  for(auto inEdgePair : llvm::enumerate(dfg.inEdges[id])){
    auto inEdge = inEdgePair.value();
    auto idx = inEdgePair.index();
    auto srcNode = dfg.getNode(inEdge.id);
    auto currNodeId = id;
    auto srcNodeId = inEdge.id;
    for(auto currPermInfoPair : llvm::enumerate(currNode->nodeInfo)){
      auto currPermIdx = currPermInfoPair.index();
      auto currPermInfo = currPermInfoPair.value();
      auto currAccessMap = currPermInfo.loadsMap[inEdge.dstOp].accessMap;
      for(auto srcPermInfoPair : llvm::enumerate(srcNode->nodeInfo)){
        auto srcPermIdx = srcPermInfoPair.index();
        auto srcPermInfo = srcPermInfoPair.value();
        auto srcAccessMap = srcPermInfo.storesMap[inEdge.srcOp].accessMap;
        if(srcAccessMap == currAccessMap){
          rhs << " + fw" << inEdge.id << "*b" << srcNodeId  << "_" << srcPermIdx << "*b" << currNodeId << "_" << currPermIdx;
        }else{
          rhs << " + lw" << inEdge.id << "*b" << srcNodeId << "_" << srcPermIdx << "*b" << currNodeId << "_" << currPermIdx;
        }
      }
    }
    if(idx == dfg.inEdges[id].size()-1){
      rhs << ")";
    } else {
      rhs << ", ";
    }
  }
  addConstraint(perfModel, "cst_st" + std::to_string(id), lhs.str(), rhs.str(), "==");
}

static void relativeLastReadEquations(std::stringstream& perfModel, DFG& dfg, unsigned id){
  auto currNode = dfg.getNode(id);
  perfModel << "# relative last reads\n";
  // perfModel << "# a_lr = c_st + node_c['a_lr']\n";
  // relative last read times of inputs
  for(auto inEdgePair : llvm::enumerate(dfg.inEdges[id])){
    auto inEdge = inEdgePair.value();
    addVariable(perfModel, "rlr" + std::to_string(inEdge.id) + "_" + std::to_string(id), "integer");
    std::stringstream lhs;
    lhs << "rlr" << inEdge.id << "_" << id;
    std::stringstream rhs;
    rhs << "st" << id;
    for(auto currPermInfoPair : llvm::enumerate(currNode->nodeInfo)){
      auto currPermIdx = currPermInfoPair.index();
      auto currPermInfo = currPermInfoPair.value();
      auto currII = currPermInfo.II;
      auto alr = currPermInfo.loadsMap[inEdge.dstOp].lastElementTimeEq;
      if(alr == "")
        rhs << " + 0 * b" << id << "_" << currPermIdx;
      else
        rhs << " + (" << alr << ") * " << currII << " * b" << id << "_" << currPermIdx;
    }
    addConstraint(perfModel, "cst_rlr" + std::to_string(inEdge.id) + "_" + std::to_string(id), lhs.str(), rhs.str(), "==");
  }
}

static void epiloguesEquations(std::stringstream& perfModel, DFG& dfg, unsigned id){
  auto currNode = dfg.getNode(id);
  perfModel << "# epilogues\n";
  // perfModel << "# c_epilogue_a = node_c['lw'] - node_c['a_lr']\n";
  // epilogues
  for(auto inEdgePair : llvm::enumerate(dfg.inEdges[id])){
    auto inEdge = inEdgePair.value();
    // auto idx = inEdgePair.index();
    // auto srcNode = dfg.getNode(inEdge.id);
    // perfModel<< "var ep" << inEdge.id << "_" << id << ", integer;\n";
    addVariable(perfModel, "ep" + std::to_string(inEdge.id) + "_" + std::to_string(id), "integer");
    // perfModel << "subject to cep" << inEdge.id << "_" << id << ":\n\tep" << inEdge.id << "_" << id << " == ";
    std::stringstream lhs;
    lhs << "ep" << inEdge.id << "_" << id;
    std::stringstream rhs;
    if(dfg.outEdges[id].size() == 0){
        rhs << "0";
    }else{
      for(auto currPermInfoPair : llvm::enumerate(currNode->nodeInfo)){
        auto currPermIdx = currPermInfoPair.index();
        auto currPermInfo = currPermInfoPair.value();
        auto currII = currPermInfo.II;
        for(auto outEdge : dfg.outEdges[id]){ // may need to handle multiple outputs
          auto srcOp = outEdge.srcOp;
          auto alw = currPermInfo.storesMap[srcOp].lastElementTimeEq;
          rhs << " + (" << alw << ") * " << currII << " * b" << id << "_" << currPermIdx;
        }
        auto alr = currPermInfo.loadsMap[inEdge.dstOp].lastElementTimeEq;
        rhs << " - (" << alr << ") * "<< currII << " * b" << id << "_" << currPermIdx;
      }
    }
    addConstraint(perfModel, "cep" + std::to_string(inEdge.id) + "_" + std::to_string(id), lhs.str(), rhs.str(), "==");
  }
}

static void dependencyEquations(std::stringstream& perfModel, DFG& dfg, unsigned id){
  perfModel << "# inputs last read/write dependencies\n";
  // perfModel << "# a_lt = max(a_lw, a_lr)\n";
  // inputs last read/write dependencies
  for(auto inEdgePair : llvm::enumerate(dfg.inEdges[id])){
    auto inEdge = inEdgePair.value();
    auto idx = inEdgePair.index();
    auto srcNode = dfg.getNode(inEdge.id);
    // perfModel<< "var lt" << inEdge.id << "_" << id << ", integer;\n";
    addVariable(perfModel, "lt" + std::to_string(inEdge.id) + "_" + std::to_string(id), "integer");
    // perfModel << "subject to clt" << inEdge.id << "_" << id << ":\n\tlt" << inEdge.id << "_" << id << " == max(rlr" << inEdge.id << "_" << id << ", lw" << inEdge.id << ");\n";
    std::stringstream lhs;
    lhs << "lt" << inEdge.id << "_" << id;
    std::stringstream rhs;
    rhs << "max(rlr" << inEdge.id << "_" << id << ", lw" << inEdge.id << ")";
    addConstraint(perfModel, "clt" + std::to_string(inEdge.id) + "_" + std::to_string(id), lhs.str(), rhs.str(), "==");
  }
}

static void dependencyEquations2(std::stringstream& perfModel, DFG& dfg, unsigned id){
  // curr last write times
  perfModel << "# curr last write times\n";
  // perfModel << "# c_a_et = max(a_lt, a_ft) + c_epilogue_a # max may not be necessary\n";
  for(auto inEdgePair : llvm::enumerate(dfg.inEdges[id])){
    auto inEdge = inEdgePair.value();
    auto idx = inEdgePair.index();
    auto srcNode = dfg.getNode(inEdge.id);
    // perfModel<< "var lw" << inEdge.id << "_" << id << ", integer;\n";
    addVariable(perfModel, "lw" + std::to_string(inEdge.id) + "_" + std::to_string(id), "integer");
    // perfModel << "subject to clw" << inEdge.id << "_" << id << ":\n\tlw" << inEdge.id << "_" << id << " == ";
    // perfModel << "lt" << inEdge.id << "_" << id << " + ";
    // perfModel << "ep" << inEdge.id << "_" << id << ";\n";
    std::stringstream lhs;
    lhs << "lw" << inEdge.id << "_" << id;
    std::stringstream rhs;
    rhs << "lt" << inEdge.id << "_" << id << " + ep" << inEdge.id << "_" << id;
    addConstraint(perfModel, "clw" + std::to_string(inEdge.id) + "_" + std::to_string(id), lhs.str(), rhs.str(), "==");
  }
}


static void lastWriteTimeEquation(std::stringstream& perfModel, DFG& dfg, unsigned id){
  relativeLastReadEquations(perfModel, dfg, id);
  epiloguesEquations(perfModel, dfg, id);
  dependencyEquations(perfModel, dfg, id);
  dependencyEquations2(perfModel, dfg, id);
  // node lw time
  perfModel << "# node last write time\n";
  // perfModel << "# c_et = max(c_a_et, c_b_et) + 1\n";
  // perfModel << "var lw" << id << ", integer;\n";
  addVariable(perfModel, "lw" + std::to_string(id), "integer");
  // perfModel << "subject to clw" << id << ":\n\tlw" << id << " == max(";
  std::stringstream lhs;
  lhs << "lw" << id;
  std::stringstream rhs;
  if(dfg.inEdges[id].size() > 1)
    rhs << "max(";
  else
    rhs << "(";
  for(auto inEdgePair : llvm::enumerate(dfg.inEdges[id])){
    auto inEdge = inEdgePair.value();
    auto idx = inEdgePair.index();
    // perfModel << "lw" << inEdge.id << "_" << id;
    rhs << "lw" << inEdge.id << "_" << id;
    if(idx != dfg.inEdges[id].size()-1){
      rhs << ", ";
    } else {
      rhs << ")";
    }
  }
  addConstraint(perfModel, "clw" + std::to_string(id), lhs.str(), rhs.str(), "==");
}

static void firstWriteTimeEquation(std::stringstream& perfModel, DFG& dfg, unsigned id){
  auto currNode = dfg.getNode(id);
  // node fw time
  if(dfg.outEdges[id].size() != 0){
    perfModel << "# node first write time\n";
    // perfModel << "# c_fw = c_st + node_c['fw']\n";
    // perfModel << "var fw" << id << ", integer;\n";
    addVariable(perfModel, "fw" + std::to_string(id), "integer");
    // perfModel << "subject to cfw" << id << ":\n\tfw" << id << " == ";
    // perfModel << "st" << id;
    std::stringstream lhs;
    lhs << "fw" << id;
    std::stringstream rhs;
    rhs << "st" << id;
    for(auto outEdge : dfg.outEdges[id]){ // may need to handle multiple outputs
      auto srcOp = outEdge.srcOp;
      for(auto currPermInfoPair : llvm::enumerate(currNode->nodeInfo)){
        auto currPermIdx = currPermInfoPair.index();
        auto currPermInfo = currPermInfoPair.value();
        auto currII = currPermInfo.II;
        auto afw = currPermInfo.storesMap[srcOp].firstElementTimeEq;
        // perfModel << " + " << afw << "*b" << id << "_" << currPermIdx;
        rhs << " + (" << afw << ") * " << currII << " * b" << id << "_" << currPermIdx;
      }
    }
    addConstraint(perfModel, "cfw" + std::to_string(id), lhs.str(), rhs.str(), "==");
  }
}
static void declarePermutationParameters(std::stringstream& perfModel, DFG& dfg){
  // declare fixed loop permutations
  for(auto nodePair : dfg.nodes){
    auto& node = nodePair.second;
    perfModel << "param b" << node.id << "_0 := 1;\n"; 
  }
}

static void declareTilingParameters(std::stringstream& perfModel, DFG& dfg){
  // declare fixed tiling factors
  for(auto nodePair : dfg.nodes){
    auto& node = nodePair.second;
    if(!node.op)
      continue;
    if(auto forOp = dyn_cast<AffineForOp>(node.op)){
      AffineLoopBand band;
      getLoopBandFromOutermost(forOp, band);
      for(unsigned i = 0; i < band.size(); i++){
        auto ub = band[i].getConstantUpperBound();
        perfModel << "param x" << node.id << "_" << i << " := " << ub << ";\n";
      }
    }else{
      assert(false && "node is not a for op");
    }
  }
}

static void declarePermutationVariables(std::stringstream& perfModel, DFG& dfg){
  // declare variable for each node
  perfModel << "# variable declation and constraints\n";
  for(auto nodePair : dfg.nodes){
    auto& node = nodePair.second;
    auto permutationsInfo = node.nodeInfo;
    if(permutationsInfo.size() > 0){
      for(auto permInfoPair : llvm::enumerate(permutationsInfo)){
        auto permIdx = permInfoPair.index();
        addVariable(perfModel, "b" + std::to_string(node.id) + "_" + std::to_string(permIdx), "binary");
      }
      std::stringstream lhs;
      lhs << "(";
      for(auto permInfoPair : llvm::enumerate(permutationsInfo)){
        auto permIdx = permInfoPair.index();
        lhs << "b" << node.id << "_" << permIdx;
        if(permIdx != (permutationsInfo.size()-1)){
          lhs << " + ";
        } else {
          lhs << ")";
        }
      }
      addConstraint(perfModel, "c" + std::to_string(node.id), lhs.str(), "1", "==");
    }
  }
}

static void declareRandomPermutationParams(std::stringstream& perfModel, DFG& dfg){
  // declare variable for each node
  perfModel << "# params declation and constraints\n";
  auto seed = std::chrono::system_clock::now().time_since_epoch().count();
  srand(seed);
  for(auto nodePair : dfg.nodes){
    auto& node = nodePair.second;
    auto permutationsInfo = node.nodeInfo;
    if(permutationsInfo.size() > 0){
      auto randIdx = rand() % permutationsInfo.size();
      for(auto permInfoPair : llvm::enumerate(permutationsInfo)){
        auto permIdx = permInfoPair.index();
        // addVariable(perfModel, "b" + std::to_string(node.id) + "_" + std::to_string(permIdx), "binary");
        if (permIdx == randIdx){
          perfModel << "param b" << node.id << "_" << std::to_string(permIdx) << " := " << 1 << ";\n";
        }else{
          perfModel << "param b" << node.id << "_" << std::to_string(permIdx) << " := " << 0 << ";\n";
        }
      }
      std::stringstream lhs;
      lhs << "(";
      for(auto permInfoPair : llvm::enumerate(permutationsInfo)){
        auto permIdx = permInfoPair.index();
        lhs << "b" << node.id << "_" << permIdx;
        if(permIdx != (permutationsInfo.size()-1)){
          lhs << " + ";
        } else {
          lhs << ")";
        }
      }
      addConstraint(perfModel, "c" + std::to_string(node.id), lhs.str(), "1", "==");
    }
  }
}

static void declareTilingVariables(std::stringstream& perfModel, DFG& dfg, unsigned tilingLimit){
  // declare variable for each node
  for(auto nodePair : dfg.nodes){
    auto& node = nodePair.second;
    if(!node.op)
      continue;
    if(auto forOp = dyn_cast<AffineForOp>(node.op)){
      AffineLoopBand band;
      getLoopBandFromOutermost(forOp, band);
      for(unsigned i = 0; i < band.size(); i++){
        auto ub = band[i].getConstantUpperBound();
        addVariable(perfModel, "x" + std::to_string(node.id) + "_" + std::to_string(i), "integer", (tilingLimit>ub? ub : tilingLimit), ub);
        addVariable(perfModel, "u" + std::to_string(node.id) + "_" + std::to_string(i), "integer", 1, ub);
        addConstraint(
          perfModel, 
          "c_uf" + std::to_string(node.id) + "_" + std::to_string(i), 
          "u" + std::to_string(node.id) + "_" + std::to_string(i) + " * x" + std::to_string(node.id) + "_" + std::to_string(i), 
          std::to_string(ub), 
          "=="
        );
      }
      addVariable(perfModel, "DSP_" + std::to_string(node.id), "integer");
      auto lhs = "DSP_" + std::to_string(node.id);
      std::stringstream rhs;
      // if(node.isReduction){
      for(unsigned i = 0; i < band.size(); i++){
        rhs << "u" << node.id << "_" << i;
        if(i != band.size()-1){
          rhs << " * ";
        } 
      }
      rhs << " * " << node.DSP_factor;
      // }else{
      //   rhs << "0";
      // }
      addConstraint(perfModel, "c_DSP" + std::to_string(node.id), lhs, rhs.str(), "==");
    }else{
      assert(false && "node is not a for op");
    }
  }
}
static int getRandomDivisor(int num, int limit){
  auto seed = std::chrono::system_clock::now().time_since_epoch().count();
  srand(seed);
  SmallVector<int> divisors;
  for(int i = 1; i <= num; i++){
    if(num % i == 0 && i >= limit){
      divisors.push_back(i);
    }
  }
  return divisors[rand() % divisors.size()];
}
static void declareRandomTilingParams(std::stringstream& perfModel, DFG& dfg, unsigned tilingLimit){
  // declare variable for each node

  for(auto nodePair : dfg.nodes){
    auto& node = nodePair.second;
    if(!node.op)
      continue;
    if(auto forOp = dyn_cast<AffineForOp>(node.op)){
      AffineLoopBand band;
      getLoopBandFromOutermost(forOp, band);
      for(unsigned i = 0; i < band.size(); i++){
        auto ub = band[i].getConstantUpperBound();
        // addVariable(perfModel, "x" + std::to_string(node.id) + "_" + std::to_string(i), "integer", (tilingLimit>ub? ub : tilingLimit), ub);
        // get random divisor of ub
        auto divisor = getRandomDivisor(ub, tilingLimit);
        // llvm::dbgs() <<"ub: " << ub << ", divisor: " << divisor << "\n";
        perfModel << "param x" << node.id << "_" << std::to_string(i) << " := " << divisor << ";\n";
        // addVariable(perfModel, "u" + std::to_string(node.id) + "_" + std::to_string(i), "integer", 1, ub);
        perfModel << "param u" << node.id << "_" << std::to_string(i) << " := " << ub/divisor << ";\n";
        // addConstraint(
        //   perfModel, 
        //   "c_uf" + std::to_string(node.id) + "_" + std::to_string(i), 
        //   "u" + std::to_string(node.id) + "_" + std::to_string(i) + " * x" + std::to_string(node.id) + "_" + std::to_string(i), 
        //   std::to_string(ub), 
        //   "=="
        // );
      }
      addVariable(perfModel, "DSP_" + std::to_string(node.id), "integer");
      auto lhs = "DSP_" + std::to_string(node.id);
      std::stringstream rhs;
      if(node.isReduction){
        for(unsigned i = 0; i < band.size(); i++){
          rhs << "u" << node.id << "_" << i;
          if(i != band.size()-1){
            rhs << " * ";
          } 
        }
      }else{
        rhs << "0";
      }
      addConstraint(perfModel, "c_DSP" + std::to_string(node.id), lhs, rhs.str(), "==");
    }else{
      assert(false && "node is not a for op");
    }
  }
}

static void tilingFactorConstraints(std::stringstream& perfModel, DFG& dfg){
  // add constraints
  // 1: edge parallel factor constraints
  for(auto nodePair : dfg.nodes){
    auto id = nodePair.first;
    auto& currNode = nodePair.second;
    auto currNodenodeInfo = currNode.nodeInfo[0];
    for(auto inEdgePair : llvm::enumerate(dfg.inEdges[id])){
      auto inEdge = inEdgePair.value();
      auto srcNode = dfg.getNode(inEdge.id);
      auto srcNodenodeInfo = srcNode->nodeInfo[0];
      auto storeOp = inEdge.srcOp;
      auto loadOp = inEdge.dstOp;
      auto srcIndexLoopInfo = srcNodenodeInfo.storesMap[storeOp].indexLoopInfo;
      auto currIndexLoopInfo = currNodenodeInfo.loadsMap[loadOp].indexLoopInfo;
      for(auto pair : llvm::zip(currIndexLoopInfo, srcIndexLoopInfo)){
        auto currIdx = std::get<0>(pair).first;
        auto srcIdx = std::get<1>(pair).first;
        auto currTripCount = std::get<0>(pair).second;
        auto srcTripCount = std::get<1>(pair).second;
        // assert(currTripCount == srcTripCount && "trip counts are not equal");
        // perfModel << "subject to " << "c_u" << id << "_" << currIdx << "_u" << srcNode->id << "_" << srcIdx << ":\n";
        // perfModel << "\tu" << id << "_" << currIdx << " == u" << srcNode->id << "_" << srcIdx << ";\n";
        addConstraint(
          perfModel, 
          "c_u" + std::to_string(id) + "_" + std::to_string(currIdx) + "_u" + std::to_string(srcNode->id) + "_" + std::to_string(srcIdx), 
          "u" + std::to_string(id) + "_" + std::to_string(currIdx), 
          "u" + std::to_string(srcNode->id) + "_" + std::to_string(srcIdx), 
          "=="
        );
      }
    }
  }
}

static void DSPsConstraints(std::stringstream& perfModel, DFG& dfg, unsigned DSPs){
  // 2: all DSPs
  std::stringstream totalDSP;
  bool has_dsps = false;
  for(auto nodePair : llvm::enumerate(dfg.nodes)){
    auto idx = nodePair.index();
    auto id = nodePair.value().first;
    if (id == 10000)
      continue;
    totalDSP << "DSP_" << id;
    has_dsps = true;
    if(idx != dfg.nodes.size()-1){
      totalDSP << " + ";
    }
  }

  if(!has_dsps){
    addVariable(perfModel, "totalDSPs", "integer");
    addConstraint(perfModel, "totalDSPDef", "totalDSPs" , "0", "==");
  }else{
    addVariable(perfModel, "totalDSPs", "integer");
    addConstraint(perfModel, "totalDSPDef", "totalDSPs" , totalDSP.str(), "==");
    addConstraint(perfModel, "totalDSPConst", "totalDSPs", std::to_string(DSPs), "<=");
  }
}

static void displayPermutationVariables(std::stringstream& perfModel, DFG& dfg){
  for(auto nodePair : dfg.nodes){
    auto& node = nodePair.second;
    auto permutationsInfo = node.nodeInfo;
    if(permutationsInfo.size() > 0){
      for(auto permInfoPair : llvm::enumerate(permutationsInfo)){
        auto permIdx = permInfoPair.index();
        perfModel << "display b" << node.id << "_" << permIdx << ";\n";
      }
    }
  }
}

static void displayTilingVariables(std::stringstream& perfModel, DFG& dfg){
  for(auto nodePair : dfg.nodes){
    auto& node = nodePair.second;
    if(!node.op)
      continue;
    if(auto forOp = dyn_cast<AffineForOp>(node.op)){
      AffineLoopBand band;
      getLoopBandFromOutermost(forOp, band);
      for(unsigned i = 0; i < band.size(); i++){
        perfModel << "display x" << node.id << "_" << i << ";\n";
        perfModel << "display u" << node.id << "_" << i << ";\n";
      }
    }else{
      assert(false && "node is not a for op");
    }
  }
}

static void displayDSPs(std::stringstream& perfModel, DFG& dfg){
  for(auto nodePair : dfg.nodes){
    auto& node = nodePair.second;
    if(!node.op)
      continue;
    if(auto forOp = dyn_cast<AffineForOp>(node.op)){
      perfModel << "display DSP_" << node.id << ";\n";
    }else{
      assert(false && "node is not a for op");
    }
  }
}

bool DFG::createPermutationPerformanceModel(std::string fileName, uint timeLimitMinutes){
  if(failed(populateNodeInfo(true))){
    return false;
  }
  // get the name after the last /
  auto kernelName = fileName.substr(fileName.find_last_of("/")+1);
  createRootNode();
  // createSinkNode();
  std::error_code stdErr;
  llvm::raw_fd_ostream graphFile(fileName + ".dot", stdErr);
  printAsDot(graphFile);
  std::stringstream perfModel;
  // solver type
  auto seconds = timeLimitMinutes * 60;
  perfModel << "# solver type\n";
  perfModel << "option solver gurobi;\n";
  // tech:logfile=gurobi_combined.log
  // perfModel << "option gurobi_options 'lim:time=" << seconds << "';\n";
  perfModel << "option gurobi_options 'opttol=1e-8 lim:time=" << seconds << " tech:logfile=" << kernelName << "_min.log';\n";

  // perfModel << "option gurobi_options 'tech:logfile=gurobi_combined.log';\n";

  declareTilingParameters(perfModel, *this);
  // declareRandomPermutationParams(perfModel, *this);

  declarePermutationVariables(perfModel, *this);

  // create time equations
  /*
  xfr: relative first read, x = a for absolute, x = r for relative
  xlr: relative last read, x = a for absolute, x = r for relative
  xfw: relative first write, x = a for absolute, x = r for relative
  xlw: relative last write, x = a for absolute, x = r for relative
  for a node n, we need to get:
  1. st(n) = max(rfw(i1), rfw(i2), ...) where i1, i2, ... are incoming nodes
  // 2. n_fw = st_n + fw_n, where fw is absolute time, with respect to node n
  2. for each incoming node i of node n, we need to get:
    rfr(i) = st(n) + afr(i)
    rlr(i) = st(n) + alr(i)
    ep(i) = alw(n) - alr(i)
    ft(i) = max(rfw(i), rfr(i))
    lt(i) = max(rlw(i), rlr(i))
  */
  unsigned nodeIdx = 0;
  SmallVector<unsigned, 2> outNodes;
  for(auto id : topologicalSort()){
    perfModel << "# node: " << id << " info\n";
    if(inEdges[id].size() == 0){ // root node
      addVariable(perfModel, "st" + std::to_string(id), "integer", 0, 0);
      addVariable(perfModel, "fw" + std::to_string(id), "integer", 0, 0);
      addVariable(perfModel, "lw" + std::to_string(id), "integer", 0, 0);
    }else{
      startTimeEquation(perfModel, *this, id);
      lastWriteTimeEquation(perfModel, *this, id);
      firstWriteTimeEquation(perfModel, *this, id);
    }
    if(outEdges[id].size() == 0){
      outNodes.push_back(id);
    }
    nodeIdx++;
    if(nodeIdx == nodes.size()){
      perfModel << "# objective function\n";
      perfModel << "var latency, integer;\n";
      perfModel << "subject to clatency:\n\tlatency == max(";
      for(auto idPair : llvm::enumerate(outNodes)){
        auto id = idPair.value();
        auto idx = idPair.index();
        perfModel << "lw" << id;
        if(idx != outNodes.size()-1){
          perfModel << ", ";
        }
      }
      perfModel << ");\n";
      perfModel << "minimize obj: latency;\n";
      perfModel << "solve obj;\n";
    }
  }
  displayPermutationVariables(perfModel, *this);
  perfModel << "display latency;\n";
  // write to files
  std::string perfModelStr = perfModel.str();
  std::error_code stdError;
  llvm::raw_fd_ostream perfFileMin(fileName + "_min.mod", stdError);
  if(!perfFileMin.has_error()) {
    perfFileMin << perfModelStr;
    perfFileMin.close();
  } else {
    return false;
  }
  return true;
}


bool DFG::createParallelizationPerformanceModel(std::string fileName, uint DSPs, uint tilingLimit, uint timeLimitMinutes){
  if(failed(populateNodeInfo(false))){
    return false;
  }
  createRootNode();
  // createSinkNode();
  std::error_code stdErr;
  llvm::raw_fd_ostream graphFile(fileName + "_parallel.dot", stdErr);
  printAsDot(graphFile);
  std::stringstream perfModel;
  auto seconds = timeLimitMinutes * 60;
  // solver type
  perfModel << "# solver type\n";
  perfModel << "option solver gurobi;\n";
  // perfModel << "option gurobi_options 'lim:time=" << seconds << "';\n";
  auto kernelName = fileName.substr(fileName.find_last_of("/")+1);
  perfModel << "option gurobi_options 'opttol=1e-8 lim:time=" << seconds << " tech:logfile=" << kernelName << "_parallel.log';\n";


  declarePermutationParameters(perfModel, *this);
  declareTilingVariables(perfModel, *this, tilingLimit);
  // declareRandomTilingParams(perfModel, *this, tilingLimit);
  tilingFactorConstraints(perfModel, *this);
  DSPsConstraints(perfModel, *this, DSPs);

  unsigned nodeIdx = 0;
  SmallVector<unsigned, 2> outNodes;
  for(auto id : topologicalSort()){
    perfModel << "# node: " << id << " info\n";
    if(inEdges[id].size() == 0){ // root node
      addVariable(perfModel, "st" + std::to_string(id), "integer", 0, 0);
      addVariable(perfModel, "fw" + std::to_string(id), "integer", 0, 0);
      addVariable(perfModel, "lw" + std::to_string(id), "integer", 0, 0);
    } else {
      startTimeEquation(perfModel, *this, id);
      lastWriteTimeEquation(perfModel, *this, id);
      firstWriteTimeEquation(perfModel, *this, id);
    }
    if(outEdges[id].size() == 0){
      outNodes.push_back(id);
    }
    nodeIdx++;
    if(nodeIdx == nodes.size()){
      perfModel << "# objective function\n";
      perfModel << "var latency, integer;\n";
      perfModel << "subject to clatency:\n\tlatency == max(";
      for(auto idPair : llvm::enumerate(outNodes)){
        auto id = idPair.value();
        auto idx = idPair.index();
        perfModel << "lw" << id;
        if(idx != outNodes.size()-1){
          perfModel << ", ";
        }
      }
      perfModel << ");\n";
      perfModel << "minimize obj: latency;\n";
      perfModel << "solve obj;\n";
    }
  }
  displayTilingVariables(perfModel, *this);
  displayDSPs(perfModel, *this);
  perfModel << "display latency;\n";
  perfModel << "display totalDSPs;\n";
  // write to files
  std::string perfModelStr = perfModel.str();
  std::error_code stdError;
  llvm::raw_fd_ostream perfFileMin(fileName + "_parallel.mod", stdError);
  if(!perfFileMin.has_error()) {
    perfFileMin << perfModelStr;
    perfFileMin.close();
    return true;
  } else {
    return false;
  }
}
bool DFG::createCombinedOptimizationPerformanceModel(std::string fileName, uint DSPs, uint tilingLimit, uint timeLimitMinutes){
  if(failed(populateNodeInfo(true))){
    return false;
  }
  createRootNode();
  // createSinkNode();
  std::error_code stdErr;
  llvm::raw_fd_ostream graphFile(fileName + "_parallel.dot", stdErr);
  printAsDot(graphFile);
  std::stringstream perfModel;
  auto seconds = timeLimitMinutes * 60;
  perfModel << "# solver type\n";
  perfModel << "option solver gurobi;\n";
  // perfModel << "option gurobi_options 'lim:time=" << seconds << "';\n";
  auto kernelName = fileName.substr(fileName.find_last_of("/")+1);
  perfModel << "option gurobi_options 'opttol=1e-8 lim:time=" << seconds << " tech:logfile=" << kernelName << "_combined.log';\n";


  declarePermutationVariables(perfModel, *this);
  declareTilingVariables(perfModel, *this, tilingLimit);
  tilingFactorConstraints(perfModel, *this);
  DSPsConstraints(perfModel, *this, DSPs);
  unsigned nodeIdx = 0;
  SmallVector<unsigned, 2> outNodes;
  for(auto id : topologicalSort()){
    perfModel << "# node: " << id << " info\n";
    if(inEdges[id].size() == 0){ // root node
      addVariable(perfModel, "st" + std::to_string(id), "integer", 0, 0);
      addVariable(perfModel, "fw" + std::to_string(id), "integer", 0, 0);
      addVariable(perfModel, "lw" + std::to_string(id), "integer", 0, 0);
    } else {
      startTimeEquation(perfModel, *this, id);
      lastWriteTimeEquation(perfModel, *this, id);
      firstWriteTimeEquation(perfModel, *this, id);
    }
    if(outEdges[id].size() == 0){
      outNodes.push_back(id);
    }
    nodeIdx++;
    if(nodeIdx == nodes.size()){
      perfModel << "# objective function\n";
      perfModel << "var latency, integer;\n";
      perfModel << "subject to clatency:\n\tlatency == max(";
      for(auto idPair : llvm::enumerate(outNodes)){
        auto id = idPair.value();
        auto idx = idPair.index();
        perfModel << "lw" << id;
        if(idx != outNodes.size()-1){
          perfModel << ", ";
        }
      }
      perfModel << ");\n";
      perfModel << "minimize obj: latency;\n";
      perfModel << "solve obj;\n";
    }
    // if(nodeIdx == nodes.size()){
    //   perfModel << "# objective function\n";
    //   perfModel << "var latency, integer;\n";
    //   perfModel << "subject to clatency:\n\tlatency == lw" << id << ";\n";
    //   perfModel << "minimize obj: latency;\n";
    //   perfModel << "solve obj;\n";
    // }
  }
  displayPermutationVariables(perfModel, *this);
  displayTilingVariables(perfModel, *this);
  displayDSPs(perfModel, *this);
  perfModel << "display latency;\n";
  perfModel << "display totalDSPs;\n";

  // write to files
  std::string perfModelStr = perfModel.str();
  std::error_code stdError;
  llvm::raw_fd_ostream perfFileMin(fileName + "_combined.mod", stdError);
  if(!perfFileMin.has_error()) {
    perfFileMin << perfModelStr;
    perfFileMin.close();
    return true;
  } else {
    return false;
  }
}

// bool DFG::createCombinedOptimizationPerformanceModel(std::string fileName, uint DSPs, uint tilingLimit){
//   populateCombinedOptimizationInfo();
//   createRootNode();
//   // createSinkNode();
//   std::error_code stdErr;
//   llvm::raw_fd_ostream graphFile(fileName + ".dot", stdErr);
//   printAsDot(graphFile);
//   std::stringstream perfModel;
//   // solver type
//   perfModel << "# solver type\n";
//   perfModel << "option solver gurobi;\n";

//   // declare indicator variables and constraints for loop permutation
//   perfModel << "# indicator variables and constraints for loop permutation\n";
//   for(auto nodePair : nodes){
//     auto& node = nodePair.second;
//     auto permutationsInfo = node.combinedOptimizationInfo;
//     if(permutationsInfo.size() > 0){
//       for(auto permInfoPair : llvm::enumerate(permutationsInfo)){
//         auto permIdx = permInfoPair.index();
//         perfModel << "var b" << node.id << "_" << permIdx << ", binary;\n";
//       }
//       perfModel << "subject to c_b" << node.id << ":\n\t(";
//       for(auto permInfoPair : llvm::enumerate(permutationsInfo)){
//         auto permIdx = permInfoPair.index();
//         perfModel << "b" << node.id << "_" << permInfoPair.index();
//         if(permIdx != (permutationsInfo.size()-1)){
//           perfModel << " + ";
//         } else {
//           perfModel << ") == 1;\n";
//         }
//       }
//     }
//   }
//   perfModel << "\n";  

//   // declare parallelization variables and constraints
//   perfModel << "# parallelization variables and constraints\n";
//   for(auto nodePair : nodes){
//     auto& node = nodePair.second;
//     if(!node.op)
//       continue;
//     if(auto forOp = dyn_cast<AffineForOp>(node.op)){
//       AffineLoopBand band;
//       getLoopBandFromOutermost(forOp, band);
//       for(unsigned i = 0; i < band.size(); i++){
//         auto ub = band[i].getConstantUpperBound();
//         perfModel << "var x" << node.id << "_" << i << " >= " << (tilingLimit>ub? ub : tilingLimit) << ", <= " << ub << ", integer;\n";
//         perfModel << "var u" << node.id << "_" << i << " >= 1, <= " << ub << ", integer;\n";
//         perfModel << "subject to c_uf" << node.id << "_" << i << ":\n\t";
//         perfModel << "u" << node.id << "_" << i << " * x" << node.id << "_" << i << " == " << ub << ";\n";
//         // perfModel << "# divisiblility dummy variable\n";
//         // perfModel << "var d" << node.id << "_" << i << " >= 1, integer;\n";
//         // perfModel << "subject to c" << node.id << "_" << i << ":\n\t";
//         // perfModel << "x" << node.id << "_" << i << " * d" << node.id << "_" << i << " == u" << node.id << "_" << i << ";\n";
//       }
//       perfModel << "var DSP_" << node.id << ", integer;\n";
//       perfModel << "subject to c_DSP" << node.id << ":\n\t DSP_" << node.id << " == ";
//       if(node.isReduction){
//         for(unsigned i = 0; i < band.size(); i++){
//           perfModel << "u" << node.id << "_" << i;
//           if(i != band.size()-1){
//             perfModel << " * ";
//           } else {
//             perfModel << ";\n";
//           }
//         }
//       }else{
//         perfModel << "0;\n";
//       }
//     }else{
//       assert(false && "node is not a for op");
//     }
//   }
//   // add constraints
//   // 1: edge parallel factor constraints
//   for(auto nodePair : nodes){
//     auto id = nodePair.first;
//     auto& currNode = nodePair.second;
//     if(id == 10000)
//       continue;
//     auto currNodeParallelizationInfo = currNode.combinedOptimizationInfo[0];
//     for(auto inEdgePair : llvm::enumerate(inEdges[id])){
//       auto inEdge = inEdgePair.value();
//       auto idx = inEdgePair.index();
//       auto srcNode = getNode(inEdge.id);
//       if(inEdge.id == 10000)
//         continue;
//       auto srcNodeParallelizationInfo = srcNode->combinedOptimizationInfo[0];
//       auto storeOp = inEdge.srcOp;
//       auto loadOp = inEdge.dstOp;
//       auto srcIndexLoopInfo = srcNodeParallelizationInfo.storesMap[storeOp].indexLoopInfo;
//       auto currIndexLoopInfo = currNodeParallelizationInfo.loadsMap[loadOp].indexLoopInfo;
//       for(auto pair : llvm::zip(currIndexLoopInfo, srcIndexLoopInfo)){
//         auto currIdx = std::get<0>(pair).first;
//         auto currTripCount = std::get<0>(pair).second;
//         auto srcIdx = std::get<1>(pair).first;
//         auto srcTripCount = std::get<1>(pair).second;
//         // perfModel << "subject to " << "c" << "x" << srcNode->id << "_" << srcIdx << "_x" << id << "_" << currIdx << ":\n";
//         // perfModel << "\t(" << "x" << srcNode->id << "_" << srcIdx << " == x" << id << "_" << currIdx << ";\n";
//         perfModel << "subject to " << "c_u" << id << "_" << currIdx << "_u" << srcNode->id << "_" << srcIdx << ":\n";
//         perfModel << "\tu" << id << "_" << currIdx << " == u" << srcNode->id << "_" << srcIdx << ";\n";
//       }
//     }
//   }
//   // 2: all DSPs
//   perfModel << "subject to totalDSP:\n\t";
//   for(auto nodePair : llvm::enumerate(nodes)){
//     auto idx = nodePair.index();
//     auto id = nodePair.value().first;
//     if (id == 10000)
//       continue;
//     perfModel << "DSP_" << id;
//     if(idx != nodes.size()-1){
//       perfModel << " + ";
//     } else {
//       perfModel << " <= " << DSPs << ";\n";
//     }
//   }
//   // create time equations
//   unsigned nodeIdx = 0;
//   /*
//   xfr: relative first read, x = a for absolute, x = r for relative
//   xlr: relative last read, x = a for absolute, x = r for relative
//   xfw: relative first write, x = a for absolute, x = r for relative
//   xlw: relative last write, x = a for absolute, x = r for relative
//   for a node n, we need to get:
//   1. st(n) = max(rfw(i1), rfw(i2), ...) where i1, i2, ... are incoming nodes
//   // 2. n_fw = st_n + fw_n, where fw is absolute time, with respect to node n
//   2. for each incoming node i of node n, we need to get:
//     rfr(i) = st(n) + afr(i)
//     rlr(i) = st(n) + alr(i)
//     ep(i) = alw(n) - alr(i)
//     ft(i) = max(rfw(i), rfr(i))
//     lt(i) = max(rlw(i), rlr(i))
//   */
//   for(auto id : topologicalSort()){
//     perfModel << "# node: " << id << " info\n";
//     auto currNode = getNode(id);
//     // root node
//     if(inEdges[id].size() == 0){
//       perfModel << "var st" << id << ", integer;\n";
//       perfModel << "subject to cst" << id << ":\n\tst" << id << " == 0;\n";
//       perfModel << "var fw" << id << ", integer;\n";
//       perfModel << "subject to cfw" << id << ":\n\tfw" << id << " == 0;\n";
//       perfModel << "var lw" << id << ", integer;\n";
//       perfModel << "subject to clw" << id << ":\n\tlw" << id << " == 0;\n";
//     }else{
//       // start time
//       // perfModel << "# start time\n";
//       perfModel <<"# c_st = max(a_fw, b_fw)\n";
//       perfModel << "var st" << id << ", integer;\n";
//       perfModel << "subject to cst" << id << ":\n\tst" << id << " == ";
//       if(inEdges[id].size() > 1)
//         perfModel << "max(\n";
//       else
//         perfModel << "(\n0";
//       for(auto inEdgePair : llvm::enumerate(inEdges[id])){
//         auto inEdge = inEdgePair.value();
//         auto idx = inEdgePair.index();
//         auto srcNode = getNode(inEdge.id);
//         auto currNodeId = id;
//         auto srcNodeId = inEdge.id;
//         for(auto currPermInfoPair : llvm::enumerate(currNode->combinedOptimizationInfo)){
//           auto currPermIdx = currPermInfoPair.index();
//           auto currPermInfo = currPermInfoPair.value();
//           auto currAccessMap = currPermInfo.loadsMap[inEdge.dstOp].accessMap;
//           for(auto srcPermInfoPair : llvm::enumerate(srcNode->combinedOptimizationInfo)){
//             auto srcPermIdx = srcPermInfoPair.index();
//             auto srcPermInfo = srcPermInfoPair.value();
//             auto srcAccessMap = srcPermInfo.storesMap[inEdge.srcOp].accessMap;
//             if(srcAccessMap == currAccessMap){
//               perfModel << " + fw" << inEdge.id << "*b" << srcNodeId  << "_" << srcPermIdx << "*b" << currNodeId << "_" << currPermIdx;
//             }else{
//               perfModel << " + lw" << inEdge.id << "*b" << srcNodeId << "_" << srcPermIdx << "*b" << currNodeId << "_" << currPermIdx;
//             }
//           }
//         }
//         if(idx == inEdges[id].size()-1){
//           perfModel << "\n);\n";
//         } else {
//           perfModel << ",\n";
//         }
//       }
//       // perfModel << "# relative first write times of inputs\n";
//       // perfModel << "# a_fr = c_st + node_c['a_fr']\n";
//       // // relative first read times of inputs
//       // for(auto inEdgePair : llvm::enumerate(inEdges[id])){
//       //   auto inEdge = inEdgePair.value();
//       //   auto idx = inEdgePair.index();
//       //   auto srcNode = getNode(inEdge.id);
//       //   perfModel<< "var rfr" << inEdge.id << "_" << id << ", integer;\n";
//       //   perfModel << "subject to crfr" << inEdge.id << "_" << id << ":\n\trfr" << inEdge.id << "_" << id << " == st" << id;
//       //   for(auto currPermPair : currNode->permutationMaps){
//       //     auto currPermIdx = currPermPair.first;
//       //     auto currII = currNode->permutationIIMaps[currPermIdx];
//       //     auto afr = currNode->nodeInfoMap[currPermIdx].loads[inEdge.dstOp].firstElementTime * currII;
//       //     perfModel << " + " << afr << "*x" << currPermIdx;
//       //   }
//       //   perfModel << ";\n";
//       // }
//       // perfModel << "# relative first write times of inputs\n";
//       perfModel << "# a_lr = c_st + node_c['a_lr']\n";
//       // relative last read times of inputs
//       for(auto inEdgePair : llvm::enumerate(inEdges[id])){
//         auto inEdge = inEdgePair.value();
//         // auto idx = inEdgePair.index();
//         // auto srcNode = getNode(inEdge.id);
//         perfModel<< "var rlr" << inEdge.id << "_" << id << ", integer;\n";
//         perfModel << "subject to crlr" << inEdge.id << "_" << id << ":\n\trlr" << inEdge.id << "_" << id << " == st" << id;
//         for(auto currPermInfoPair : llvm::enumerate(currNode->combinedOptimizationInfo)){
//           auto currPermIdx = currPermInfoPair.index();
//           auto currPermInfo = currPermInfoPair.value();
//           auto currII = currPermInfo.II;
//           std::string alr = "0";
//           if(inEdge.id != 10000){
//             alr = "((" + currPermInfo.loadsMap[inEdge.dstOp].lastElementTime + ") * " + std::to_string(currII) + ")";
//           }
//           perfModel << " + " << alr << "*b" << id << "_" << currPermIdx;
//         }
//         perfModel << ";\n";
//       }
//       // perfModel << "# epilogues\n";
//       perfModel << "# c_epilogue_a = node_c['lw'] - node_c['a_lr']\n";
//       // epilogues
//       for(auto inEdgePair : llvm::enumerate(inEdges[id])){
//         auto inEdge = inEdgePair.value();
//         auto idx = inEdgePair.index();
//         auto srcNode = getNode(inEdge.id);
//         perfModel<< "var ep" << inEdge.id << "_" << id << ", integer;\n";
//         perfModel << "subject to cep" << inEdge.id << "_" << id << ":\n\tep" << inEdge.id << "_" << id << " == ";
//         if(outEdges[id].size() == 0){
//             perfModel << "0\n";
//         }else{
//           for(auto currPermInfoPair : llvm::enumerate(currNode->combinedOptimizationInfo)){
//             auto currPermIdx = currPermInfoPair.index();
//             auto currPermInfo = currPermInfoPair.value();
//             auto currII = currPermInfo.II;
//             for(auto outEdge : outEdges[id]){ // may need to handle multiple outputs
//               auto srcOp = outEdge.srcOp;
//               auto alw = "((" + currPermInfo.storesMap[srcOp].lastElementTime + ") * " + std::to_string(currII) + ")";
//               perfModel << " + " << alw << "*b" << id << "_" << currPermIdx;
//             }
//             std::string alr = "0";
//             if(inEdge.id != 10000){
//               alr = "((" + currPermInfo.loadsMap[inEdge.dstOp].lastElementTime + ") * " + std::to_string(currII) + ")";
//             }
//             perfModel << " - " << alr << "*b" << id << "_" << currPermIdx;
//           }
//         }
//         perfModel << ";\n";
//       }

//       // // perfModel << "# inputs first read/write dependencies\n";
//       // perfModel << "# a_ft = max(a_fw, a_fr)\n";
//       // // inputs first read/write dependencies
//       // for(auto inEdgePair : llvm::enumerate(inEdges[id])){
//       //   auto inEdge = inEdgePair.value();
//       //   auto idx = inEdgePair.index();
//       //   auto srcNode = getNode(inEdge.id);
//       //   perfModel<< "var ft" << inEdge.id << "_" << id << ", integer;\n";
//       //   perfModel << "subject to cft" << inEdge.id << "_" << id << ":\n\tft" << inEdge.id << "_" << id << " == max(rfr" << inEdge.id << "_" << id << ", fw" << inEdge.id << ");\n";
//       // }
//       // perfModel << "# inputs last read/write dependencies\n";
//       perfModel << "# a_lt = max(a_lw, a_lr)\n";
//       // inputs last read/write dependencies
//       for(auto inEdgePair : llvm::enumerate(inEdges[id])){
//         auto inEdge = inEdgePair.value();
//         auto idx = inEdgePair.index();
//         auto srcNode = getNode(inEdge.id);
//         perfModel<< "var lt" << inEdge.id << "_" << id << ", integer;\n";
//         perfModel << "subject to clt" << inEdge.id << "_" << id << ":\n\tlt" << inEdge.id << "_" << id << " == max(rlr" << inEdge.id << "_" << id << ", lw" << inEdge.id << ");\n";
//       }
//       // curr last write times
//       // perfModel << "# curr last write times\n";
//       perfModel << "# c_a_et = max(a_lt, a_ft) + c_epilogue_a # max may not be necessary\n";
//       for(auto inEdgePair : llvm::enumerate(inEdges[id])){
//         auto inEdge = inEdgePair.value();
//         auto idx = inEdgePair.index();
//         auto srcNode = getNode(inEdge.id);
//         perfModel<< "var lw" << inEdge.id << "_" << id << ", integer;\n";
//         perfModel << "subject to clw" << inEdge.id << "_" << id << ":\n\tlw" << inEdge.id << "_" << id << " == ";
//         perfModel << "lt" << inEdge.id << "_" << id << " + ";
//         // perfModel << "ft" << inEdge.id << "_" << id << " + ";
//         perfModel << "ep" << inEdge.id << "_" << id << ";\n";
//       }

//       // node lw time
//       // perfModel << "# node last write time\n";
//       perfModel << "# c_et = max(c_a_et, c_b_et) + 1\n";
//       perfModel << "var lw" << id << ", integer;\n";
//       perfModel << "subject to clw" << id << ":\n\tlw" << id << " == max(";
//       for(auto inEdgePair : llvm::enumerate(inEdges[id])){
//         auto inEdge = inEdgePair.value();
//         auto idx = inEdgePair.index();
//         perfModel << "lw" << inEdge.id << "_" << id;
//         if(idx != inEdges[id].size()-1){
//           perfModel << ", ";
//         } else {
//           perfModel << ");\n";
//         }
//       }

//       // node fw time
//       // perfModel << "# node first write time\n";
//       if(outEdges[id].size() != 0){
//         perfModel << "# c_fw = c_st + node_c['fw']\n";
//         perfModel << "var fw" << id << ", integer;\n";
//         perfModel << "subject to cfw" << id << ":\n\tfw" << id << " == ";
//         perfModel << "st" << id;
//         for(auto outEdge : outEdges[id]){ // may need to handle multiple outputs
//           auto srcOp = outEdge.srcOp;
//           for(auto currPermInfoPair : llvm::enumerate(currNode->combinedOptimizationInfo)){
//             auto currPermIdx = currPermInfoPair.index();
//             auto currPermInfo = currPermInfoPair.value();
//             auto currII = currPermInfo.II;
//             auto afw = "((" + currPermInfo.storesMap[srcOp].firstElementTime + ") * " + std::to_string(currII) + ")";
//             perfModel << " + " << afw << "*b" << id << "_" << currPermIdx;
//           }
//         }
//       }
//       perfModel << ";\n\n";
//     }
//     nodeIdx++;
//     if(nodeIdx == nodes.size()){
//       perfModel << "# objective function\n";
//       perfModel << "var latency, integer;\n";
//       perfModel << "subject to clatency:\n\tlatency == lw" << id << ";\n";
//       perfModel << "minimize obj: latency;\n";
//       perfModel << "solve obj;\n";
//     }
//   }

//   for(auto nodePair : nodes){
//     auto& node = nodePair.second;
//     auto permutationsInfo = node.combinedOptimizationInfo;
//     if(permutationsInfo.size() > 0){
//       for(auto permInfoPair : llvm::enumerate(permutationsInfo)){
//         auto permIdx = permInfoPair.index();
//         perfModel << "display b" << node.id << "_" << permIdx << ";\n";
//       }
//     }
//   }
//   for(auto nodePair : nodes){
//     auto& node = nodePair.second;
//     if(!node.op)
//       continue;
//     if(auto forOp = dyn_cast<AffineForOp>(node.op)){
//       AffineLoopBand band;
//       getLoopBandFromOutermost(forOp, band);
//       for(unsigned i = 0; i < band.size(); i++){
//         // perfModel << "var x" << node.id << "_" << i << " >= 1, <= " << ub << ", integer;\n";
//         // perfModel << "var u" << node.id << "_" << i << " >= 1, <= " << ub << ", integer;\n";
//         // perfModel << "subject to c_uf" << node.id << "_" << i << ":\n\t";
//         // perfModel << "u" << node.id << "_" << i << " * x" << node.id << "_" << i << " == " << ub << ";\n";
//         perfModel << "display x" << node.id << "_" << i << ";\n";
//         perfModel << "display u" << node.id << "_" << i << ";\n";
//       }
//       perfModel << "display DSP_" << node.id << ";\n";
//     }else{
//       assert(false && "node is not a for op");
//     }
//   }
//   perfModel << "display latency;\n";

//   // write to files
//   std::string perfModelStr = perfModel.str();
//   std::error_code stdError;
//   llvm::raw_fd_ostream perfFileMin(fileName + "_combined.mod", stdError);
//   if(!perfFileMin.has_error()) {
//     perfFileMin << perfModelStr;
//     perfFileMin.close();
//     return true;
//   } else {
//     return false;
//   }
// }

bool DFG::createPermutationPythonModel(std::string fileName, DFG::PermutationType permType, bool optimizeOverlap){
  std::stringstream pythonModel;
  // declare variable for each node
  for(auto nodePair : nodes){
    auto& node = nodePair.second;
    if(node.nodeInfo.size() > 0){
      auto permIdx = -1;
      switch(permType){
        case DFG::PermutationType::Default:
          permIdx = node.defaultPermIdx;
          break;
        case DFG::PermutationType::Minimize:
          permIdx = node.minPermIdx;
          break;
        case DFG::PermutationType::Maximize:
          permIdx = node.maxPermIdx;
          break;
      }
      if(permIdx == -1)
        continue;
      pythonModel << "x" << node.id << "_" << permIdx << " = 1\n";
    }
  }
  unsigned nodeIdx = 0;
  for(auto id : topologicalSort()){
    auto currNode = getNode(id);
    auto currPermIdx = -1;
    switch(permType){
      case DFG::PermutationType::Default:
        currPermIdx = currNode->defaultPermIdx;
        break;
      case DFG::PermutationType::Minimize:
        currPermIdx = currNode->minPermIdx;
        break;
      case DFG::PermutationType::Maximize:
        currPermIdx = currNode->maxPermIdx;
        break;
    }
    pythonModel << "# node: " << id << " info\n";
    if(inEdges[id].size() == 0){
      pythonModel << "st" << id << " = 0\n";
      pythonModel << "fw" << id << " = 0\n";
      pythonModel << "lw" << id << " = 0\n\n";
    } else {
      // start time
      pythonModel << "# start time\n";
      // st equation
      pythonModel << "st" << id << " = ";
      if(inEdges[id].size() > 1)
        pythonModel << "max(";
      else
        pythonModel << "(";
      for(auto inEdgePair : llvm::enumerate(inEdges[id])){
        auto inEdge = inEdgePair.value();
        auto idx = inEdgePair.index();
        auto srcNode = getNode(inEdge.id);
        auto srcPermIdx = -1;
        switch(permType){
          case DFG::PermutationType::Default:
            srcPermIdx = srcNode->defaultPermIdx;
            break;
          case DFG::PermutationType::Minimize:
            srcPermIdx = srcNode->minPermIdx;
            break;
          case DFG::PermutationType::Maximize:
            srcPermIdx = srcNode->maxPermIdx;
            break;
        }
        pythonModel << "0";//<< inEdge.id;
        if(srcPermIdx != -1){
          auto srcAccessMap = srcNode->nodeInfo[srcPermIdx].storesMap[inEdge.srcOp].accessMap;
          auto currAccessMap = currNode->nodeInfo[currPermIdx].loadsMap[inEdge.dstOp].accessMap;
          auto flag = optimizeOverlap? true : srcNode->tripCount == currNode->tripCount;
          if(srcAccessMap == currAccessMap && flag){
            pythonModel << " + fw" << inEdge.id << "*x" << srcNode->id << "_" << srcPermIdx << "*x" << id << "_" << currPermIdx;
          } else {
            pythonModel << " + lw" << inEdge.id << "*x" << srcNode->id << "_" << srcPermIdx << "*x" << id << "_" << currPermIdx;
          }
        }
        if(idx == inEdges[id].size()-1){
          pythonModel << ")\n";
        } else {
          pythonModel << ", ";
        }
      }
      // pythonModel << "# relative first write times of inputs\n";
      // // relative first read times of inputs
      // for(auto inEdgePair : llvm::enumerate(inEdges[id])){
      //   auto inEdge = inEdgePair.value();
      //   auto idx = inEdgePair.index();
      //   auto srcNode = getNode(inEdge.id);
      //   // perfModel<< "var rfr" << inEdge.id << "_" << id << ", integer;\n";
      //   // perfModel << "subject to crfr" << inEdge.id << "_" << id << ":\n\trfr" << inEdge.id << "_" << id << " == st" << id;
      //   pythonModel << "rfw" << inEdge.id << "_" << id << " = ";
      //   auto currII = currNode->permutationIIMaps[currPermIdx];
      //   auto afr = currNode->nodeInfoMap[currPermIdx].loads[inEdge.dstOp].firstElementTime * currII;
      //   pythonModel << "st" << id << " + " << afr << "*x" << currPermIdx << "\n";
      // }

      pythonModel << "# relative first write times of inputs\n";
      // relative last read times of inputs
      for(auto inEdgePair : llvm::enumerate(inEdges[id])){
        auto inEdge = inEdgePair.value();
        auto idx = inEdgePair.index();
        auto srcNode = getNode(inEdge.id);
        // perfModel<< "var rlr" << inEdge.id << "_" << id << ", integer;\n";
        // perfModel << "subject to crlr" << inEdge.id << "_" << id << ":\n\trlr" << inEdge.id << "_" << id << " == st" << id;
        pythonModel << "rlr" << inEdge.id << "_" << id << " = ";
        auto currII = currNode->nodeInfo[currPermIdx].II;
        auto alr = currNode->nodeInfo[currPermIdx].loadsMap[inEdge.dstOp].lastElementTime * currII;
        pythonModel << "st" << id << " + " << alr << "*x" << id << "_" << currPermIdx << "\n";
      }
      pythonModel << "# epilogues\n";
      // epilogues
      for(auto inEdgePair : llvm::enumerate(inEdges[id])){
        auto inEdge = inEdgePair.value();
        auto idx = inEdgePair.index();
        auto srcNode = getNode(inEdge.id);
        // perfModel<< "var ep" << inEdge.id << "_" << id << ", integer;\n";
        // perfModel << "subject to cep" << inEdge.id << "_" << id << ":\n\tep" << inEdge.id << "_" << id << " == ";
        pythonModel << "ep" << inEdge.id << "_" << id << " = ";
        auto currII = currNode->nodeInfo[currPermIdx].II;
        if(outEdges[id].size() == 0){
          pythonModel << "0\n";
        }else{
          for(auto outEdge : outEdges[id]){ // may need to handle multiple outputs
            auto srcOp = outEdge.srcOp;
            auto alw = currNode->nodeInfo[currPermIdx].storesMap[srcOp].lastElementTime * currII;
            pythonModel << " + " << alw << "*x" << id << "_" << currPermIdx;
          }
          auto alr = currNode->nodeInfo[currPermIdx].loadsMap[inEdge.dstOp].lastElementTime * currII;
          pythonModel << " - " << alr << "*x" << id << "_" << currPermIdx << "\n";
        }
      }
      
      // pythonModel << "# inputs first read/write dependencies\n";
      // // inputs first read/write dependencies
      // for(auto inEdgePair : llvm::enumerate(inEdges[id])){
      //   auto inEdge = inEdgePair.value();
      //   auto idx = inEdgePair.index();
      //   auto srcNode = getNode(inEdge.id);
      //   // perfModel<< "var ft" << inEdge.id << "_" << id << ", integer;\n";
      //   // perfModel << "subject to cft" << inEdge.id << "_" << id << ":\n\tft" << inEdge.id << "_" << id << " == max(rfr" << inEdge.id << "_" << id << ", fw" << inEdge.id << ");\n";
      //   pythonModel << "ft" << inEdge.id << "_" << id << " = max(rfw" << inEdge.id << "_" << id << ", fw" << inEdge.id << ")\n";
      // }
      pythonModel << "# inputs last read/write dependencies\n";
      // inputs last read/write dependencies
      for(auto inEdgePair : llvm::enumerate(inEdges[id])){
        auto inEdge = inEdgePair.value();
        auto idx = inEdgePair.index();
        auto srcNode = getNode(inEdge.id);
        // perfModel<< "var lt" << inEdge.id << "_" << id << ", integer;\n";
        // perfModel << "subject to clt" << inEdge.id << "_" << id << ":\n\tlt" << inEdge.id << "_" << id << " == max(rlr" << inEdge.id << "_" << id << ", lw" << inEdge.id << ");\n";
        pythonModel << "lt" << inEdge.id << "_" << id << " = max(rlr" << inEdge.id << "_" << id << ", lw" << inEdge.id << ")\n";
      }
      // curr last write times
      pythonModel << "# curr last write times\n";
      // perfModel << "# c_a_et = max(a_lt, a_ft) + c_epilogue_a # max may not be necessary\n";
      for(auto inEdgePair : llvm::enumerate(inEdges[id])){
        auto inEdge = inEdgePair.value();
        auto idx = inEdgePair.index();
        auto srcNode = getNode(inEdge.id);
        // perfModel<< "var lw" << inEdge.id << "_" << id << ", integer;\n";
        // perfModel << "subject to clw" << inEdge.id << "_" << id << ":\n\tlw" << inEdge.id << "_" << id << " == max(";
        // perfModel << "lt" << inEdge.id << "_" << id << ", ";
        // perfModel << "ft" << inEdge.id << "_" << id << ") + ";
        // perfModel << "ep" << inEdge.id << "_" << id << ";\n";
        pythonModel << "lw" << inEdge.id << "_" << id << " = ";
        pythonModel << "lt" << inEdge.id << "_" << id << " + ";
        // pythonModel << "ft" << inEdge.id << "_" << id << ") + ";
        pythonModel << "ep" << inEdge.id << "_" << id << "\n";
      }
      // node lw time
      pythonModel << "# node last write time\n";
      // perfModel << "# c_et = max(c_a_et, c_b_et) + 1\n";
      // perfModel << "var lw" << id << ", integer;\n";
      // perfModel << "subject to clw" << id << ":\n\tlw" << id << " == max(";
      pythonModel << "lw" << id << " = ";
      if(inEdges[id].size() > 1)
        pythonModel << "max(";
      else
        pythonModel << "(";
      for(auto inEdgePair : llvm::enumerate(inEdges[id])){
        auto inEdge = inEdgePair.value();
        auto idx = inEdgePair.index();
        // perfModel << "lw" << inEdge.id << "_" << id;
        pythonModel << "lw" << inEdge.id << "_" << id;
        if(idx != inEdges[id].size()-1){
          pythonModel << ", ";
        } else {
          pythonModel << ")\n";
        }
      }

      // node fw time
      if(outEdges[id].size() != 0){
        pythonModel << "# node first write time\n";
        // perfModel << "# c_fw = c_st + node_c['fw']\n";
        // perfModel << "var fw" << id << ", integer;\n";
        // perfModel << "subject to cfw" << id << ":\n\tfw" << id << " == ";
        // perfModel << "st" << id;
        pythonModel << "fw" << id << " = st" << id;
        for(auto outEdge : outEdges[id]){ // may need to handle multiple outputs
          auto srcOp = outEdge.srcOp;
          auto currII = currNode->nodeInfo[currPermIdx].II;
          auto afw = currNode->nodeInfo[currPermIdx].storesMap[srcOp].firstElementTime * currII;
          pythonModel << " + " << afw << "*x" << id << "_" << currPermIdx;
        }
      }
      pythonModel << "\n\n";
    }
    // pythonModel << "print('Info for node: " << id << "')\n";
    // pythonModel << "print(f'tc" << id << ": ', tc" << id << ")\n";
    // pythonModel << "print(f'st" << id << ": ', st" << id << ")\n";
    // pythonModel << "print(f'et" << id << ": ', et" << id << ")\n";
    // pythonModel << "print('\\n')\n\n";
    nodeIdx++;
    if(nodeIdx == nodes.size()){
      // perfModel << "# objective function\n";
      // perfModel << "var latency, integer;\n";
      // perfModel << "subject to clatency:\n\tlatency == lw" << id << ";\n";
      // perfModel << "minimize obj: latency;\n";
      // perfModel << "solve obj;\n";
      pythonModel << "print(lw" << id << ")\n";
    }
  }
  // write to files
  std::string perfModelStr = pythonModel.str();
  std::error_code stdError;
  switch(permType){
    case DFG::PermutationType::Default:{
      fileName += optimizeOverlap? "_default.py" : "_orig.py";
      break;
    }
    case DFG::PermutationType::Minimize:
      fileName += "_min.py";
      break;
    case DFG::PermutationType::Maximize:
      fileName += "_max.py";
      break;
  }
  llvm::raw_fd_ostream perfFileMin(fileName, stdError);
  if(!perfFileMin.has_error()) {
    perfFileMin << perfModelStr;
    perfFileMin.close();
    return true;
  } else {
    return false;
  }
}

bool DFG::createParallelizationPythonModel(std::string fileName){
  std::stringstream pythonModel;
  // declare variable for each node
  for(auto nodePair : nodes){
    auto& node = nodePair.second;
    if(!node.op)
      continue;
    if(auto forOp = dyn_cast<AffineForOp>(node.op)){
      AffineLoopBand band;
      getLoopBandFromOutermost(forOp, band);
      for(unsigned i = 0; i < band.size(); i++){
        auto ub = band[i].getConstantUpperBound();
        pythonModel << "x" << node.id << "_" << i << " = " << ub / node.tilingFactors[i] << "\n";
        pythonModel << "u" << node.id << "_" << i << " = " << node.tilingFactors[i] << "\n";
        pythonModel << "assert u" << node.id << "_" << i << " * x" << node.id << "_" << i << " == " << ub << "\n";
        // perfModel << "# divisiblility dummy variable\n";
        // perfModel << "var d" << node.id << "_" << i << " >= 1, integer;\n";
        // perfModel << "subject to c" << node.id << "_" << i << ":\n\t";
        // pythonModel << "assert x" << node.id << "_" << i << " * d" << node.id << "_" << i << " == u" << node.id << "_" << i << "\n";
      }
      // perfModel << "var DSP_" << node.id << ", integer;\n";
      // perfModel << "subject to c_DSP" << node.id << ":\n\t DSP_" << node.id << " == ";
      pythonModel << "DSP_" << node.id << " = ";
      if(node.isReduction){
        for(unsigned i = 0; i < band.size(); i++){
          pythonModel << "u" << node.id << "_" << i;
          if(i != band.size()-1){
            pythonModel << " * ";
          } else {
            pythonModel << "\n";
          }
        }
      }else{
        pythonModel << "0\n";
      }
    }else{
      assert(false && "node is not a for op");
    }
  }
  // 2: all DSPs
  pythonModel << "totalDSP = ";
  for(auto nodePair : llvm::enumerate(nodes)){
    auto idx = nodePair.index();
    auto id = nodePair.value().first;
    if (id == 10000)
      continue;
    pythonModel << "DSP_" << id;
    if(idx != nodes.size()-1){
      pythonModel << " + ";
    } else {
      pythonModel << "\n";
    }
  }
  unsigned nodeIdx = 0;
  for(auto id : topologicalSort()){
    auto currNode = getNode(id);
    pythonModel << "# node: " << id << " info\n";
    if(inEdges[id].size() == 0){
      // perfModel << "var st" << id << ", integer;\n";
      // perfModel << "subject to cst" << id << ":\n\tst" << id << " == 0;\n";
      pythonModel << "st" << id << " = 0\n";
      // perfModel << "var fw" << id << ", integer;\n";
      // perfModel << "subject to cfw" << id << ":\n\tfw" << id << " == 0;\n";
      pythonModel << "fw" << id << " = 0\n";
      // perfModel << "var lw" << id << ", integer;\n";
      // perfModel << "subject to clw" << id << ":\n\tlw" << id << " == 0;\n";
      pythonModel << "lw" << id << " = 0\n";
    } else {
     // start time
      pythonModel << "# start time\n";
      // st equation
      // perfModel << "var st" << id << ", integer;\n";
      // perfModel << "subject to cst" << id << ":\n\tst" << id << " == ";
      pythonModel << "st" << id << " = ";
      if(inEdges[id].size() > 1)
        pythonModel << "max(";
      else
        pythonModel << "(0 + ";
      for(auto inEdgePair : llvm::enumerate(inEdges[id])){
        auto inEdge = inEdgePair.value();
        auto idx = inEdgePair.index();
        auto srcNode = getNode(inEdge.id);
        auto srcAccessMap = srcNode->nodeInfo[0].storesMap[inEdge.srcOp].accessMap;
        auto currAccessMap = currNode->nodeInfo[0].loadsMap[inEdge.dstOp].accessMap;
        if(srcAccessMap == currAccessMap){
          pythonModel << "fw" << inEdge.id;
        } else {
          pythonModel << "lw" << inEdge.id;
        }
        if(idx == inEdges[id].size()-1){
          pythonModel << ")\n";
        } else {
          pythonModel << ", ";
        }
      }
      pythonModel << "# relative first write times of inputs\n";
      // relative last read times of inputs
      for(auto inEdgePair : llvm::enumerate(inEdges[id])){
        auto inEdge = inEdgePair.value();
        auto idx = inEdgePair.index();
        auto srcNode = getNode(inEdge.id);
        // perfModel<< "var rlr" << inEdge.id << "_" << id << ", integer;\n";
        pythonModel << "rlr" << inEdge.id << "_" << id << " = ";
        auto currII = currNode->nodeInfo[0].II;
        // alr is a variable
        if(inEdge.id != 10000){
          auto alr = currNode->nodeInfo[0].loadsMap[inEdge.dstOp].lastElementTime;// * currII;
          // perfModel << "subject to crlr" << inEdge.id << "_" << id << ":\n\trlr" << inEdge.id << "_" << id;
          // perfModel << " == st" << id << " + (" << alr << ") * "<< currII << ";\n";
          pythonModel << "st" << id << " + (" << alr << ") * "<< currII << "\n";
        }else{
        // auto alr = currNode->nodeInfo[0].loadsMap[inEdge.dstOp].lastElementTime;// * currII;
          // perfModel << "subject to crlr" << inEdge.id << "_" << id << ":\n\trlr" << inEdge.id << "_" << id;
          // perfModel << " == st" << id << ";\n";
          pythonModel << "st" << id << "\n";
        }
      }
      pythonModel << "# epilogues\n";
      // epilogues
      for(auto inEdgePair : llvm::enumerate(inEdges[id])){
        auto inEdge = inEdgePair.value();
        auto idx = inEdgePair.index();
        auto srcNode = getNode(inEdge.id);
        // perfModel<< "var ep" << inEdge.id << "_" << id << ", integer;\n";
        // perfModel << "subject to cep" << inEdge.id << "_" << id << ":\n\tep" << inEdge.id << "_" << id << " == ";
        pythonModel << "ep" << inEdge.id << "_" << id << " = ";
        auto currII = currNode->nodeInfo[0].II;
        if(outEdges[id].size() == 0){
          // perfModel << "0;\n";
          pythonModel << "0\n";
        }else{
          for(auto outEdge : outEdges[id]){ // may need to handle multiple outputs
            auto srcOp = outEdge.srcOp;
            auto alw = currNode->nodeInfo[0].storesMap[srcOp].lastElementTime;
            // perfModel << " + (" << alw << ") * " << currII;
            pythonModel << " + (" << alw << ") * " << currII;
          }
          if(inEdge.id != 10000){
            auto alr = currNode->nodeInfo[0].loadsMap[inEdge.dstOp].lastElementTime;
            // perfModel << " - (" << alr << ") * " << currII << ";\n";
            pythonModel << " - (" << alr << ") * " << currII << "\n";
          }else{
            // perfModel << ";\n";
            pythonModel << "\n";
          }
        }
      }
      pythonModel << "# inputs last read/write dependencies\n";
      // inputs last read/write dependencies
      for(auto inEdgePair : llvm::enumerate(inEdges[id])){
        auto inEdge = inEdgePair.value();
        auto idx = inEdgePair.index();
        auto srcNode = getNode(inEdge.id);
        // perfModel<< "var lt" << inEdge.id << "_" << id << ", integer;\n";
        // perfModel << "subject to clt" << inEdge.id << "_" << id << ":\n\tlt" << inEdge.id << "_" << id << " == max(rlr" << inEdge.id << "_" << id << ", lw" << inEdge.id << ");\n";
        pythonModel << "lt" << inEdge.id << "_" << id << " = max(rlr" << inEdge.id << "_" << id << ", lw" << inEdge.id << ")\n";
      }
      // curr last write times
      pythonModel << "# curr last write times\n";
      for(auto inEdgePair : llvm::enumerate(inEdges[id])){
        auto inEdge = inEdgePair.value();
        auto idx = inEdgePair.index();
        auto srcNode = getNode(inEdge.id);
        // perfModel<< "var lw" << inEdge.id << "_" << id << ", integer;\n";
        // perfModel << "subject to clw" << inEdge.id << "_" << id << ":\n\tlw" << inEdge.id << "_" << id << " == ";
        // perfModel << "lt" << inEdge.id << "_" << id << " + ";
        // perfModel << "ep" << inEdge.id << "_" << id << ";\n";
        pythonModel << "lw" << inEdge.id << "_" << id << " = ";
        pythonModel << "lt" << inEdge.id << "_" << id << " + ";
        pythonModel << "ep" << inEdge.id << "_" << id << "\n";
      }
      
      // node lw time
      // perfModel << "# node last write time\n";
      pythonModel << "# c_et = max(c_a_et, c_b_et) + 1\n";
      // perfModel << "var lw" << id << ", integer;\n";
      // perfModel << "subject to clw" << id << ":\n\tlw" << id << " == max(";
      pythonModel << "lw" << id << " = ";
      if(inEdges[id].size() > 1)
        pythonModel << "max(";
      else
        pythonModel << "(";
      for(auto inEdgePair : llvm::enumerate(inEdges[id])){
        auto inEdge = inEdgePair.value();
        auto idx = inEdgePair.index();
        pythonModel << "lw" << inEdge.id << "_" << id;
        if(idx != inEdges[id].size()-1){
          pythonModel << ", ";
        } else {
          pythonModel << ")\n";
        }
      }

      // node fw time
      if(outEdges[id].size() != 0){
        pythonModel << "# node first write time\n";
        // perfModel << "fw" << id << " = st" << id;
        // perfModel << "var fw" << id << ", integer;\n";
        // perfModel << "subject to cfw" << id << ":\n\tfw" << id << " == st" << id;
        pythonModel << "fw" << id << " = st" << id;
        for(auto outEdge : outEdges[id]){ // may need to handle multiple outputs
          auto srcOp = outEdge.srcOp;
          auto currII = currNode->nodeInfo[0].II;
          auto afw = currNode->nodeInfo[0].storesMap[srcOp].firstElementTime;
          pythonModel << " + (" << afw << ") * " << currII;
        }
      }
      pythonModel << "\n\n";
    }
    nodeIdx++;
    if(nodeIdx == nodes.size()){
      pythonModel << "# objective function\n";
      // perfModel << "var latency, integer;\n";
      // perfModel << "subject to clatency:\n\tlatency == lw" << id << ";\n";
      // perfModel << "minimize obj: latency;\n";
      // perfModel << "solve obj;\n";
      pythonModel << "latency = lw" << id << "\n";
    }
  }
  pythonModel << "print(latency)\n";
  pythonModel << "print(totalDSP)\n";
  // write to files
  std::string perfModelStr = pythonModel.str();
  std::error_code stdError;
  llvm::raw_fd_ostream perfFileMin(fileName + "_parallel.py", stdError);
  if(!perfFileMin.has_error()) {
    perfFileMin << perfModelStr;
    perfFileMin.close();
    return true;
  } else {
    return false;
  }
  return true;
}

bool DFG::createCombinedOptimizationPythonModel(std::string fileName){
  std::stringstream pythonModel;
  // declare variable for each node
  for(auto nodePair : nodes){
    auto& node = nodePair.second;
    if(node.nodeInfo.size() > 0){
      pythonModel << "b" << node.id << "_" << node.minPermIdx << " = 1\n";
    }
  }
  // declare variable for each node
  for(auto nodePair : nodes){
    auto& node = nodePair.second;
    if(!node.op)
      continue;
    auto optPermutation = node.nodeInfo[node.minPermIdx].permutation;
    if(auto forOp = dyn_cast<AffineForOp>(node.op)){
      AffineLoopBand band;
      getLoopBandFromOutermost(forOp, band);
      for(unsigned i = 0; i < band.size(); i++){
        auto ub = band[i].getConstantUpperBound();
        pythonModel << "x" << node.id << "_" << optPermutation[i] << " = " << ub / node.tilingFactors[optPermutation[i]] << "\n";
        pythonModel << "u" << node.id << "_" << optPermutation[i] << " = " << node.tilingFactors[optPermutation[i]] << "\n";
        pythonModel << "assert u" << node.id << "_" << optPermutation[i] << " * x" << node.id << "_" << optPermutation[i] << " == " << ub << "\n";
        // perfModel << "# divisiblility dummy variable\n";
        // perfModel << "var d" << node.id << "_" << i << " >= 1, integer;\n";
        // perfModel << "subject to c" << node.id << "_" << i << ":\n\t";
        // pythonModel << "assert x" << node.id << "_" << i << " * d" << node.id << "_" << i << " == u" << node.id << "_" << i << "\n";
      }
      // perfModel << "var DSP_" << node.id << ", integer;\n";
      // perfModel << "subject to c_DSP" << node.id << ":\n\t DSP_" << node.id << " == ";
      pythonModel << "DSP_" << node.id << " = ";
      if(node.isReduction){
        for(unsigned i = 0; i < band.size(); i++){
          pythonModel << "u" << node.id << "_" << i;
          if(i != band.size()-1){
            pythonModel << " * ";
          } else {
            pythonModel << "\n";
          }
        }
      }else{
        pythonModel << "0\n";
      }
    }else{
      assert(false && "node is not a for op");
    }
  }
  // 2: all DSPs
  pythonModel << "totalDSP = ";
  for(auto nodePair : llvm::enumerate(nodes)){
    auto idx = nodePair.index();
    auto id = nodePair.value().first;
    if (id == 10000)
      continue;
    pythonModel << "DSP_" << id;
    if(idx != nodes.size()-1){
      pythonModel << " + ";
    } else {
      pythonModel << "\n";
    }
  }
  unsigned nodeIdx = 0;
  for(auto id : topologicalSort()){
    auto currNode = getNode(id);
    pythonModel << "# node: " << id << " info\n";
    if(inEdges[id].size() == 0){
      pythonModel << "st" << id << " = 0\n";
      pythonModel << "fw" << id << " = 0\n";
      pythonModel << "lw" << id << " = 0\n";
    } else {
     // start time
      pythonModel << "# start time\n";
      // st equation
      pythonModel << "st" << id << " = ";
      if(inEdges[id].size() > 1)
        pythonModel << "max(";
      else
        pythonModel << "(";
      for(auto inEdgePair : llvm::enumerate(inEdges[id])){
        auto inEdge = inEdgePair.value();
        auto idx = inEdgePair.index();
        auto srcNode = getNode(inEdge.id);
        auto currPermIdx = currNode->minPermIdx;
        auto srcPermIdx = srcNode->minPermIdx;
        pythonModel << "0";//<< inEdge.id;
        if(srcPermIdx != -1){
          auto srcAccessMap = srcNode->nodeInfo[srcPermIdx].storesMap[inEdge.srcOp].accessMap;
          auto currAccessMap = currNode->nodeInfo[currPermIdx].loadsMap[inEdge.dstOp].accessMap;
          if(srcAccessMap == currAccessMap){
            pythonModel << " + fw" << inEdge.id << "*b" << srcNode->id << "_" << srcPermIdx << "*b" << id << "_" << currPermIdx;
          } else {
            pythonModel << " + lw" << inEdge.id << "*b" << srcNode->id << "_" << srcPermIdx << "*b" << id << "_" << currPermIdx;
          }
        }
        if(idx == inEdges[id].size()-1){
          pythonModel << ")\n";
        } else {
          pythonModel << ", ";
        }
      }
      pythonModel << "# relative first write times of inputs\n";
      // relative last read times of inputs
      for(auto inEdgePair : llvm::enumerate(inEdges[id])){
        auto inEdge = inEdgePair.value();
        auto idx = inEdgePair.index();
        auto srcNode = getNode(inEdge.id);
        auto currPermIdx = currNode->minPermIdx;
        auto srcPermIdx = srcNode->minPermIdx;
        pythonModel << "rlr" << inEdge.id << "_" << id << " = ";
        auto currII = currNode->nodeInfo[currPermIdx].II;
        // alr is a variable
        if(inEdge.id != 10000){
          auto alr = currNode->nodeInfo[currPermIdx].loadsMap[inEdge.dstOp].lastElementTime;// * currII;
          pythonModel << "st" << id << " + ((" << alr << ") * "<< currII << ") * b" << id << "_" << currPermIdx << "\n";
        }else{
          pythonModel << "st" << id << "\n";
        }
      }
      pythonModel << "# epilogues\n";
      // epilogues
      for(auto inEdgePair : llvm::enumerate(inEdges[id])){
        auto inEdge = inEdgePair.value();
        auto idx = inEdgePair.index();
        auto srcNode = getNode(inEdge.id);
        auto currPermIdx = currNode->minPermIdx;
        auto srcPermIdx = srcNode->minPermIdx;
        pythonModel << "ep" << inEdge.id << "_" << id << " = ";
        auto currII = currNode->nodeInfo[currPermIdx].II;
        if(outEdges[id].size() == 0){
          // perfModel << "0;\n";
          pythonModel << "0\n";
        }else{
          for(auto outEdge : outEdges[id]){ // may need to handle multiple outputs
            auto srcOp = outEdge.srcOp;
            auto alw = currNode->nodeInfo[currPermIdx].storesMap[srcOp].lastElementTime;
            pythonModel << " + ((" << alw << ") * " << currII << ") * b" << id << "_" << currPermIdx;
          }
          if(inEdge.id != 10000){
            auto alr = currNode->nodeInfo[currPermIdx].loadsMap[inEdge.dstOp].lastElementTime;
            pythonModel << " - ((" << alr << ") * " << currII << ") * b" << id << "_" << currPermIdx << "\n";
          }else{
            pythonModel << "\n";
          }
        }
      }
      pythonModel << "# inputs last read/write dependencies\n";
      // inputs last read/write dependencies
      for(auto inEdgePair : llvm::enumerate(inEdges[id])){
        auto inEdge = inEdgePair.value();
        auto idx = inEdgePair.index();
        auto srcNode = getNode(inEdge.id);
        pythonModel << "lt" << inEdge.id << "_" << id << " = max(rlr" << inEdge.id << "_" << id << ", lw" << inEdge.id << ")\n";
      }
      // curr last write times
      pythonModel << "# curr last write times\n";
      for(auto inEdgePair : llvm::enumerate(inEdges[id])){
        auto inEdge = inEdgePair.value();
        auto idx = inEdgePair.index();
        auto srcNode = getNode(inEdge.id);
        pythonModel << "lw" << inEdge.id << "_" << id << " = ";
        pythonModel << "lt" << inEdge.id << "_" << id << " + ";
        pythonModel << "ep" << inEdge.id << "_" << id << "\n";
      }
      
      // node lw time
      pythonModel << "# c_et = max(c_a_et, c_b_et) + 1\n";
      pythonModel << "lw" << id << " = ";
      if(inEdges[id].size() > 1)
        pythonModel << "max(";
      else
        pythonModel << "(";
      for(auto inEdgePair : llvm::enumerate(inEdges[id])){
        auto inEdge = inEdgePair.value();
        auto idx = inEdgePair.index();
        pythonModel << "lw" << inEdge.id << "_" << id;
        if(idx != inEdges[id].size()-1){
          pythonModel << ", ";
        } else {
          pythonModel << ")\n";
        }
      }

      // node fw time
      if(outEdges[id].size() != 0){
        pythonModel << "# node first write time\n";
        pythonModel << "fw" << id << " = st" << id;
        for(auto outEdge : outEdges[id]){ // may need to handle multiple outputs
          auto srcOp = outEdge.srcOp;
          auto currPermIdx = currNode->minPermIdx;
          auto currII = currNode->nodeInfo[currPermIdx].II;
          auto afw = currNode->nodeInfo[currPermIdx].storesMap[srcOp].firstElementTime;
          pythonModel << " + (" << afw << ") * " << currII;
        }
      }
      pythonModel << "\n\n";
    }
    nodeIdx++;
    if(nodeIdx == nodes.size()){
      pythonModel << "# objective function\n";
      pythonModel << "latency = lw" << id << "\n";
    }
  }
  pythonModel << "print(latency)\n";
  pythonModel << "print(totalDSP)\n";
  // write to files
  std::string perfModelStr = pythonModel.str();
  std::error_code stdError;
  llvm::raw_fd_ostream perfFileMin(fileName + "_combined.py", stdError);
  if(!perfFileMin.has_error()) {
    perfFileMin << perfModelStr;
    perfFileMin.close();
    return true;
  } else {
    return false;
  }
  return true;
}

bool DFG::callParallelizationSolver(std::string filePath){
  // if(isMinimize){
  //   filePath += "_min.mod";
  // } else {
  //   filePath += "_max.mod";
  // }
  llvm::dbgs() << "Parallelization solver: ";
  filePath += "_parallel.mod";
  std::string command = "ampl " + filePath + "\n";
  // get file name
  // call the command and direct its output to a file
  std::array<char, 128> buffer;
  std::string result;

  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);
  if (!pipe) {
    return false;
  }
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }
  // DenseMap<const char*, bool> solution;
  std::map<std::string, uint64_t> solution;
  // go through result checking for pattern x[0-9]+_[0-9]+ = 1 | 0
  std::regex pattern("(u[0-9]+_[0-9]+) = (.+)");
  std::regex latencyPattern("latency = ([0-9]+)");
  std::regex totalDSPsPattern("totalDSPs = ([0-9]+)");
  std::smatch match;
  std::istringstream iss(result);
  std::string line;
  while (std::getline(iss, line)) {
    if(std::regex_search(line, match, pattern)){
      auto key = match.str(1);
      auto val = std::stoi(match.str(2));
      solution[key] = val;
    }
    if(std::regex_search(line, match, latencyPattern)){
      auto latency = std::stoi(match.str(1));
      llvm::dbgs() << "Parallel Latency: " << latency << "\n";
    }
    if(std::regex_search(line, match, totalDSPsPattern)){
      auto totalDSPs = std::stoi(match.str(1));
      llvm::dbgs() << "Total DSPs: " << totalDSPs << "\n";
    }
  }
  if(solution.size() == 0){
    assert(false && "no solution found");
  }
  for(auto sol : solution){
    LLVM_DEBUG(
      llvm::dbgs() << sol.first << " = " << sol.second << "\n";
    );
  }
  // for(auto nodePair : nodes){
  //   LLVM_DEBUG(
  //     llvm::dbgs() << "Node: " << nodePair.second.optPermutation.size() << "\n";
  //   );
  // }
  LLVM_DEBUG(
    llvm::dbgs() << "solution size: " << solution.size() << "\n";
  );
  for(auto& nodePair : nodes){
    auto& node = nodePair.second;
    if(!node.op)
      continue;
    if(auto forOp = dyn_cast<AffineForOp>(node.op)){
      AffineLoopBand band;
      getLoopBandFromOutermost(forOp, band);
      node.tilingFactors.resize(band.size());
      for(unsigned i = 0; i < band.size(); i++){
        node.tilingFactors[i] = solution["u" + std::to_string(node.id) + "_" + std::to_string(i)];
      }
    }else{
      assert(false && "node is not a for op");
    }
  }
  //   auto permutationsInfo = node.permutationInfo;
  //   if(permutationsInfo.size() > 0){
  //     for(auto const& permInfoPair : llvm::enumerate(permutationsInfo)){
  //       auto permIdx = permInfoPair.index();
  //       auto permInfo = permInfoPair.value();
  //       auto key = "x" + std::to_string(nodeId) + "_" + std::to_string(permIdx);
  //       if(solution[key]){
  //         // node.optPermutation = permMapPair.second;
  //         // deep copy
  //         // if(isMinimize)
  //         //   nodePair.second.minPermIdx = permIdx;
  //         // else
  //         //   nodePair.second.maxPermIdx = permIdx;
  //         // for(auto i : permMapPair.second){
  //         //   nodePair.second.optPermutation.push_back(i);
  //         // }
  //       }
  //     }
  //   }
  // }
  // for(auto nodePair : nodes){
  //   LLVM_DEBUG(
  //     llvm::dbgs() << "Node: " << nodePair.second.optPermutation.size() << "\n";
  //   );
  // }
  return true;
}

bool DFG::callCombinedOptimizationSolver(std::string filePath){
  llvm::dbgs() << "Combined solver: ";

  filePath += "_combined.mod";
  std::string command = "ampl " + filePath + "\n";
  // get file name
  // call the command and direct its output to a file
  std::array<char, 128> buffer;
  std::string result;

  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);
  if (!pipe) {
    return false;
  }
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }
  // DenseMap<const char*, bool> solution;
  std::map<std::string, uint64_t> parallelizationSolution;
  std::map<std::string, bool> permutationSolution;
  // go through result checking for pattern x[0-9]+_[0-9]+ = 1 | 0
  std::regex parallelizationPattern("(u[0-9]+_[0-9]+) = (.+)");
  std::regex permutationPattern("(b[0-9]+_[0-9]+) = 1");
  std::regex latencyPattern("latency = ([0-9]+)");
  std::regex totalDSPsPattern("totalDSPs = ([0-9]+)");
  std::smatch match;
  std::istringstream iss(result);
  std::string line;
  while (std::getline(iss, line)) {
    if(std::regex_search(line, match, parallelizationPattern)){
      auto key = match.str(1);
      auto val = std::stoi(match.str(2));
      parallelizationSolution[key] = val;
    }
    if(std::regex_search(line, match, permutationPattern)){
      auto key = match.str(1);
      permutationSolution[key] = true;
    }
    if(std::regex_search(line, match, latencyPattern)){
      auto latency = std::stoi(match.str(1));
      llvm::dbgs() << "Combined Latency: " << latency << "\n";
    }
    if(std::regex_search(line, match, totalDSPsPattern)){
      auto totalDSPs = std::stoi(match.str(1));
      llvm::dbgs() << "Total DSPs: " << totalDSPs << "\n";
    }
  }
  if(parallelizationSolution.size() == 0 || permutationSolution.size() == 0){
    assert(false && "no solution found");
  }
  // for(auto sol : solution){
  //   LLVM_DEBUG(
  //     llvm::dbgs() << sol.first << " = " << sol.second << "\n";
  //   );
  // }
  // for(auto nodePair : nodes){
  //   LLVM_DEBUG(
  //     llvm::dbgs() << "Node: " << nodePair.second.optPermutation.size() << "\n";
  //   );
  // }
  // LLVM_DEBUG(
  //   llvm::dbgs() << "solution size: " << solution.size() << "\n";
  // );
  for(auto& nodePair : nodes){
    auto node = nodePair.second;
    auto nodeId = node.id;
    auto permutationsInfo = node.nodeInfo;
    if(permutationsInfo.size() > 0){
      for(auto const& permInfoPair : llvm::enumerate(permutationsInfo)){
        auto permIdx = permInfoPair.index();
        auto permInfo = permInfoPair.value();
        auto key = "b" + std::to_string(nodeId) + "_" + std::to_string(permIdx);
        if(permutationSolution[key]){
          // deep copy
          nodePair.second.minPermIdx = permIdx;
        }
      }
    }
  }
  for(auto& nodePair : nodes){
    auto& node = nodePair.second;
    if(!node.op)
      continue;
    if(auto forOp = dyn_cast<AffineForOp>(node.op)){
      AffineLoopBand band;
      getLoopBandFromOutermost(forOp, band);
      node.tilingFactors.resize(band.size());
      auto optPermutation = node.nodeInfo[node.minPermIdx].permutation;
      for(unsigned i = 0; i < band.size(); i++){
        node.tilingFactors[optPermutation[i]] = parallelizationSolution["u" + std::to_string(node.id) + "_" + std::to_string(i)];
      }
    }else{
      assert(false && "node is not a for op");
    }
  }
  //   auto permutationsInfo = node.permutationInfo;
  //   if(permutationsInfo.size() > 0){
  //     for(auto const& permInfoPair : llvm::enumerate(permutationsInfo)){
  //       auto permIdx = permInfoPair.index();
  //       auto permInfo = permInfoPair.value();
  //       auto key = "x" + std::to_string(nodeId) + "_" + std::to_string(permIdx);
  //       if(solution[key]){
  //         // node.optPermutation = permMapPair.second;
  //         // deep copy
  //         // if(isMinimize)
  //         //   nodePair.second.minPermIdx = permIdx;
  //         // else
  //         //   nodePair.second.maxPermIdx = permIdx;
  //         // for(auto i : permMapPair.second){
  //         //   nodePair.second.optPermutation.push_back(i);
  //         // }
  //       }
  //     }
  //   }
  // }
  // for(auto nodePair : nodes){
  //   LLVM_DEBUG(
  //     llvm::dbgs() << "Node: " << nodePair.second.optPermutation.size() << "\n";
  //   );
  // }
  return true;
}

bool DFG::callPermutationSolver(std::string filePath, bool isMinimize){
  llvm::dbgs() << "Permutation solver: ";

  if(isMinimize){
    filePath += "_min.mod";
  } else {
    filePath += "_max.mod";
  }
  std::string command = "ampl " + filePath + "\n";
  // get file name
  // call the command and direct its output to a file
  std::array<char, 128> buffer;
  std::string result;

  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);
  if (!pipe) {
    return false;
  }
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }
  // DenseMap<const char*, bool> solution;
  std::map<std::string, bool> solution;
  // go through result checking for pattern x[0-9]+_[0-9]+ = 1 | 0
  std::regex pattern1("b[0-9]+_[0-9]+ = 1");
  std::regex pattern2("b[0-9]+_[0-9]+ = 0");
  std::regex latencyPattern("latency = [0-9]+");
  std::smatch match;
  std::istringstream iss(result);
  std::string line;
  while (std::getline(iss, line)) {
    if(std::regex_search(line, match, pattern1)){
      auto key = match.str().substr(0, match.str().size()-4);
      solution[key] = true;
    }
    if(std::regex_search(line, match, pattern2)){
      auto key = match.str().substr(0, match.str().size()-4);
      solution[key] = false;
    }
    if(std::regex_search(line, match, latencyPattern)){
      auto latency = std::stoi(match.str().substr(10));
      llvm::dbgs() << "latency: " << latency << "\n";
    }
  }
  if(solution.size() == 0){
    assert(false && "no solution found");
  }
  for(auto sol : solution){
    LLVM_DEBUG(
      llvm::dbgs() << sol.first << " = " << sol.second << "\n";
    );
  }
  // for(auto nodePair : nodes){
  //   LLVM_DEBUG(
  //     llvm::dbgs() << "Node: " << nodePair.second.optPermutation.size() << "\n";
  //   );
  // }
  LLVM_DEBUG(
    llvm::dbgs() << "solution size: " << solution.size() << "\n";
  );
  for(auto& nodePair : nodes){
    auto node = nodePair.second;
    auto nodeId = node.id;
    auto permutationsInfo = node.nodeInfo;
    if(permutationsInfo.size() > 0){
      for(auto const& permInfoPair : llvm::enumerate(permutationsInfo)){
        auto permIdx = permInfoPair.index();
        auto permInfo = permInfoPair.value();
        auto key = "b" + std::to_string(nodeId) + "_" + std::to_string(permIdx);
        if(solution[key]){
          // node.optPermutation = permMapPair.second;
          // deep copy
          if(isMinimize)
            nodePair.second.minPermIdx = permIdx;
          else
            nodePair.second.maxPermIdx = permIdx;
          // for(auto i : permMapPair.second){
          //   nodePair.second.optPermutation.push_back(i);
          // }
        }
      }
    }
  }
  // for(auto nodePair : nodes){
  //   LLVM_DEBUG(
  //     llvm::dbgs() << "Node: " << nodePair.second.optPermutation.size() << "\n";
  //   );
  // }
  return true;
}

bool DFG::applyNodePermutations(DFG::PermutationType permType){
  for(auto& nodePair : nodes){
    auto& node = nodePair.second;
    auto &forOp = node.op;
    if(node.id == 10000)
      continue;
    // llvm::dbgs() << "Applying permutation for node: " << node.id << "\n";
    if(auto forOp = dyn_cast<AffineForOp>(node.op)){
      AffineLoopBand band;
      getLoopBandFromOutermost(forOp, band);
      LLVM_DEBUG(
        llvm::dbgs() << "band size: " << band.size() << "\n";
      );
      auto permIdx = permType == DFG::PermutationType::Minimize ? node.minPermIdx : node.maxPermIdx;
      auto& permMap = node.nodeInfo[permIdx].permutation;
      if(permMap.size() > 0){
        auto newRoot = band[permuteLoops(band, permMap)];
        band.clear();
        getLoopBandFromOutermost(newRoot, band);
        // set the new loop band
        node.op = band[0];
        // setLoopInfo(newRoot, 1, 1, node.permutationInfo[permIdx].II);
      }
    } else {
      LLVM_DEBUG(
        llvm::dbgs() << "node is not a for op\n";
      );
    }
  }
  return true;
}

/// Apply the specified array partition factors and kinds.
static bool applyArrayPartition(Value array, SmallVector<unsigned> &factors,
                                  SmallVector<dataflow::PartitionKind> &kinds,
                                  bool updateFuncSignature) {
  auto builder = Builder(array.getContext());
  auto arrayType = array.getType().dyn_cast<MemRefType>();
  if (!arrayType || !arrayType.hasStaticShape()){
    return false;
  }

  if(((int64_t)factors.size() == (arrayType.getRank()-1) || (int64_t)kinds.size() != (arrayType.getRank()-1)) &&
      arrayType.getShape()[0] == 1
  ){
    factors.insert(factors.begin(), 1);
    kinds.insert(kinds.begin(), dataflow::PartitionKind::CYCLIC);
  }

  if(arrayType.getRank() != factors.size() || arrayType.getRank() != kinds.size())
    return false;

  // Walk through each dimension of the current memory.
  SmallVector<AffineExpr, 4> partitionIndices;
  SmallVector<AffineExpr, 4> addressIndices;
  // llvm::dbgs() << "Array type: " << arrayType << "\n";
  // llvm::dbgs() << "Factors size: " << factors.size() << "\n";
  // llvm::dbgs() << "Kinds size: " << kinds.size() << "\n";
  for (int64_t dim = 0; dim < arrayType.getRank(); ++dim) {
    auto kind = kinds[dim];
    auto factor = factors[dim];

    if (kind == PartitionKind::CYCLIC) {
      partitionIndices.push_back(builder.getAffineDimExpr(dim) % factor);
      addressIndices.push_back(builder.getAffineDimExpr(dim).floorDiv(factor));

    } else if (kind == PartitionKind::BLOCK) {
      auto blockFactor = (arrayType.getShape()[dim] + factor - 1) / factor;
      partitionIndices.push_back(
          builder.getAffineDimExpr(dim).floorDiv(blockFactor));
      addressIndices.push_back(builder.getAffineDimExpr(dim) % blockFactor);

    } else {
      partitionIndices.push_back(builder.getAffineConstantExpr(0));
      addressIndices.push_back(builder.getAffineDimExpr(dim));
    }
  }

  // Construct new layout map.
  partitionIndices.append(addressIndices.begin(), addressIndices.end());
  auto layoutMap = AffineMap::get(arrayType.getRank(), 0, partitionIndices,
                                  builder.getContext());

  // Construct new array type.
  auto newType =
      MemRefType::get(arrayType.getShape(), arrayType.getElementType(),
                      layoutMap, arrayType.getMemorySpace());

  // Set new type.
  array.setType(newType);

  if (updateFuncSignature)
    if (auto func =
            dyn_cast<func::FuncOp>(array.getParentBlock()->getParentOp())) {
      // Align function type with entry block argument types only if the array
      // is defined as an argument of the function.
      if (!array.getDefiningOp()) {
        auto resultTypes = func.front().getTerminator()->getOperandTypes();
        auto inputTypes = func.front().getArgumentTypes();
        func.setType(builder.getFunctionType(inputTypes, resultTypes));
      }
    }
  return true;
}

bool DFG::applyNodeParallelization(){
  // // array partitioning
  // for(auto memRefPair : memRefFactorVars){
  //   auto memRef = memRefPair.first;
  //   auto loopVars = memRefPair.second;
  //   SmallVector<unsigned, 6> factors;
  //   SmallVector<dataflow::PartitionKind, 6> kinds;
  //   for(auto loopVar : loopVars){
  //     auto factor = solution[loopVar];
  //     factors.push_back(factor);
  //     kinds.push_back(dataflow::PartitionKind::CYCLIC);
  //   }
  //   applyArrayPartition(memRef, factors, kinds, true);
  // }
  for(auto& nodePair : nodes){
    auto& node = nodePair.second;
    if(node.id == 10000)
      continue;
    if(auto forOp = dyn_cast<AffineForOp>(node.op)){
      AffineLoopBand band;
      getLoopBandFromOutermost(forOp, band);
      auto innerMostLoop = band.back();
      SmallVector<AffineLoadOp, 3> loads;
      SmallVector<AffineStoreOp, 3> stores;
      forOp->walk([&](AffineLoadOp loadOp) {
        loads.push_back(loadOp);
      });
      forOp->walk([&](AffineStoreOp storeOp) {
        stores.push_back(storeOp);
      });
      for(auto load : loads){
        SmallVector<unsigned> factors;
        SmallVector<dataflow::PartitionKind> kinds;
        auto memRef = load.getMemRef();
        bool isLocal = true;
        for(auto user : memRef.getUsers()){
          if(user->getParentOfType<AffineForOp>() != innerMostLoop){
            isLocal = false;
            break;
          }
        }
        DenseMap<unsigned, unsigned> indicesToLoop;
        // if(isLocal){
          for(auto iv : load.getMapOperands()){
            auto srcLoop = dyn_cast_if_present<AffineForOp>(iv.getParentBlock()->getParentOp());
            if(!srcLoop) {
              assert(false && "src must be inside a loop");
            }
            for(auto loop : llvm::enumerate(band)){
              if(loop.value() == srcLoop){
                unsigned factor = node.tilingFactors[loop.index()];
                factors.push_back(factor);
                kinds.push_back(dataflow::PartitionKind::CYCLIC);
                break;
              }
            }
          // }
        }
        if(!applyArrayPartition(memRef, factors, kinds, true)){
          // llvm::dbgs() << "Failed to apply array partition for load\n";
          // memRef.dump();
          // llvm::dbgs() << "Factors size: " << factors.size() << "\n";
          // llvm::dbgs() << "Kinds size: " << kinds.size() << "\n";
        }
      }
      for(auto store : stores){
        SmallVector<unsigned> factors;
        SmallVector<dataflow::PartitionKind> kinds;
        auto memRef = store.getMemRef();
        bool isLocal = true;
        for(auto user : memRef.getUsers()){
          if(user->getParentOfType<AffineForOp>() != innerMostLoop){
            isLocal = false;
            break;
          }
        }
        DenseMap<unsigned, unsigned> indicesToLoop;
        // if(isLocal){
          for(auto iv : store.getMapOperands()){
            auto srcLoop = dyn_cast_if_present<AffineForOp>(iv.getParentBlock()->getParentOp());
            if(!srcLoop) {
              assert(false && "src must be inside a loop");
            }
            for(auto loop : llvm::enumerate(band)){
              if(loop.value() == srcLoop){
                unsigned factor = node.tilingFactors[loop.index()];
                factors.push_back(factor);
                kinds.push_back(dataflow::PartitionKind::CYCLIC);
                break;
              }
            }
          // }
        }
        if(!applyArrayPartition(memRef, factors, kinds, true)){
          // llvm::dbgs() << "Failed to apply array partition for store\n";
          // memRef.dump();
          // llvm::dbgs() << "Factors size: " << factors.size() << "\n";
          // llvm::dbgs() << "Kinds size: " << kinds.size() << "\n";
        }      
      }
      
    }
    // for(auto loadPair : parallelizationInfo.loadsMap){
    //   auto loadOp = loadPair.first;
    //   loadOp.dump();
    //   // auto memRef = loadOp.getMemRef();
    //   // auto loadInfo = loadPair.second;
    //   // SmallVector<unsigned, 3> factors;
    //   // SmallVector<dataflow::PartitionKind, 3> kinds;
    //   // for(auto dim : memRef.getType().cast<MemRefType>().getShape()){
    //   //   auto factor = 2;//node.tilingFactors[loadInfo.indexLoopInfo[dim].second];
    //   //   llvm::dbgs() << "factor: " << factor << "\n";
    //   //   factors.push_back(factor);
    //   //   kinds.push_back(dataflow::PartitionKind::CYCLIC);
    //   // }
    //   // applyArrayPartition(memRef, factors, kinds, true);
    // }
  }
  for(auto& nodePair : nodes){
    auto& node = nodePair.second;
    auto &forOp = node.op;
    if(node.id == 10000)
      continue;
    // llvm::dbgs() << "Applying permutation for node: " << node.id << "\n";
    if(auto forOp = dyn_cast<AffineForOp>(node.op)){
      AffineLoopBand band;
      getLoopBandFromOutermost(forOp, band);
      LLVM_DEBUG(
        llvm::dbgs() << "band size: " << band.size() << "\n";
      );
      SmallVector<AffineForOp, 6> tiledNest;
      auto bandSize = band.size();
      if (failed(tilePerfectlyNested(band, node.tilingFactors, &tiledNest))) {
        // An empty band always succeeds.
        assert(!band.empty() && "guaranteed to succeed on empty bands");
        LLVM_DEBUG(band.front()->emitRemark("loop tiling failed!\n"));
        continue;
      }
      setLoopDirective(tiledNest[bandSize - 1], true, 1, false, false);
    } else {
      LLVM_DEBUG(
        llvm::dbgs() << "node is not a for op\n";
      );
    }
  }
  return true;
}

bool DFG::applyCombinedOptimization(){
  for(auto& nodePair : nodes){
    auto& node = nodePair.second;
    auto &forOp = node.op;
    if(node.id == 10000)
      continue;
    // llvm::dbgs() << "Applying permutation for node: " << node.id << "\n";
    if(auto forOp = dyn_cast<AffineForOp>(node.op)){
      AffineLoopBand band;
      getLoopBandFromOutermost(forOp, band);
      LLVM_DEBUG(
        llvm::dbgs() << "band size: " << band.size() << "\n";
      );
      auto permIdx = node.minPermIdx;

      auto& permMap = node.nodeInfo[permIdx].permutation;
      if(permMap.size() > 0){
        auto newRoot = band[permuteLoops(band, permMap)];
        band.clear();
        getLoopBandFromOutermost(newRoot, band);
        // set the new loop band
        node.op = band[0];
        setLoopInfo(newRoot, 1, 1, node.nodeInfo[permIdx].II);
      }
    } else {
      LLVM_DEBUG(
        llvm::dbgs() << "node is not a for op\n";
      );
    }
  }
  for(auto& nodePair : nodes){
    auto& node = nodePair.second;
    if(node.id == 10000)
      continue;
    if(auto forOp = dyn_cast<AffineForOp>(node.op)){
      AffineLoopBand band;
      getLoopBandFromOutermost(forOp, band);
      auto innerMostLoop = band.back();
      SmallVector<AffineLoadOp, 3> loads;
      SmallVector<AffineStoreOp, 3> stores;
      forOp->walk([&](AffineLoadOp loadOp) {
        loads.push_back(loadOp);
      });
      forOp->walk([&](AffineStoreOp storeOp) {
        stores.push_back(storeOp);
      });
      for(auto load : loads){
        SmallVector<unsigned> factors;
        SmallVector<dataflow::PartitionKind> kinds;
        auto memRef = load.getMemRef();
        bool isLocal = true;
        for(auto user : memRef.getUsers()){
          if(user->getParentOfType<AffineForOp>() != innerMostLoop){
            isLocal = false;
            break;
          }
        }
        DenseMap<unsigned, unsigned> indicesToLoop;
        // if(isLocal){
          for(auto iv : load.getMapOperands()){
            auto srcLoop = dyn_cast_if_present<AffineForOp>(iv.getParentBlock()->getParentOp());
            if(!srcLoop) {
              assert(false && "src must be inside a loop");
            }
            for(auto loop : llvm::enumerate(band)){
              if(loop.value() == srcLoop){
                unsigned factor = node.tilingFactors[loop.index()];
                factors.push_back(factor);
                kinds.push_back(dataflow::PartitionKind::CYCLIC);
                break;
              }
            }
          }
        // }
        if(!applyArrayPartition(memRef, factors, kinds, true)){
          // llvm::dbgs() << "Failed to apply array partition for load\n";
          // memRef.dump();
          // llvm::dbgs() << "Factors size: " << factors.size() << "\n";
          // llvm::dbgs() << "Kinds size: " << kinds.size() << "\n";
        }         }
      for(auto store : stores){
        SmallVector<unsigned> factors;
        SmallVector<dataflow::PartitionKind> kinds;
        auto memRef = store.getMemRef();
        bool isLocal = true;
        for(auto user : memRef.getUsers()){
          if(user->getParentOfType<AffineForOp>() != innerMostLoop){
            isLocal = false;
            break;
          }
        }
        DenseMap<unsigned, unsigned> indicesToLoop;
        // if(isLocal){
          for(auto iv : store.getMapOperands()){
            auto srcLoop = dyn_cast_if_present<AffineForOp>(iv.getParentBlock()->getParentOp());
            if(!srcLoop) {
              assert(false && "src must be inside a loop");
            }
            for(auto loop : llvm::enumerate(band)){
              if(loop.value() == srcLoop){
                unsigned factor = node.tilingFactors[loop.index()];
                factors.push_back(factor);
                kinds.push_back(dataflow::PartitionKind::CYCLIC);
                break;
              }
            }
          }
        // }
        if(!applyArrayPartition(memRef, factors, kinds, true)){
          // llvm::dbgs() << "Failed to apply array partition for store\n";
          // memRef.dump();
          // llvm::dbgs() << "Factors size: " << factors.size() << "\n";
          // llvm::dbgs() << "Kinds size: " << kinds.size() << "\n";
        }   
      }
    }
    // for(auto loadPair : parallelizationInfo.loadsMap){
    //   auto loadOp = loadPair.first;
    //   loadOp.dump();
    //   // auto memRef = loadOp.getMemRef();
    //   // auto loadInfo = loadPair.second;
    //   // SmallVector<unsigned, 3> factors;
    //   // SmallVector<dataflow::PartitionKind, 3> kinds;
    //   // for(auto dim : memRef.getType().cast<MemRefType>().getShape()){
    //   //   auto factor = 2;//node.tilingFactors[loadInfo.indexLoopInfo[dim].second];
    //   //   llvm::dbgs() << "factor: " << factor << "\n";
    //   //   factors.push_back(factor);
    //   //   kinds.push_back(dataflow::PartitionKind::CYCLIC);
    //   // }
    //   // applyArrayPartition(memRef, factors, kinds, true);
    // }
  }
  for(auto& nodePair : nodes){
    auto& node = nodePair.second;
    auto &forOp = node.op;
    if(node.id == 10000)
      continue;
    // llvm::dbgs() << "Applying permutation for node: " << node.id << "\n";
    if(auto forOp = dyn_cast<AffineForOp>(node.op)){
      AffineLoopBand band;
      getLoopBandFromOutermost(forOp, band);
      LLVM_DEBUG(
        llvm::dbgs() << "band size: " << band.size() << "\n";
      );
      SmallVector<AffineForOp, 6> tiledNest;
      auto bandSize = band.size();
      if (failed(tilePerfectlyNested(band, node.tilingFactors, &tiledNest))) {
        // An empty band always succeeds.
        assert(!band.empty() && "guaranteed to succeed on empty bands");
        LLVM_DEBUG(band.front()->emitRemark("loop tiling failed!\n"));
        continue;
      }
      setLoopDirective(tiledNest[bandSize - 1], true, 1, false, false);
    } else {
      LLVM_DEBUG(
        llvm::dbgs() << "node is not a for op\n";
      );
    }
  }
  return true;
}

bool DFG::findRandomSolution(){
  populateNodeInfo(true);
  createRootNode();
  auto seed = std::chrono::system_clock::now().time_since_epoch().count();
  srand(seed);
  unsigned varIdx = 0;
  for(auto& nodePair : nodes){
    auto id = nodePair.first;
    auto& node = nodePair.second;
    if(!node.op){
      continue;
    }
    auto forOp = dyn_cast<AffineForOp>(node.op);
    AffineLoopBand band;
    getLoopBandFromOutermost(forOp, band);
    Node* currNode = getNode(id);
    // llvm::dbgs() << "node: " << id << " trip count " << getLoopNestIterations(band) << "\n";
    // num permutations = factorial of the number of loops in the band
    auto numPerms = factorial(band.size());
    // choose random permutation between varIdx and varIdx + numPerms
    currNode->minPermIdx = varIdx + (rand() % numPerms);
    // llvm::dbgs() << "node: " << id << " random permutation idx: " << currNode->minPermIdx << "\n";
    varIdx += numPerms;
  }
  return true;
}

bool DFG::findReversePermutations(){
  populateNodeInfo(true);
  createRootNode();
  auto seed = std::chrono::system_clock::now().time_since_epoch().count();
  srand(seed);
  unsigned varIdx = 0;
  for(auto& nodePair : nodes){
    auto id = nodePair.first;
    auto& node = nodePair.second;
    if(!node.op){
      continue;
    }
    auto forOp = dyn_cast<AffineForOp>(node.op);
    AffineLoopBand band;
    getLoopBandFromOutermost(forOp, band);
    Node* currNode = getNode(id);
    // llvm::dbgs() << "node: " << id << " trip count " << getLoopNestIterations(band) << "\n";
    // num permutations = factorial of the number of loops in the band
    auto numPerms = factorial(band.size());
    // choose random permutation between varIdx and varIdx + numPerms
    currNode->minPermIdx = numPerms - 1;
    // llvm::dbgs() << "node: " << id << " random permutation idx: " << currNode->minPermIdx << "\n";
    varIdx += numPerms;
  }
  return true;
}

// Returns the graph node for 'id'.
Node *DFG::getNode(unsigned id) {
  auto it = nodes.find(id);
  assert(it != nodes.end());
  return &it->second;
}

// Returns the graph node for 'forOp'.
Node *DFG::getForOpNode(AffineForOp forOp) {
  for (auto &idAndNode : nodes)
    if (idAndNode.second.op == forOp)
      return &idAndNode.second;
  return nullptr;
}

// Adds a node with 'op' to the graph and returns its unique identifier.
unsigned DFG::addNode(Operation *op) {
  Node node(nextNodeId++, op);
  nodes.insert({node.id, node});
  return node.id;
}

// Remove node 'id' (and its associated edges) from graph.
void DFG::removeNode(unsigned id) {
  // Remove each edge in 'inEdges[id]'.
  if (inEdges.count(id) > 0) {
    SmallVector<Edge, 2> oldInEdges = inEdges[id];
    for (auto &inEdge : oldInEdges) {
      removeEdge(inEdge.id, id, inEdge.value);
    }
  }
  // Remove each edge in 'outEdges[id]'.
  if (outEdges.count(id) > 0) {
    SmallVector<Edge, 2> oldOutEdges = outEdges[id];
    for (auto &outEdge : oldOutEdges) {
      removeEdge(id, outEdge.id, outEdge.value);
    }
  }
  // Erase remaining node state.
  inEdges.erase(id);
  outEdges.erase(id);
  nodes.erase(id);
}

// // Returns true if node 'id' writes to any memref which escapes (or is an
// // argument to) the block. Returns false otherwise.
// bool DFG::writesToLiveInOrEscapingMemrefs(unsigned id) {
//   Node *node = getNode(id);
//   for (auto *storeOpInst : node->stores) {
//     auto memref = cast<AffineWriteOpInterface>(storeOpInst).getMemRef();
//     auto *op = memref.getDefiningOp();
//     // Return true if 'memref' is a block argument.
//     if (!op)
//       return true;
//     // Return true if any use of 'memref' does not deference it in an affine
//     // way.
//     for (auto *user : memref.getUsers())
//       if (!isa<AffineMapAccessInterface>(*user))
//         return true;
//   }
//   return false;
// }

// Returns true iff there is an edge from node 'srcId' to node 'dstId' which
// is for 'value' if non-null, or for any value otherwise. Returns false
// otherwise.
bool DFG::hasEdge(unsigned srcId, unsigned dstId,
                                    Value value) {
  if (outEdges.count(srcId) == 0 || inEdges.count(dstId) == 0) {
    return false;
  }
  bool hasOutEdge = llvm::any_of(outEdges[srcId], [=](Edge &edge) {
    return edge.id == dstId && (!value || edge.value == value);
  });
  bool hasInEdge = llvm::any_of(inEdges[dstId], [=](Edge &edge) {
    return edge.id == srcId && (!value || edge.value == value);
  });
  return hasOutEdge && hasInEdge;
}

// Adds an edge from node 'srcId' to node 'dstId' for 'value'.
void DFG::addEdge(unsigned srcId, unsigned dstId,
                                    Value value) {
  if (!hasEdge(srcId, dstId, value)) {
    outEdges[srcId].push_back({dstId, value});
    inEdges[dstId].push_back({srcId, value});
    if (isa<MemRefType>(value.getType()))
      memrefEdgeCount[value]++;
  }
}

// Adds an edge from node 'srcId' to node 'dstId' for 'value'.
void DFG::addEdge(
  unsigned srcId, unsigned dstId, 
  AffineStoreOp srcOp, AffineLoadOp dstOp,
  Value value
) {
  if (!hasEdge(srcId, dstId, value)) {
    outEdges[srcId].push_back({dstId, value, srcOp, dstOp});
    inEdges[dstId].push_back({srcId, value, srcOp, dstOp});
    if (isa<MemRefType>(value.getType()))
      memrefEdgeCount[value]++;
  }
}

// Removes an edge from node 'srcId' to node 'dstId' for 'value'.
void DFG::removeEdge(unsigned srcId, unsigned dstId,
                                       Value value) {
  assert(inEdges.count(dstId) > 0);
  assert(outEdges.count(srcId) > 0);
  if (isa<MemRefType>(value.getType())) {
    assert(memrefEdgeCount.count(value) > 0);
    memrefEdgeCount[value]--;
  }
  // Remove 'srcId' from 'inEdges[dstId]'.
  for (auto *it = inEdges[dstId].begin(); it != inEdges[dstId].end(); ++it) {
    if ((*it).id == srcId && (*it).value == value) {
      inEdges[dstId].erase(it);
      break;
    }
  }
  // Remove 'dstId' from 'outEdges[srcId]'.
  for (auto *it = outEdges[srcId].begin(); it != outEdges[srcId].end(); ++it) {
    if ((*it).id == dstId && (*it).value == value) {
      outEdges[srcId].erase(it);
      break;
    }
  }
}

// Returns true if there is a path in the dependence graph from node 'srcId'
// to node 'dstId'. Returns false otherwise. `srcId`, `dstId`, and the
// operations that the edges connected are expected to be from the same block.
bool DFG::hasDependencePath(unsigned srcId, unsigned dstId) {
  // Worklist state is: <node-id, next-output-edge-index-to-visit>
  SmallVector<std::pair<unsigned, unsigned>, 4> worklist;
  worklist.push_back({srcId, 0});
  Operation *dstOp = getNode(dstId)->op;
  // Run DFS traversal to see if 'dstId' is reachable from 'srcId'.
  while (!worklist.empty()) {
    auto &idAndIndex = worklist.back();
    // Return true if we have reached 'dstId'.
    if (idAndIndex.first == dstId)
      return true;
    // Pop and continue if node has no out edges, or if all out edges have
    // already been visited.
    if (outEdges.count(idAndIndex.first) == 0 ||
        idAndIndex.second == outEdges[idAndIndex.first].size()) {
      worklist.pop_back();
      continue;
    }
    // Get graph edge to traverse.
    Edge edge = outEdges[idAndIndex.first][idAndIndex.second];
    // Increment next output edge index for 'idAndIndex'.
    ++idAndIndex.second;
    // Add node at 'edge.id' to the worklist. We don't need to consider
    // nodes that are "after" dstId in the containing block; one can't have a
    // path to `dstId` from any of those nodes.
    bool afterDst = dstOp->isBeforeInBlock(getNode(edge.id)->op);
    if (!afterDst && edge.id != idAndIndex.first)
      worklist.push_back({edge.id, 0});
  }
  return false;
}

// // Returns the input edge count for node 'id' and 'memref' from src nodes
// // which access 'memref' with a store operation.
// unsigned DFG::getIncomingMemRefAccesses(unsigned id,
//                                                           Value memref) {
//   unsigned inEdgeCount = 0;
//   if (inEdges.count(id) > 0)
//     for (auto &inEdge : inEdges[id])
//       if (inEdge.value == memref) {
//         Node *srcNode = getNode(inEdge.id);
//         // Only count in edges from 'srcNode' if 'srcNode' accesses 'memref'
//         if (srcNode->getStoreOpCount(memref) > 0)
//           ++inEdgeCount;
//       }
//   return inEdgeCount;
// }

// Returns the output edge count for node 'id' and 'memref' (if non-null),
// otherwise returns the total output edge count from node 'id'.
unsigned DFG::getOutEdgeCount(unsigned id, Value memref) {
  unsigned outEdgeCount = 0;
  if (outEdges.count(id) > 0)
    for (auto &outEdge : outEdges[id])
      if (!memref || outEdge.value == memref)
        ++outEdgeCount;
  return outEdgeCount;
}

/// Return all nodes which define SSA values used in node 'id'.
void DFG::gatherDefiningNodes(
    unsigned id, DenseSet<unsigned> &definingNodes) {
  for (DFG::Edge edge : inEdges[id])
    // By definition of edge, if the edge value is a non-memref value,
    // then the dependence is between a graph node which defines an SSA value
    // and another graph node which uses the SSA value.
    if (!isa<MemRefType>(edge.value.getType()))
      definingNodes.insert(edge.id);
}

// Computes and returns an insertion point operation, before which the
// the fused <srcId, dstId> loop nest can be inserted while preserving
// dependences. Returns nullptr if no such insertion point is found.
Operation *
DFG::getFusedLoopNestInsertionPoint(unsigned srcId,
                                                      unsigned dstId) {
  if (outEdges.count(srcId) == 0)
    return getNode(dstId)->op;

  // Skip if there is any defining node of 'dstId' that depends on 'srcId'.
  DenseSet<unsigned> definingNodes;
  gatherDefiningNodes(dstId, definingNodes);
  if (llvm::any_of(definingNodes,
                   [&](unsigned id) { return hasDependencePath(srcId, id); })) {
    LLVM_DEBUG(llvm::dbgs()
               << "Can't fuse: a defining op with a user in the dst "
                  "loop has dependence from the src loop\n");
    return nullptr;
  }

  // Build set of insts in range (srcId, dstId) which depend on 'srcId'.
  SmallPtrSet<Operation *, 2> srcDepInsts;
  for (auto &outEdge : outEdges[srcId])
    if (outEdge.id != dstId)
      srcDepInsts.insert(getNode(outEdge.id)->op);

  // Build set of insts in range (srcId, dstId) on which 'dstId' depends.
  SmallPtrSet<Operation *, 2> dstDepInsts;
  for (auto &inEdge : inEdges[dstId])
    if (inEdge.id != srcId)
      dstDepInsts.insert(getNode(inEdge.id)->op);

  Operation *srcNodeInst = getNode(srcId)->op;
  Operation *dstNodeInst = getNode(dstId)->op;

  // Computing insertion point:
  // *) Walk all operation positions in Block operation list in the
  //    range (src, dst). For each operation 'op' visited in this search:
  //   *) Store in 'firstSrcDepPos' the first position where 'op' has a
  //      dependence edge from 'srcNode'.
  //   *) Store in 'lastDstDepPost' the last position where 'op' has a
  //      dependence edge to 'dstNode'.
  // *) Compare 'firstSrcDepPos' and 'lastDstDepPost' to determine the
  //    operation insertion point (or return null pointer if no such
  //    insertion point exists: 'firstSrcDepPos' <= 'lastDstDepPos').
  SmallVector<Operation *, 2> depInsts;
  std::optional<unsigned> firstSrcDepPos;
  std::optional<unsigned> lastDstDepPos;
  unsigned pos = 0;
  for (Block::iterator it = std::next(Block::iterator(srcNodeInst));
       it != Block::iterator(dstNodeInst); ++it) {
    Operation *op = &(*it);
    if (srcDepInsts.count(op) > 0 && firstSrcDepPos == std::nullopt)
      firstSrcDepPos = pos;
    if (dstDepInsts.count(op) > 0)
      lastDstDepPos = pos;
    depInsts.push_back(op);
    ++pos;
  }

  if (firstSrcDepPos.has_value()) {
    if (lastDstDepPos.has_value()) {
      if (*firstSrcDepPos <= *lastDstDepPos) {
        // No valid insertion point exists which preserves dependences.
        return nullptr;
      }
    }
    // Return the insertion point at 'firstSrcDepPos'.
    return depInsts[*firstSrcDepPos];
  }
  // No dependence targets in range (or only dst deps in range), return
  // 'dstNodInst' insertion point.
  return dstNodeInst;
}

// Updates edge mappings from node 'srcId' to node 'dstId' after fusing them,
// taking into account that:
//   *) if 'removeSrcId' is true, 'srcId' will be removed after fusion,
//   *) memrefs in 'privateMemRefs' has been replaced in node at 'dstId' by a
//      private memref.
void DFG::updateEdges(unsigned srcId, unsigned dstId,
                                        const DenseSet<Value> &privateMemRefs,
                                        bool removeSrcId) {
  // For each edge in 'inEdges[srcId]': add new edge remapping to 'dstId'.
  if (inEdges.count(srcId) > 0) {
    SmallVector<Edge, 2> oldInEdges = inEdges[srcId];
    for (auto &inEdge : oldInEdges) {
      // Add edge from 'inEdge.id' to 'dstId' if it's not a private memref.
      if (privateMemRefs.count(inEdge.value) == 0)
        addEdge(inEdge.id, dstId, inEdge.value);
    }
  }
  // For each edge in 'outEdges[srcId]': remove edge from 'srcId' to 'dstId'.
  // If 'srcId' is going to be removed, remap all the out edges to 'dstId'.
  if (outEdges.count(srcId) > 0) {
    SmallVector<Edge, 2> oldOutEdges = outEdges[srcId];
    for (auto &outEdge : oldOutEdges) {
      // Remove any out edges from 'srcId' to 'dstId' across memrefs.
      if (outEdge.id == dstId)
        removeEdge(srcId, outEdge.id, outEdge.value);
      else if (removeSrcId) {
        addEdge(dstId, outEdge.id, outEdge.value);
        removeEdge(srcId, outEdge.id, outEdge.value);
      }
    }
  }
  // Remove any edges in 'inEdges[dstId]' on 'oldMemRef' (which is being
  // replaced by a private memref). These edges could come from nodes
  // other than 'srcId' which were removed in the previous step.
  if (inEdges.count(dstId) > 0 && !privateMemRefs.empty()) {
    SmallVector<Edge, 2> oldInEdges = inEdges[dstId];
    for (auto &inEdge : oldInEdges)
      if (privateMemRefs.count(inEdge.value) > 0)
        removeEdge(inEdge.id, dstId, inEdge.value);
  }
}

// Update edge mappings for nodes 'sibId' and 'dstId' to reflect fusion
// of sibling node 'sibId' into node 'dstId'.
void DFG::updateEdges(unsigned sibId, unsigned dstId) {
  // For each edge in 'inEdges[sibId]':
  // *) Add new edge from source node 'inEdge.id' to 'dstNode'.
  // *) Remove edge from source node 'inEdge.id' to 'sibNode'.
  if (inEdges.count(sibId) > 0) {
    SmallVector<Edge, 2> oldInEdges = inEdges[sibId];
    for (auto &inEdge : oldInEdges) {
      addEdge(inEdge.id, dstId, inEdge.value);
      removeEdge(inEdge.id, sibId, inEdge.value);
    }
  }

  // For each edge in 'outEdges[sibId]' to node 'id'
  // *) Add new edge from 'dstId' to 'outEdge.id'.
  // *) Remove edge from 'sibId' to 'outEdge.id'.
  if (outEdges.count(sibId) > 0) {
    SmallVector<Edge, 2> oldOutEdges = outEdges[sibId];
    for (auto &outEdge : oldOutEdges) {
      addEdge(dstId, outEdge.id, outEdge.value);
      removeEdge(sibId, outEdge.id, outEdge.value);
    }
  }
}

// // Adds ops in 'loads' and 'stores' to node at 'id'.
// void DFG::addToNode(
//     unsigned id, const SmallVectorImpl<Operation *> &loads,
//     const SmallVectorImpl<Operation *> &stores) {
//   Node *node = getNode(id);
//   llvm::append_range(node->loads, loads);
//   llvm::append_range(node->stores, stores);
// }

void DFG::print(raw_ostream &os) const {
  os << "\nMemRefDependenceGraph\n";
  os << "\nNodes:\n";
  for (const auto &idAndNode : nodes) {
    os << "Node: " << idAndNode.first << "\n";
    auto it = inEdges.find(idAndNode.first);
    if (it != inEdges.end()) {
      for (const auto &e : it->second)
        os << "  InEdge: " << e.id << " " << e.value << "\n";
    }
    it = outEdges.find(idAndNode.first);
    if (it != outEdges.end()) {
      for (const auto &e : it->second)
        os << "  OutEdge: " << e.id << " " << e.value << "\n";
    }
  }
}

void DFG::printAsDot(raw_ostream &os) const {
  os << "digraph {\n";
  for (const auto &idAndNode : nodes) {
    auto color = idAndNode.second.isReduction ? "blue" : "black";
    // 
    os << "  " << idAndNode.first << " [label=\"" << idAndNode.first << "\", color=" << color;
    // idAndNode.second.op->print(os);
    os << "];\n";
    auto it = inEdges.find(idAndNode.first);
    if (it != inEdges.end()) {
      for (const auto &e : it->second){
        // if edge is memref make color red, if fifo make blue
        if(e.value == nullptr)
          os << "  " << e.id << " -> " << idAndNode.first << " [label=\"\", color=black];\n";
        else if (isa<MemRefType>(e.value.getType()))
          os << "  " << e.id << " -> " << idAndNode.first << " [label=\"\", color=red];\n";
        else if (isa<StreamType>(e.value.getType()))
          os << "  " << e.id << " -> " << idAndNode.first << " [label=\"\", color=blue];\n";
        else if (isa<RankedTensorType>(e.value.getType()))
          os << "  " << e.id << " -> " << idAndNode.first << " [label=\"\", color=green, penwidth=2];\n";
        else
          assert(false && "unknown edge type");

      }
        // os << "  " << e.id << " -> " << idAndNode.first << " [label=\"" << e.value << "\"];\n";
    }
    // it = outEdges.find(idAndNode.first);
    // if (it != outEdges.end()) {
    //   for (const auto &e : it->second){
    //     // if edge is memref make color red, if fifo make blue
    //     if (isa<MemRefType>(e.value.getType()))
    //       os << "  " << idAndNode.first << " -> " << e.id << " [label=\"" << e.value << "\", color=red];\n";
    //     else
    //       os << "  " << idAndNode.first << " -> " << e.id << " [label=\"" << e.value << "\", color=blue];\n";
    //   }
    //     // os << "  " << idAndNode.first << " -> " << e.id << " [label=\"" << e.value << "\"];\n";
    // }
  }
  os << "}\n";
}
