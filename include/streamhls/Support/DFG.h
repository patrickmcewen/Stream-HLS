#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IntegerSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Analysis/Presburger/PWMAFunction.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Analysis/Presburger/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallBitVector.h"
#include <memory>
#include <optional>
#include <string>
#include <fstream>

#include <queue>
#include <set>
#include <stack>

#include "streamhls/Dialect/Dataflow/Dataflow.h"


namespace mlir { namespace streamhls {
using namespace affine;
using namespace dataflow;
// LoopNestStateCollector walks loop nests and collects load and store
// operations, and whether or not a region holding op other than ForOp and IfOp
// was encountered in the loop nest.
struct LoopNestInfo {
  SmallVector<AffineForOp, 4> forOps;
  SmallVector<Operation *, 4> loadOpInsts;
  SmallVector<Operation *, 4> storeOpInsts;
  SmallVector<Operation *, 4> fifoReadOps;
  SmallVector<Operation *, 4> fifoWriteOps;
  SmallVector<memref::AllocOp, 4> allocOps;
  bool hasNonAffineRegionOp = false;

  // Collects load and store operations, and whether or not a region holding op
  // other than ForOp and IfOp was encountered in the loop nest.
  void collect(Operation *opToWalk);
};


// structure to represent dataflow graph
// Nodes are:
// Store/Load operations
// FIFO read/write operations
// Edges are:
// buffers or FIFOs
struct DFG {
public:
  // Node represents a node in the graph. A Node is either an entire loop nest
  // rooted at the top level which contains loads/stores, or a top level
  // load/store.
  struct AccessStats {
    // presburger::IntegerRelation* timeIndexRelation;
    // presburger::IntegerRelation* timeIndexRelevantRelation;
    AffineMap timeFunction;
    AffineMap accessMap;
    uint64_t firstElementTime;
    uint64_t lastElementTime;
  };

  struct ParallelAccessStats{
    AffineMap accessMap;
    std::string firstElementTime;
    std::string lastElementTime;
    // DenseMap<unsigned, unsigned> indexLoopMap;
    SmallVector<std::pair<unsigned, unsigned>, 3> indexLoopInfo;
  };
  struct EdgeInfo{
    AffineMap accessMap;
    std::string firstElementTimeEq = "0";
    std::string lastElementTimeEq = "0";
    uint64_t firstElementTime = 0;
    uint64_t lastElementTime = 0;
    SmallVector<std::pair<unsigned, unsigned>, 3> indexLoopInfo;
  };

  // struct CombinedAccessStats{
  //   AffineMap accessMap;
  //   std::string firstElementTime;
  //   std::string lastElementTime;
  //   SmallVector<std::pair<unsigned, unsigned>, 3> indexLoopInfo;
  // };
  // struct NodeInfo{
  //   DenseMap<Operation *, AccessStats> loads;
  //   DenseMap<Operation *, AccessStats> stores;
  //   DenseMap<Operation *, uint64_t> localBufferSizes;
  // };

  enum PermutationType{
    Default,
    Minimize,
    Maximize
  };

  struct NodeInfo{
    SmallVector<unsigned> permutation;
    unsigned II = 0;
    unsigned tripCount = 0;
    // incoming edges
    DenseMap<AffineLoadOp, EdgeInfo> loadsMap;
    // outgoing edges
    DenseMap<AffineStoreOp, EdgeInfo> storesMap;
  };

  struct Node {
    // The unique identifier of this node in the graph.
    unsigned id;
    // The top-level statement which is (or contains) a load/store.
    Operation *op;

    uint64_t tripCount = 0;

    bool isReduction = false;

    uint64_t DSP_factor = 0;

    SmallVector<AffineForOp, 4> forOps;
    SmallVector<memref::AllocOp, 4> allocOps;

    // List of load operations.
    SmallVector<AffineLoadOp, 4> loads;
    // List of store op insts.
    SmallVector<AffineStoreOp, 4> stores;
    // // List of FIFO read operations.
    // SmallVector<Operation *, 4> fifoReads;
    // // List of FIFO write operations.
    // SmallVector<Operation *, 4> fifoWrites;

    // DenseMap<unsigned, SmallVector<unsigned>> permutationMaps;
    // DenseMap<unsigned, unsigned> permutationIIMaps;
    // DenseMap<unsigned, NodeInfo> nodeInfoMap; // for each permutation
    SmallVector<NodeInfo, 4> nodeInfo;
    // SmallVector<unsigned> optPermutation;

    int defaultPermIdx = -1;
    int minPermIdx = -1;
    int maxPermIdx = -1;

    SmallVector<unsigned, 4> tilingFactors;

    // int startTime = -1;
    // // int firstOutputTime = -1;
    // int endTime = -1;
    // int totalTime = -1;

    Node(unsigned id, Operation *op) : id(id), op(op) {}

    // // Returns the load op count for 'memref'.
    // unsigned getLoadOpCount(Value memref) const;

    // // Returns the store op count for 'memref'.
    // unsigned getStoreOpCount(Value memref) const;

    // // Returns the FIFO read op count for 'fifo'.
    // unsigned getFifoReadOpCount(Value fifo) const;

    // // Returns the FIFO write op count for 'fifo'.
    // unsigned getFifoWriteOpCount(Value fifo) const;
    
    // // Returns all store ops in 'storeOps' which access 'memref'.
    // void getStoreOpsForMemref(Value memref,
    //                           SmallVectorImpl<Operation *> *storeOps) const;

    // // Returns all load ops in 'loadOps' which access 'memref'.
    // void getLoadOpsForMemref(Value memref,
    //                          SmallVectorImpl<Operation *> *loadOps) const;

    // // Returns all memrefs in 'loadAndStoreMemrefSet' for which this node
    // // has at least one load and store operation.
    // void getLoadAndStoreMemrefSet(DenseSet<Value> *loadAndStoreMemrefSet) const;
  };

  // Edge represents a data dependence between nodes in the graph.
  struct Edge {
    // The id of the node at the other end of the edge.
    // If this edge is stored in Edge = Node.inEdges[i], then
    // 'Node.inEdges[i].id' is the identifier of the source node of the edge.
    // If this edge is stored in Edge = Node.outEdges[i], then
    // 'Node.outEdges[i].id' is the identifier of the dest node of the edge.
    unsigned id;
    // The SSA value on which this edge represents a dependence.
    // If the value is a memref, then the dependence is between graph nodes
    // which contain accesses to the same memref 'value'. If the value is a
    // non-memref value, then the dependence is between a graph node which
    // defines an SSA value and another graph node which uses the SSA value
    // (e.g. a constant or load operation defining a value which is used inside
    // a loop nest).
    Value value;
    AffineStoreOp srcOp = nullptr;
    AffineLoadOp dstOp = nullptr;
  };
  struct MyEdge {
    unsigned srcId;
    unsigned dstId;

    Value value;
  };

  struct Path{
    SmallVector<Edge> edges;
    SmallVector<unsigned> nodes;
  };

  // struct EdgeInfo{
  //   int64_t firstTimeOut = -1; // using lexmin
  //   int64_t lastTimeOut = -1; // FIXME: using trip count 
  // };

  SmallVector<MyEdge, 4> edges;

  // Map from node id to Node.
  DenseMap<unsigned, Node> nodes;
  // Map from node id to list of input edges.
  DenseMap<unsigned, SmallVector<Edge, 2>> inEdges;
  // Map from node id to list of output edges.
  DenseMap<unsigned, SmallVector<Edge, 2>> outEdges;
  // Map from memref to a count on the dependence edges associated with that
  // memref.
  DenseMap<Value, unsigned> memrefEdgeCount;

  // The next unique identifier to use for newly created graph nodes.
  unsigned nextNodeId = 0;

  DFG(Block &block) : block(block) {}

  // Initializes the dependence graph based on operations in `block'.
  // Returns true on success, false otherwise.
  bool init(bool mergeAllocNodes = true);

  Node *mergeNodes(unsigned srcId, unsigned dstId);

  // breadth first search
  bool BFS(unsigned srcId, unsigned dstId);

  // reverse breadth first search
  bool reverseBFS(unsigned srcId, unsigned dstId);

  // depth first search
  bool DFS(unsigned srcId, unsigned dstId);
  
  bool DFSUtil(unsigned currId, unsigned dstId, std::set<unsigned> &visited);
  
  // topological sort
  SmallVector<unsigned> topologicalSort();
  void topologicalSortUtil(unsigned currId, std::set<unsigned> &visited, std::stack<unsigned> &stack);
  // all paths search
  // SmallVector<SmallVector<unsigned>> getAllPaths(unsigned srcId, unsigned dstId);
  // SmallVector<SmallVector<unsigned>> getAllPathsUtil(unsigned currId, unsigned dstId, std::set<unsigned> &visited, SmallVector<unsigned> &path, SmallVector<SmallVector<unsigned>> &paths);

  SmallVector<Path> getAllPaths(unsigned srcId, unsigned dstId);
  SmallVector<Path> getAllPathsUtil(unsigned currId, unsigned dstId, std::set<unsigned> &visited, Path &path, SmallVector<Path> &paths);

  LogicalResult populateNodeInfo(bool enablePermutations);

  bool randomSearch();
  
  bool createRootNode();

  bool createSinkNode();

  bool createPermutationPerformanceModel(std::string fileName, uint timeLimitMinutes = 1440);
  
  bool createParallelizationPerformanceModel(std::string fileName, uint DSPs = 512, uint tilingLimit = 8, uint timeLimitMinutes = 1440);

  bool createCombinedOptimizationPerformanceModel(std::string fileName, uint DSPs = 512, uint tilingLimit = 8, uint timeLimitMinutes = 1440);

  bool createPermutationPythonModel(std::string fileName, PermutationType type = Default, bool optimizeOverlap = true);

  bool createParallelizationPythonModel(std::string fileName);

  bool createCombinedOptimizationPythonModel(std::string fileName);

  bool createEdgeConstraints(SmallVectorImpl<TypedValue<MemRefType>> &memRefs);

  bool callPermutationSolver(std::string fileName, bool isMinimize = true);

  bool callParallelizationSolver(std::string fileName);

  bool callCombinedOptimizationSolver(std::string fileName);

  bool saveSolutionToFile(std::string filePath);

  bool loadSolutionFromFile(std::string filePath);

  bool applyNodePermutations(PermutationType type = Minimize);

  bool applyNodeParallelization();

  bool applyCombinedOptimization();

  bool findRandomSolution();
  
  bool findUniformParallelSolution(uint tilingFactor = 2);

  bool findReversePermutations();

  bool createAmplModel();

  // Returns the graph node for 'id'.
  Node *getNode(unsigned id);

  // Returns the graph node for 'forOp'.
  Node *getForOpNode(AffineForOp forOp);

  // Adds a node with 'op' to the graph and returns its unique identifier.
  unsigned addNode(Operation *op);

  // Remove node 'id' (and its associated edges) from graph.
  void removeNode(unsigned id);

  // Returns true if node 'id' writes to any memref which escapes (or is an
  // argument to) the block. Returns false otherwise.
  bool writesToLiveInOrEscapingMemrefs(unsigned id);

  // Returns true iff there is an edge from node 'srcId' to node 'dstId' which
  // is for 'value' if non-null, or for any value otherwise. Returns false
  // otherwise.
  bool hasEdge(unsigned srcId, unsigned dstId, Value value = nullptr);

  // Adds an edge from node 'srcId' to node 'dstId' for 'value'.
  void addEdge(unsigned srcId, unsigned dstId, Value value);

  // Adds an edge from node 'srcId' to node 'dstId' for 'value'.
  void addEdge(
    unsigned srcId, unsigned dstId, 
    AffineStoreOp srcOp, AffineLoadOp dstOp,
    Value value
  );
  // Removes an edge from node 'srcId' to node 'dstId' for 'value'.
  void removeEdge(unsigned srcId, unsigned dstId, Value value);

  // Returns true if there is a path in the dependence graph from node 'srcId'
  // to node 'dstId'. Returns false otherwise. `srcId`, `dstId`, and the
  // operations that the edges connected are expected to be from the same block.
  bool hasDependencePath(unsigned srcId, unsigned dstId);

  // Returns the input edge count for node 'id' and 'memref' from src nodes
  // which access 'memref' with a store operation.
  unsigned getIncomingMemRefAccesses(unsigned id, Value memref);

  // Returns the output edge count for node 'id' and 'memref' (if non-null),
  // otherwise returns the total output edge count from node 'id'.
  unsigned getOutEdgeCount(unsigned id, Value memref = nullptr);

  /// Return all nodes which define SSA values used in node 'id'.
  void gatherDefiningNodes(unsigned id, DenseSet<unsigned> &definingNodes);

  // Computes and returns an insertion point operation, before which the
  // the fused <srcId, dstId> loop nest can be inserted while preserving
  // dependences. Returns nullptr if no such insertion point is found.
  Operation *getFusedLoopNestInsertionPoint(unsigned srcId, unsigned dstId);

  // Updates edge mappings from node 'srcId' to node 'dstId' after fusing them,
  // taking into account that:
  //   *) if 'removeSrcId' is true, 'srcId' will be removed after fusion,
  //   *) memrefs in 'privateMemRefs' has been replaced in node at 'dstId' by a
  //      private memref.
  void updateEdges(unsigned srcId, unsigned dstId,
                   const DenseSet<Value> &privateMemRefs, bool removeSrcId);

  // Update edge mappings for nodes 'sibId' and 'dstId' to reflect fusion
  // of sibling node 'sibId' into node 'dstId'.
  void updateEdges(unsigned sibId, unsigned dstId);

  // Adds ops in 'loads' and 'stores' to node at 'id'.
  void addToNode(unsigned id, const SmallVectorImpl<Operation *> &loads,
                 const SmallVectorImpl<Operation *> &stores);

  void print(raw_ostream &os) const;

  void printAsDot(raw_ostream &os) const;

  void dump() const { print(llvm::errs()); }

  /// The block for which this graph is created to perform fusion.
  Block &block;
};
} // namespace streamhls
} // namespace mlir