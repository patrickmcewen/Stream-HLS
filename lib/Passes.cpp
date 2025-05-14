/*
 * Copyright (c) 2024 Suhail Basalama
 *
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 */
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "streamhls/Transforms/Passes.h"

using namespace mlir;
using namespace streamhls;

namespace {
#define GEN_PASS_REGISTRATION
#include "streamhls/Transforms/Passes.h.inc"
} // namespace

namespace {
struct StreamHLSKernelPipelineOptions
    : public PassPipelineOptions<StreamHLSKernelPipelineOptions> {
  Option<std::string> hlsTopFunc{
      *this, "top-func", llvm::cl::init("forward"),
      llvm::cl::desc("Specify the top function of the design")};
  Option<int> debugPoint{
      *this, "debug-point", llvm::cl::init(0),
      llvm::cl::desc("Stop the pipeline at the given debug point")};
  Option<std::string> graphPath{
      *this, "graph-file", llvm::cl::init("graph.dot"),
      llvm::cl::desc("Specify the graph file path")};
  Option<std::string> reportPath{
      *this, "report-file", llvm::cl::init("report.csv"),
      llvm::cl::desc("Specify the report file path")};
  Option<std::string> loopPermutationType{
      *this, "loop-permutation-type", llvm::cl::init("default"),
      llvm::cl::desc("Type of loop permutation")};
  Option<bool> optimizeSchedule{
      *this, "optimize-schedule", llvm::cl::init(false),
      llvm::cl::desc("Optimize schedule")};
  Option<bool> parallelizeNodes{
      *this, "parallelize-nodes", llvm::cl::init(false),
      llvm::cl::desc("Parallelize nodes")};
  Option<uint> DSPs{
      *this, "board-dsps", llvm::cl::init(512),
      llvm::cl::desc("Number of DSPs")};
  Option<uint> tilingLimit{
      *this, "tiling-limit", llvm::cl::init(8),
      llvm::cl::desc("Tiling limit")};
  Option<bool> combinedOptimization{
      *this, "combined-optimization", llvm::cl::init(false),
      llvm::cl::desc("Combined optimization")};
  Option<uint> timeLimitMinutes{
      *this, "time-limit-minutes", llvm::cl::init(1440),
      llvm::cl::desc("MINLP solver time limit in minutes")};
  Option<bool> bufferizeFuncArgs{
      *this, "bufferize-func-args", llvm::cl::init(false),
      llvm::cl::desc("Bufferize function arguments")};
  Option<bool> optimizeConvReuse{
      *this, "optimize-conv-reuse", llvm::cl::init(false),
      llvm::cl::desc("Optimize convolution reuse")};
  Option<bool> minimizeOnChipBuffers{
      *this, "minimize-on-chip-buffers", llvm::cl::init(false),
      llvm::cl::desc("Minimize on-chip buffers")};
};
} // namespace

void streamhls::registerStreamHLSKernelPipeline() {
  PassPipelineRegistration<StreamHLSKernelPipelineOptions>(
      "streamhls-kernel-pipeline",
      "StreamHLS kernel pipeline",
      [](OpPassManager &pm, const StreamHLSKernelPipelineOptions &opts) {
        // Linalg fake quantization.
        // pm.addPass(streamhls::createLinalgFakeQuantizePass());
        // pm.addPass(mlir::createCanonicalizerPass());

        pm.addPass(streamhls::createRemoveRedundantOpsPass());
        pm.addPass(streamhls::createCreateWeightBinsPass(false, opts.hlsTopFunc));
        pm.addPass(mlir::createCanonicalizerPass());
        if (opts.debugPoint == 1)
          return;

        // Linalg optimization.
        pm.addPass(mlir::createLinalgElementwiseOpFusionPass()); // fuse layers of dnn
        pm.addPass(mlir::createConvertTensorToLinalgPass());
        pm.addPass(mlir::createCanonicalizerPass());
        if (opts.debugPoint == 2)
          return;

        // Bufferization.
        pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());
        pm.addPass(mlir::createLinalgBufferizePass());
        pm.addPass(arith::createArithBufferizePass());
        pm.addPass(mlir::tensor::createTensorBufferizePass());
        pm.addPass(func::createFuncBufferizePass());
        pm.addPass(bufferization::createBufferResultsToOutParamsPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCanonicalizerPass());
        if (opts.debugPoint == 3)
          return;

        // Linalg to Affine conversion.
        pm.addPass(mlir::createLinalgGeneralizationPass());
        pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
        pm.addPass(memref::createFoldMemRefAliasOpsPass());
        pm.addPass(mlir::createCanonicalizerPass());
        if (opts.debugPoint == 4)
          return;

        pm.addPass(streamhls::createLowerCopyToAffinePass());
        pm.addPass(memref::createFoldMemRefAliasOpsPass());
        pm.addPass(mlir::createCanonicalizerPass());

        if(opts.bufferizeFuncArgs){
          pm.addPass(streamhls::createBufferizeFuncArgsPass());
          pm.addPass(mlir::createCanonicalizerPass());
        }

        if (opts.debugPoint == 5){
          pm.addPass(streamhls::createPipelineInnerLoopsPass());
          pm.addPass(mlir::createCanonicalizerPass());
          return;
        }

        pm.addPass(streamhls::createRemoveLoopsOfUnitIterPass());
        pm.addPass(mlir::createCanonicalizerPass());
        if (opts.debugPoint == 6){
          pm.addPass(streamhls::createPipelineInnerLoopsPass());
          pm.addPass(mlir::createCanonicalizerPass());
          return;
        }
        
        if(opts.optimizeConvReuse){
          pm.addPass(streamhls::createStencilDataReusePass());
          pm.addPass(mlir::createCanonicalizerPass());
        }

        if (opts.debugPoint == 7){
          pm.addPass(streamhls::createPipelineInnerLoopsPass());
          pm.addPass(mlir::createCanonicalizerPass());
          return;
        }

        pm.addPass(streamhls::createConvertToSingleProducerSingleConsumerPass());
        pm.addPass(mlir::createCanonicalizerPass());

        pm.addPass(streamhls::createConstantFIFOPropogationPass());
        pm.addPass(mlir::createCanonicalizerPass());

        if (opts.debugPoint == 8){
          pm.addPass(streamhls::createPipelineInnerLoopsPass());
          pm.addPass(mlir::createCanonicalizerPass());
          return;
        }

        if(opts.combinedOptimization){
          pm.addPass(streamhls::createCombinedOptimizationPass(
            opts.reportPath, 
            opts.parallelizeNodes, 
            opts.DSPs, 
            opts.tilingLimit,
            opts.timeLimitMinutes
          ));
          pm.addPass(mlir::affine::createAffineLoopNormalizePass());
          pm.addPass(mlir::createCanonicalizerPass());
        }else{
          // if(opts.optimizeSchedule){
            pm.addPass(streamhls::createNodeGraphPipeliningPass(
              opts.reportPath,
              opts.loopPermutationType, 
              opts.optimizeSchedule,
              opts.timeLimitMinutes
            ));
            pm.addPass(mlir::createCanonicalizerPass());
          // }
          if(opts.parallelizeNodes){
            pm.addPass(streamhls::createNodeParallelizationPass(
              opts.reportPath, 
              opts.parallelizeNodes, 
              opts.DSPs, 
              opts.tilingLimit,
              opts.timeLimitMinutes
            ));
            pm.addPass(mlir::affine::createAffineLoopNormalizePass());
            pm.addPass(mlir::createCanonicalizerPass());
          }else{
            pm.addPass(streamhls::createPipelineInnerLoopsPass());
            pm.addPass(mlir::createCanonicalizerPass());
          }
        }

        if(opts.debugPoint == 9){
          // pm.addPass(streamhls::createPipelineInnerLoopsPass());
          pm.addPass(mlir::createCanonicalizerPass());
          return;
        }

        // if(opts.parallelizeNodes){
        //   pm.addPass(streamhls::createNodeParallelizationPass(opts.reportPath, opts.parallelizeNodes, opts.DSPs, opts.tilingLimit));
        //   pm.addPass(mlir::affine::createAffineLoopNormalizePass());
        //   pm.addPass(mlir::createCanonicalizerPass());
        // }
        // pm.addPass(streamhls::createMinimalBufferSizesPass());
        // pm.addPass(mlir::createCanonicalizerPass());

        if(opts.debugPoint == 10){
          // pm.addPass(streamhls::createPipelineInnerLoopsPass());
          pm.addPass(mlir::createCanonicalizerPass());
          return;
        }

        pm.addPass(streamhls::createConvertMemRefsToFIFOsPass(opts.parallelizeNodes));
        pm.addPass(mlir::createCanonicalizerPass());

        if (opts.debugPoint == 11){
          if(!opts.parallelizeNodes){
            pm.addPass(streamhls::createPipelineInnerLoopsPass());
            pm.addPass(mlir::createCanonicalizerPass());
          }
          return;
        }

        if(opts.minimizeOnChipBuffers){
          pm.addPass(streamhls::createMinimalBufferSizesPass());
          pm.addPass(mlir::createCanonicalizerPass());
        }

        pm.addPass(streamhls::createPrintDataflowGraphPass(opts.graphPath + ".dot", /*merge nodes*/ false));
        pm.addPass(streamhls::createPrintDataflowGraphPass(opts.graphPath + "_merged.dot", /*merge nodes*/ true));
        pm.addPass(mlir::createCanonicalizerPass());

        if (opts.debugPoint == 12)
          return;

        pm.addPass(streamhls::createCreateTasksPass());
        pm.addPass(mlir::createCanonicalizerPass());
        if (opts.debugPoint == 13)
          return;

        pm.addPass(streamhls::createCreateDataflowFromAffinePass());
        pm.addPass(mlir::createCanonicalizerPass());

        // pm.addPass(streamhls::createPipelineInnerLoopsPass());
        // pm.addPass(mlir::createCanonicalizerPass());

      });
}

namespace {
struct StreamHLSBaseKernelPipelineOptions
    : public PassPipelineOptions<StreamHLSBaseKernelPipelineOptions> {
  Option<std::string> hlsTopFunc{
      *this, "top-func", llvm::cl::init("forward"),
      llvm::cl::desc("Specify the top function of the design")};
  Option<unsigned> debugPoint{
      *this, "debug-point", llvm::cl::init(0),
      llvm::cl::desc("Stop the pipeline at the given debug point")};
};
} // namespace

void streamhls::registerStreamHLSBaseKernelPipeline() {
  PassPipelineRegistration<StreamHLSBaseKernelPipelineOptions>(
      "streamhls-base-kernel-pipeline",
      "StreamHLS Base kernel pipeline",
      [](OpPassManager &pm, const StreamHLSBaseKernelPipelineOptions &opts) {

        pm.addPass(streamhls::createRemoveRedundantOpsPass());
        pm.addPass(streamhls::createCreateWeightBinsPass(false, opts.hlsTopFunc));
        pm.addPass(mlir::createCanonicalizerPass());
        if (opts.debugPoint == 1)
          return;

        // Linalg optimization.
        pm.addPass(mlir::createLinalgElementwiseOpFusionPass()); // fuse layers of dnn
        pm.addPass(mlir::createConvertTensorToLinalgPass());
        pm.addPass(mlir::createCanonicalizerPass());
        if (opts.debugPoint == 2)
          return;

        // Bufferization.
        pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());
        pm.addPass(mlir::createLinalgBufferizePass());
        pm.addPass(arith::createArithBufferizePass());
        pm.addPass(mlir::tensor::createTensorBufferizePass());
        pm.addPass(func::createFuncBufferizePass());
        pm.addPass(bufferization::createBufferResultsToOutParamsPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCanonicalizerPass());
        if (opts.debugPoint == 3)
          return;

        // Linalg to Affine conversion.
        pm.addPass(mlir::createLinalgGeneralizationPass());
        pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
        pm.addPass(memref::createFoldMemRefAliasOpsPass());
        pm.addPass(mlir::createCanonicalizerPass());
        if (opts.debugPoint == 4)
          return;

        pm.addPass(streamhls::createLowerCopyToAffinePass());
        pm.addPass(memref::createFoldMemRefAliasOpsPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(streamhls::createBufferizeFuncArgsPass());
        pm.addPass(streamhls::createRemoveLoopsOfUnitIterPass());
        pm.addPass(mlir::createCanonicalizerPass());
        if (opts.debugPoint == 5){
          return;
        }

        pm.addPass(streamhls::createPipelineInnerLoopsPass());
        pm.addPass(mlir::createCanonicalizerPass());
      });
}

namespace {
struct StreamHLSHostPipelineOptions
    : public PassPipelineOptions<StreamHLSHostPipelineOptions> {
  Option<std::string> hlsTopFunc{
      *this, "top-func", llvm::cl::init("forward"),
      llvm::cl::desc("Specify the top function of the design")};
  // Option<unsigned> debugPoint{
  //     *this, "debug-point", llvm::cl::init(0),
  //     llvm::cl::desc("Stop the pipeline at the given debug point")};
};
} // namespace

void streamhls::registerStreamHLSHostPipeline() {
  PassPipelineRegistration<StreamHLSHostPipelineOptions>(
      "streamhls-host-pipeline",
      "StreamHLS host pipeline",
      [](OpPassManager &pm, const StreamHLSHostPipelineOptions &opts) {

        pm.addPass(streamhls::createRemoveRedundantOpsPass());
        pm.addPass(streamhls::createCreateWeightBinsPass(true, opts.hlsTopFunc));
        pm.addPass(mlir::createCanonicalizerPass());

        pm.addPass(func::createFuncBufferizePass());
        pm.addPass(bufferization::createBufferResultsToOutParamsPass());
        pm.addPass(mlir::createCanonicalizerPass());

        pm.addPass(streamhls::createCreateHostPass());
        // pm.addPass(mlir::createCanonicalizerPass());

      });
}

// namespace {
// struct STREAMHLSOptions
//     : public PassPipelineOptions<STREAMHLSOptions> {
//   Option<std::string> hlsTopFunc{
//       *this, "top-func", llvm::cl::init("forward"),
//       llvm::cl::desc("Specify the top function of the design")};
//   Option<std::string> hostOutFile{
//       *this, "host-out-file", llvm::cl::init("host.mlir"),
//       llvm::cl::desc("Specify the host output file")};
//   Option<std::string> kernelOutFile{
//       *this, "kernel-out-file", llvm::cl::init("kernel.mlir"),
//       llvm::cl::desc("Specify the kernel output file")};
//   Option<unsigned> debugPoint{
//       *this, "debug-point", llvm::cl::init(0),
//       llvm::cl::desc("Stop the pipeline at the given debug point")};
// };
// } // namespace

// void streamhls::registerSTREAMHLSPipeline() {
//   PassPipelineRegistration<STREAMHLSOptions>(
//       "streamhls-pipeline",
//       "Create kernel and host mlirs",
//       [](OpPassManager &pm, const STREAMHLSOptions &opts) {
//         // Linalg fake quantization.
//         pm.addPass(mlir::createCanonicalizerPass());

//         pm.addPass(streamhls::createRemoveRedundantOpsPass());
//         pm.addPass(mlir::createCanonicalizerPass());

//         pm.addPass(streamhls::createCreateWeightBinsPass(false));
//         pm.addPass(mlir::createCanonicalizerPass());

//         pm.addPass(func::createFuncBufferizePass());
//         pm.addPass(bufferization::createBufferResultsToOutParamsPass());
//         pm.addPass(mlir::createCanonicalizerPass());
//         pm.addPass(streamhls::createCreateHostPass());
        
//         pm.addPass(streamhls::createPrintIRPass(opts.kernelOutFile.getValue()));
//       });
// }

void streamhls::registerTransformsPasses(){
  registerStreamHLSBaseKernelPipeline();
  registerStreamHLSKernelPipeline();
  registerStreamHLSHostPipeline();
  // registerSTREAMHLSPipeline();
  registerStreamHLSPasses();
}