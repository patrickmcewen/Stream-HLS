# StreamHLS Kernel Pipeline

This document describes the `streamhls-kernel-pipeline` registered by
`registerStreamHLSKernelPipeline()` in `lib/Passes.cpp`.

The pipeline starts from Torch/ML-style tensor IR, lowers it through Linalg and
bufferized affine loops, builds a dataflow graph, applies optional scheduling and
parallelization transforms, converts memory edges to StreamHLS FIFOs, and finally
outlines dataflow tasks into callable kernel functions.

## Pipeline Options

The pipeline is invoked as:

```bash
streamhls-opt input.mlir \
  -streamhls-kernel-pipeline="top-func=forward debug-point=14 ..."
```

Important options:

| Option | Default | Effect |
| --- | --- | --- |
| `top-func` | `forward` | Function used as the top kernel. `CreateWeightBins` looks this up. |
| `debug-point` | `0` | Stops the pipeline early. `0` returns immediately before running any pass. |
| `graph-file` | `graph.dot` | Prefix/path used for emitted dataflow `.dot` graphs. |
| `report-file` | `report.csv` | Prefix/path used for generated performance models, scripts, reports, and solution files. |
| `loop-permutation-type` | `default` | Intended selector for permutation strategy. Current `NodeGraphPipelining` always emits the default Python model and applies the minimize solver path when enabled. |
| `optimize-schedule` | `false` | Enables solver-backed node loop permutation in `NodeGraphPipelining`. |
| `parallelize-nodes` | `false` | Enables node parallelization path and affects FIFO conversion behavior. |
| `combined-optimization` | `false` | Uses the combined permutation/parallelization solver path instead of separate scheduling and parallelization passes. |
| `board-dsps` | `512` | DSP budget passed to node parallelization and combined optimization. |
| `tiling-limit` | `8` | Maximum tiling/parallelization factor considered by optimization passes. |
| `time-limit-minutes` | `1440` | Solver time limit passed into generated optimization models. |
| `bufferize-func-args` | `false` | Inserts local buffers for input memref arguments before dataflow conversion. |
| `optimize-conv-reuse` | `false` | Enables stencil/convolution data reuse rewrites. |
| `minimize-on-chip-buffers` | `false` | Shrinks local buffer dimensions after FIFO conversion. |
| `tech-config` | empty | Optional JSON file with technology-specific latency/DSP values. Loaded by optimization passes. |

## Debug Points

`debug-point=0` returns before any pass. Later values return after progressively
larger prefixes:

| Debug point | Stops after |
| --- | --- |
| `1` | Redundant-op removal, weight argument extraction, canonicalization. |
| `2` | Linalg elementwise fusion, tensor-to-linalg conversion, canonicalization. |
| `3` | Bufferization and cleanup. |
| `4` | Linalg generalization, Linalg-to-affine lowering, memref alias folding. |
| `5` | Copy-to-affine lowering, optional function-argument buffering, then `PipelineInnerLoops` and canonicalization. |
| `6` | Unit-loop removal, then `PipelineInnerLoops` and canonicalization. |
| `7` | Optional stencil reuse, then `PipelineInnerLoops` and canonicalization. |
| `8` | Single-producer/single-consumer conversion and constant propagation, then `PipelineInnerLoops` and canonicalization. |
| `9` | Optimization phase, then one extra canonicalizer. |
| `10` | Currently equivalent to `9` plus another canonicalizer; old minimal-buffer work is commented out. |
| `11` | Memrefs-to-FIFOs conversion; if nodes are not parallelized, also pipelines inner loops before returning. |
| `12` | Optional buffer minimization, graph `.dot` emission, canonicalization. |
| `13` | Task creation and canonicalization. |
| `14+` | Full kernel pipeline through dataflow outlining and arithmetic blackboxing. |

## Ordered Pass Walkthrough

### 1. Initial Cleanup and Weight Extraction

1. `streamhls-remove-redundant-ops`

   Source: `lib/Transforms/RemoveRedundantOps.cpp`

   Removes unused `ml_program.global` seed operations. The implementation is a
   module-level greedy rewrite that erases only globals with no uses. This is an
   input cleanup pass for model IR that may carry an unused random seed/global.

2. `streamhls-create-weight-bins`

   Source: `lib/Transforms/CreateWeightBins.cpp`

   Rewrites non-splat dense tensor constants inside `top-func` into additional
   function arguments. In the kernel pipeline it is constructed as
   `createCreateWeightBinsPass(false, opts.hlsTopFunc)`, so weights are not kept
   in the module; constants are replaced by newly inserted top-function
   arguments. In host mode, the same pass can preserve weights as
   `DenseResourceElementsAttr`.

3. `canonicalize`

   Upstream MLIR cleanup. It folds and simplifies IR after the model-specific
   global and constant rewrites.

### 2. Linalg Frontend Lowering

4. `linalg-elementwise-op-fusion`

   Upstream MLIR Linalg pass. Fuses producer/consumer elementwise Linalg ops,
   reducing intermediate tensors and making later loop/dataflow structure less
   fragmented. In this LLVM checkout, the pass fuses tensor elementwise Linalg
   producers into consumers when profitable/valid and also installs reshape,
   Linalg canonicalization, and constant-folding patterns.

5. `convert-tensor-to-linalg`

   Upstream MLIR conversion. Rewrites selected tensor dialect operations that
   have Linalg equivalents into Linalg operations, moving the kernel toward
   structured loop-like IR. In this checkout, the pass specifically lowers
   `tensor.pad` through Linalg pad generalization.

6. `canonicalize`

   Cleans up after Linalg fusion/conversion.

### 3. Bufferization

7. `empty-tensor-to-alloc-tensor`

   Upstream bufferization pass. Converts `tensor.empty` producers into
   bufferization allocation tensors so later bufferization has explicit
   allocation sites.

8. `linalg-bufferize`

   Upstream MLIR pass. Bufferizes Linalg ops, replacing tensor-based structured
   operations with memref-based operations where possible. This is a
   partial-bufferization pass scoped to Linalg operations.

9. `arith-bufferize`

   Upstream MLIR pass. Bufferizes arith operations that participate in tensor or
   memref conversion.

10. `tensor-bufferize`

    Upstream MLIR pass. Converts remaining tensor dialect operations to
    bufferized form. This is a partial-bufferization pass scoped to Tensor
    dialect operations.

11. `func-bufferize`

    Upstream MLIR pass. Updates function signatures and function-like operation
    boundaries for bufferized values.

12. `buffer-results-to-out-params`

    Upstream MLIR pass. Converts memref-returning functions into functions that
    receive output buffers as arguments, which is a better fit for HLS-style
    explicit memory interfaces.

13. `canonicalize`

14. `canonicalize`

    Two cleanup passes are run back-to-back after bufferization. They simplify
    casts, folded values, and canonical patterns exposed by the chained
    bufferization passes.

### 4. Linalg to Affine

15. `linalg-generalize-named-ops`

    Upstream MLIR pass. Converts named Linalg operations into generic Linalg
    operations. This avoids special-case named operation handling before affine
    loop lowering.

16. `convert-linalg-to-affine-loops`

    Upstream MLIR conversion. Lowers Linalg operations into explicit
    `affine.for` loop nests with affine loads/stores. StreamHLS custom passes
    downstream assume this affine loop and memref representation.

17. `fold-memref-alias-ops`

    Upstream memref cleanup. Folds view/alias operations so later passes see
    direct memref arguments and allocations where possible.

18. `canonicalize`

### 5. Copy Lowering and Optional Argument Buffering

19. `scalehls-lower-copy-to-affine`

    Source: `lib/Transforms/LowerCopyToAffine.cpp`

    Rewrites each `memref.copy` into a perfect affine loop nest over the source
    memref shape. The generated loop body performs one `affine.load` from the
    source and one `affine.store` to the target, then erases the original copy.
    The pass has an `internal-copy-only` option, but the current implementation
    hardcodes external-copy detection to `false`, so the pipeline conversion
    applies to all matched copies.

20. `fold-memref-alias-ops`

21. `canonicalize`

22. Optional `streamhls-bufferize-func-args`

    Source: `lib/Transforms/BufferizeFuncArgs.cpp`

    Enabled by `bufferize-func-args=true`. The pass expects all top-function
    arguments to be memrefs. It classifies each memref argument as input or
    output by checking affine load/store users, asserts if an argument is both,
    and for input arguments allocates a same-shaped local buffer at function
    entry. It copies the input argument into the local buffer with affine loops
    and redirects later uses to the local buffer. This isolates external input
    ports from internal reuse/dataflow transformations.

23. Optional `canonicalize`

### 6. Affine Normalization and Optional Reuse

24. `streamhls-remove-loops-of-unit-iter`

    Source: `lib/Transforms/RemoveLoopsOfUnitIter.cpp`

    Removes degenerate affine loop structure and unit dimensions. It replaces
    induction variable uses in constant `0..1` step-1 loops with index constant
    `0`, promotes single-iteration affine loops, and collapses local memref
    dimensions of size `1` while updating affine load/store maps.

25. `canonicalize`

26. Optional `streamhls-stencil-data-reuse`

    Source: `lib/Transforms/StencilDataReuse.cpp`

    Enabled by `optimize-conv-reuse=true`. This pass targets stencil or
    convolution-like affine accesses. It recognizes sliding-window affine loads
    and padding-offset stores, creates reuse buffers, expands or guards loop
    regions with `affine.if`, inserts preload/store logic, and redirects target
    accesses to the reuse buffer. The goal is to expose on-chip reuse before
    dataflow/FIFO conversion.

27. Optional `canonicalize`

### 7. Producer/Consumer Simplification

28. `streamhls-convert-to-single-producer-single-consumer`

    Source: `lib/Transforms/ConvertToSingleProducerSingleConsumer.cpp`

    Normalizes local memrefs so the later FIFO pass sees simpler
    producer-consumer edges. Active rewrite groups include:

    - removing zero-rank temporary memrefs initialized by constants;
    - duplicating a single-store/multiple-load temporary into per-consumer
      buffers;
    - handling store/load and store/load/store/load cases by inserting
      intermediate buffers and guarded affine transfers when access analysis
      requires boundary conditions.

    Conceptually, this reduces shared memory fanout/fanin into independent
    single-producer/single-consumer channels.

29. `canonicalize`

30. `streamhls-constant-fifo-propogation`

    Source: `lib/Transforms/ConstantFIFOPropogation.cpp`

    The name suggests FIFO propagation, and a `StreamWriteOp`/`StreamReadOp`
    pattern exists in the file, but it is currently commented out. The active
    pattern propagates constants through local memrefs: if an `affine.store`
    writes an `arith.constant` to a non-argument memref with exactly two uses,
    the matching `affine.load` is replaced with the constant, then the store,
    load, and now-unused defining op are erased.

31. `canonicalize`

### 8. Scheduling and Parallelization Optimization

This section has two mutually exclusive paths controlled by
`combined-optimization`.

32a. If `combined-optimization=true`: `streamhls-combined-optimization`

   Source: `lib/Transforms/CombinedOptimization.cpp`, `lib/Support/DFG.cpp`

   Builds a `DFG` from the function body, optionally loads `tech-config`, emits
   a combined optimization performance model using `report-file`, `board-dsps`,
   `tiling-limit`, and `time-limit-minutes`, invokes the combined solver, writes
   a solution JSON, emits a Python model, and applies the combined optimization.
   The applied transform includes loop permutation, loop II metadata, tiling, and
   cyclic array partitioning through the `DFG` support code. The pass stores the
   `parallelize-nodes` option but does not use it directly; the pipeline gate is
   `combined-optimization=true`.

   Expected side effects include files derived from `report-file`, such as the
   combined model, generated Python model, solution JSON, and DOT/report files
   produced by the `DFG` helpers.

33a. `affine-loop-normalize`

   Upstream MLIR affine cleanup. Normalizes affine loop bounds/steps after
   solver-selected loop transforms.

34a. `canonicalize`

32b. If `combined-optimization=false`: `streamhls-graph-node-pipelining`

   Source: `lib/Transforms/NodeGraphPipelining.cpp`, `lib/Support/DFG.cpp`

   Builds the `DFG`, optionally loads `tech-config`, emits permutation
   performance and Python models, and when `optimize-schedule=true` calls the
   permutation solver and applies minimizing node loop permutations. The pass is
   always inserted on this path, but the solver-backed transform is conditional.
   The `loop-permutation-type` option is stored; the current implementation
   emits the default Python model and applies the minimizing solver solution when
   `optimize-schedule=true`.

33b. `canonicalize`

34b. If `parallelize-nodes=true`: `streamhls-node-parallelization`

   Source: `lib/Transforms/NodeParallelization.cpp`, `lib/Support/DFG.cpp`

   Builds the `DFG`, optionally loads `tech-config`, emits a parallelization
   performance model with DSP and tiling constraints, calls the parallelization
   solver, and applies node parallelization decisions. The `DFG` application
   path records tiling factors and performs cyclic array partitioning and
   loop/operation updates needed for the chosen factors. The pass stores
   `parallelize-nodes`; in the kernel pipeline, the pass itself is only inserted
   when that pipeline option is true.

35b. If `parallelize-nodes=true`: `affine-loop-normalize`

36b. If `parallelize-nodes=true`: `canonicalize`

34c. If `parallelize-nodes=false`: `streamhls-pipeline-inner-loops`

   Source: `lib/Transforms/PipelineInnerLoops.cpp`

   Finds affine loop bands and marks the innermost loop of each band with an HLS
   pipeline directive using initiation interval `1`.

35c. If `parallelize-nodes=false`: `canonicalize`

### 9. FIFO Conversion and Buffer Sizing

37. `streamhls-convert-memrefs-to-fifos`

    Source: `lib/Transforms/ConvertMemRefsToFIFOs.cpp`

    Converts producer/consumer temporary memrefs into StreamHLS dataflow
    channels. For eligible local `memref.alloc` buffers with a compatible
    single affine store and single affine load, it creates `dataflow.stream` or
    `dataflow.array_of_streams` operations, inserts stream writes at producer
    sites, inserts stream reads at consumer sites, replaces the load result, and
    erases the original store/load/alloc. It skips incompatible access patterns,
    including some same-loop store/load pairs, and may wrap writes/reads in
    `affine.if` guards derived from access analysis. The `parallelize-nodes`
    option selects the parallel-aware FIFO rewrite path.

38. `canonicalize`

39. At `debug-point=11` only, if `parallelize-nodes=false`:
    `streamhls-pipeline-inner-loops` and `canonicalize`

40. Optional `streamhls-minimize-buffer-sizes`

    Source: `lib/Transforms/MinimizeBufferSizes.cpp`

    Enabled by `minimize-on-chip-buffers=true`. The active rewrite,
    `MinimizeLocalBuffersNoPartition`, analyzes selected local `memref.alloc`
    users in nested affine-loop patterns, finds dimensions whose access extent
    can be reduced based on surrounding loop bounds and affine maps, then updates
    the memref type to a smaller shape and adjusts affine access operands/maps.
    It is conservative around unsupported affine expressions such as `floordiv`
    and `mod`.

41. Optional `canonicalize`

### 10. Graph Emission and Dataflow Task Lowering

42. `streamhls-print-dataflow-graph` with `mergeNodes=false`

    Source: `lib/Analysis/PrintDataflowGraph.cpp`

    Builds a `DFG` from the function body and writes it as a DOT graph to
    `graph-file + ".dot"`. Because the pipeline appends a suffix, the default
    `graph-file=graph.dot` produces `graph.dot.dot`.

43. `streamhls-print-dataflow-graph` with `mergeNodes=true`

    Emits a second DOT graph to `graph-file + "_merged.dot"`, using the DFG
    builder's merged-node mode. With the default `graph-file=graph.dot`, this
    produces `graph.dot_merged.dot`.

44. `canonicalize`

45. `streamhls-create-tasks`

    Source: `lib/Transforms/CreateTasks.cpp`

    Wraps each DFG node into a `dataflow.task`. It first dispatches the function
    body, builds the `DFG`, creates a task operation at each node, moves the
    node's allocation operations and affine loops into the task body, and moves
    streams/array-of-streams/allocation operations to the beginning of the
    dispatch block.

46. `canonicalize`

47. `streamhls-create-dataflow-from-affine`

    Source: `lib/Transforms/CreateDataflowFromAffine.cpp`

    Converts the task-level dataflow structure into outlined functions and calls.
    It dispatches the function body, rewrites `dataflow.task` operations into
    `dataflow.node` operations with live-in values classified as inputs, outputs,
    or scalar params, then outlines each node into a private `func.func` named
    `node<N>`. Each original node is replaced by a `func.call`, and the
    temporary dispatch wrapper is removed.

48. `canonicalize`

### 11. Arithmetic Blackboxing

49. `streamhls-operation-blackbox`

    Source: `lib/Transforms/OperationBlackbox.cpp`

    Walks from the `forward` function through reachable call operations and
    replaces selected arithmetic/math operations with calls to private helper
    functions. Supported operations include `arith.addf`, `arith.mulf`,
    `arith.divf`, `arith.subf`, and `math.exp`. Operations outside an affine loop
    receive a `_ctrl_chain` suffix. The pass creates helper functions such as
    `addf`, `mulf`, `divf`, `subf`, and `exp_bb` on demand. The current `exp_bb`
    helper is a placeholder implementation that multiplies the input by `2.0`.

50. `canonicalize`

## Reading Per-Pass IR Dumps

`examples/run_streamhls.py` and `examples/streamhls_pipeline.py` include helper
flags for pass IR logging:

```bash
python run_streamhls.py ... --dump-pass-ir --dump-pass-ir-diffs
```

By default, this writes:

```text
<design>/<kernel>/mlir/intermediates/<kernel>_pass_ir.log
```

The raw dumps come from MLIR's pass-manager instrumentation:

- `-mlir-disable-threading=true`
- `-mlir-print-ir-after-all`
- `-mlir-print-ir-module-scope`

With `--dump-pass-ir-diffs`, the script post-processes the log and appends a
unified diff after each dump showing changes relative to the previous dump.
