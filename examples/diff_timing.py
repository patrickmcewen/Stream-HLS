"""Differentiable analytical timing model for the Stream-HLS parametrized DFG IR.

Mirrors the recursive latency-rate walk of step_tl's timing model
(step_tl/src/timing_and_emulator/timing.py): nodes are visited in topological
order and each node's start time is defined recursively in terms of its
predecessors via `max`. Unlike the C++ HLS model it does not port the affine
access "time function"; every quantity is instead a smooth, closed-form
function of the continuous design knobs (tile sizes, soft loop permutations) so
the total-cycle estimate is differentiable end-to-end.

Cross-node coupling comes from a differentiable buffer-fill gate
(see buffer_model_slide): the per-edge buffer size divided by the producer
tile size gives `fill` -- how many producer output tiles must be emitted
before the consumer can fire. With II = 1,

    fw(p) = st(p) + T_op(p)             # first output tile
    lw(p) = st(p) + N_out(p) * T_op(p)  # last output tile

and `fill in [1, N_out(p)]` slides the consumer start smoothly between them:

    st(n) = max over in-edges (p -> n) of  st(p) + fill_e * T_op(p)

NOTE on II: II is fixed to 1 here. In the real HLS schedule II jumps to the
reduction-op latency (e.g. arith.addf) when the reduction loop is the
*innermost* loop after permutation, and is 1 otherwise. That coupling is
permutation-dependent and can later be added softly using the same soft
permutation `P` (e.g. II = 1 + (lat-1) * Pr[reduction loop innermost]).
"""

import json
import math
import re

import torch

REDUCTION_OP_LATENCY = 4.0  # StreamHLS fadd default (TechConfig); II when a
# reduction loop is innermost. Overridable per node via a JSON "reduction_latency"
# field (e.g. fdiv=15, fexp=8 for non-add reductions); see _node_ii.

_DTYPE_BYTES = {
    "f64": 8, "f32": 4, "f16": 2, "bf16": 2,
    "i64": 8, "i32": 4, "i16": 2, "i8": 1, "i1": 1,
}


def dtype_bytes(type_str):
    """Element size in bytes parsed from a memref type string, e.g.
    'memref<200x220xf32>' -> 4. (Should eventually be emitted explicitly in
    the JSON IR rather than parsed here.)"""
    m = re.search(r"x([a-z]+[0-9]+)>", type_str)
    assert m, f"cannot parse element dtype from {type_str!r}"
    et = m.group(1)
    assert et in _DTYPE_BYTES, f"unknown element dtype {et!r}"
    return _DTYPE_BYTES[et]


def _prod(terms):
    """Differentiable product of a list of scalar tensors (1.0 if empty)."""
    out = torch.tensor(1.0)
    for t in terms:
        out = out * t
    return out


class DesignParams:
    """Continuous (differentiable) design knobs, one set per node.

    Tile sizes are kept as raw positive reals (integrality relaxed). Loop
    permutation is a soft assignment: `perm_logits[nid]` is an
    (n_loops x n_positions) matrix, softmax'd over positions so each loop holds a
    distribution over loop-nest depths. Partition factor is intentionally absent:
    it has no effect in the timing model, so it is neither read nor modeled here.
    """

    # Logit peak for the init soft permutation. Large enough that leakage into
    # off positions is negligible (~3e-4 per position at 8.0), so II and the
    # streamability gate evaluate at ~the discrete JSON permutation; an optimizer
    # still gets nonzero softmax gradients to move the permutation.
    INIT_SCALE = 8.0

    def __init__(self, graph):
        self.tile = {}
        self.perm_logits = {}
        for node in graph["nodes"]:
            nid = node["id"]
            loops = node["loops"]
            n = len(loops)
            self.tile[nid] = torch.tensor(
                [float(l["tiling_factor"]) for l in loops]
            ).clone().requires_grad_(True)
            logits = torch.zeros(n, n)
            perm = node["permutation"]
            assert len(perm) == n, "permutation length must match loop count"
            for i, pos in enumerate(perm):
                logits[i, pos] = self.INIT_SCALE
            self.perm_logits[nid] = logits.clone().requires_grad_(True)

    def soft_perm(self, nid):
        return torch.softmax(self.perm_logits[nid], dim=1)

    def parameters(self):
        for d in (self.tile, self.perm_logits):
            yield from d.values()


def _expected_pos(perm_row):
    """Expected loop-nest position (0 = outermost) of a loop given its soft
    permutation row (a distribution over positions)."""
    positions = torch.arange(perm_row.shape[0], dtype=perm_row.dtype)
    return torch.dot(perm_row, positions)


class DFGTiming:
    def __init__(self, graph, sigma_t=0.5, sigma_r=0.5, tau_r=0.3):
        self.graph = graph
        self.sigma_t = sigma_t
        self.sigma_r = sigma_r  # rank-match width (order-match m_d)
        self.tau_r = tau_r      # soft-rank sharpness
        self.nodes = {n["id"]: n for n in graph["nodes"]}
        self.in_edges = {nid: [] for nid in self.nodes}
        self.out_edges = {nid: [] for nid in self.nodes}
        for e in graph["edges"]:
            self.out_edges[e["src"]].append(e)
            self.in_edges[e["dst"]].append(e)

    def _topo_order(self):
        indeg = {nid: len(self.in_edges[nid]) for nid in self.nodes}
        ready = sorted(nid for nid, d in indeg.items() if d == 0)
        order = []
        while ready:
            nid = ready.pop(0)
            order.append(nid)
            for e in self.out_edges[nid]:
                indeg[e["dst"]] -= 1
                if indeg[e["dst"]] == 0:
                    ready.append(e["dst"])
            ready.sort()
        assert len(order) == len(self.nodes), "graph is not a DAG"
        return order

    def _output_loops(self, nid):
        """Loop indices that index some output tensor dim. Loops absent from
        this set are reduction loops. A node with no out-edge (graph sink) is
        treated as having no reduction loops."""
        out = self.out_edges[nid]
        if not out:
            return set(range(len(self.nodes[nid]["loops"])))
        loops = set()
        for e in out:
            for dim in e["dims"]:
                if dim["producer_loop"] is not None:
                    loops.add(dim["producer_loop"])
        return loops

    def _node_ii(self, nid, params):
        """Soft initiation interval for a node, mirroring StreamHLS getForLoopII.

        StreamHLS sets II to the reduction op's latency L (e.g. fadd=4) when a
        loop-carried accumulation sits on the *innermost* loop after permutation,
        and 1 otherwise -- i.e. when the innermost loop does not index the output
        (it is a reduction loop). The soft permutation gives the probability that
        some reduction loop occupies the innermost nest position, so

            II = 1 + (L - 1) * P[reduction loop innermost]

        is a smooth relaxation: II -> L as a reduction loop moves innermost and
        -> 1 as it moves out. Nodes with no reduction loop get II = 1.

        L and the reduction-loop set come straight from the IR when the node
        carries "reduction_latency"/"reduction_loops" (emitted by DFG.cpp's
        computeNodeIIInfo: L is the full loop-carried recurrence-chain latency,
        e.g. fexp+fadd, not just fadd; the reduction loops are exactly the loops
        whose IV does not index the carried store). For older JSONs that predate
        those fields we fall back to inferring reduction loops as the loops absent
        from every out-edge and L = REDUCTION_OP_LATENCY (fadd).

        Tiling a reduction loop unrolls its inner band (bound = tiling factor),
        so each pipelined iteration merges U = Pi(reduction tiling factors) fresh
        partial products plus the carried accumulator. Vitis sums those U+1 values
        with a balanced adder tree, and the loop-carried path runs through the
        whole tree, so the recurrence latency is the tree depth times L:

            II = depth * L,  depth = ceil(log2(U + 1)) = 1 + log2(U)  (power-of-2 U)

        This is exact (not fitted): e.g. fadd L=4 gives II=4 at U=1 (depth 1) and
        II=12 at U=4 (depth 3), matching real HLS (Final II=4 vs 12 for a MAC node
        at reduction tile 1 vs 4). depth->L when the reduction is untiled (U=1).
        """
        node = self.nodes[nid]
        loops = node["loops"]
        n = len(loops)
        if "reduction_loops" in node:
            red_loops = list(node["reduction_loops"])
            lat = float(node["reduction_latency"])
        else:
            out_loops = self._output_loops(nid)
            red_loops = [d for d in range(n) if d not in out_loops]
            lat = float(node.get("reduction_latency", REDUCTION_OP_LATENCY))
        if not red_loops or lat <= 1.0:
            return torch.tensor(1.0)
        tile = params.tile[nid]
        log2_u = torch.stack([torch.log2(tile[r].clamp(min=1.0))
                              for r in red_loops]).sum()
        eff_lat = (1.0 + log2_u) * lat  # adder-tree depth * recurrence latency
        p = params.soft_perm(nid)  # (n_loops x n_positions), innermost = col n-1
        p_inner_red = torch.stack([p[r][n - 1] for r in red_loops]).sum()
        return 1.0 + (eff_lat - 1.0) * p_inner_red

    def _node_intrinsics(self, nid, params):
        """Return (N_out, T_op, II) for a node, differentiable in the knobs.

        StreamHLS pipelines the outer (tiled) loop band at the node's II, which
        fully unrolls the inner tile loops (bound = tiling factor) into a spatial
        datapath. So a node's cycle count is II * the *outer* iteration product
        Pi(trip/tile), not the full Pi(trip): the tiling factor is the
        parallelization (unroll) knob and larger tiles lower latency.

        N_out  = product over output loops of trip/tile   (output tiles)
        T_op = II * Pi(trip/tile over all loops) / N_out (outer cycles per tile)
               => lw = N_out * T_op = II * Pi(trip/tile), tile- and II-dependent.

        II is permutation-dependent (see _node_ii). NOTE: partition_factor has no
        effect on timing -- array partitioning is only memory banking; the
        emitted partition factor follows the tiling factor regardless.
        """
        loops = self.nodes[nid]["loops"]
        trip = [float(l["trip_count"]) for l in loops]
        tile = params.tile[nid]
        out_loops = self._output_loops(nid)

        ii = self._node_ii(nid, params)
        n_out = _prod([trip[d] / tile[d] for d in out_loops])
        total_outer = _prod([trip[d] / tile[d] for d in range(len(loops))])
        t_op = ii * total_outer / n_out
        return n_out, t_op, ii

    def _edge_ranks(self, edge, params):
        """Shared dims of an edge with each dim's soft nest-position rank in the
        producer and consumer nests.

        A dim counts only when it maps 1:1 to a single loop on *both* sides with
        matching extents. The two ways that fails are both row-major reshapes that
        linearize one tensor dim across several loops:
          - a null mapping (producer_loop or consumer_loop is None): a
            non-single-induction-var access on that side (e.g. MHSA reads a
            producer dim of extent 128 as consumer loops 8x16, expr d0*16+d2);
          - matched loops whose extents disagree: the mirror case, several
            producer dims read as one consumer loop of 128.
        Such an access preserves emission order and streams, so the dim is left
        out and contributes neutrally.

        Returns (info, rank_in, rank_out, pos_in): info[i] = (producer_loop,
        consumer_loop); pos_in/rank_in are the expected position / soft rank of
        each dim in the producer nest (rank = count of dims at a smaller
        position). info is empty when the edge has no streamable dims to gate on.
        """
        src, dst = edge["src"], edge["dst"]
        p_in = params.soft_perm(src)
        p_out = params.soft_perm(dst)
        trip_src = [float(l["trip_count"]) for l in self.nodes[src]["loops"]]
        trip_dst = [float(l["trip_count"]) for l in self.nodes[dst]["loops"]]

        info = []
        pos_in, pos_out = [], []
        for dim in edge["dims"]:
            pl, cl = dim["producer_loop"], dim["consumer_loop"]
            if pl is None or cl is None:
                continue
            if not math.isclose(trip_src[pl], trip_dst[cl]):
                continue
            info.append((pl, cl))
            pos_in.append(_expected_pos(p_in[pl]))
            pos_out.append(_expected_pos(p_out[cl]))
        if not info:
            return info, None, None, None
        pos_in = torch.stack(pos_in)
        pos_out = torch.stack(pos_out)

        def soft_rank(pos):
            diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # diff[i,j] = pos_i - pos_j
            gt = torch.sigmoid(diff / self.tau_r)
            return gt.sum(1) - torch.diagonal(gt)       # drop self (sigmoid(0)=.5)
        return info, soft_rank(pos_in), soft_rank(pos_out), pos_in

    def fusion_penalty(self, params):
        """Additive order+tile misalignment summed over every edge's shared dims.

        StreamHLS fuses an edge into a stream only when the producer store and
        consumer load have identical access maps -- the shared tensor dims read in
        the same relative nest order *and* tiled identically (the storeMap ==
        loadMap test in ConvertMemRefsToFIFOs). _edge_fill already rewards that,
        but through a *cumulative product* whose gradient vanishes on inner dims
        once an outer dim is misaligned, so a misaligned start gets almost no
        signal to climb back onto the fusion manifold. This term is a plain sum
        instead, giving every shared dim an independent gradient toward fusion:
          order: (rank_in - rank_out)^2  -- same relative nest order
          tile:  (log a - log b)^2       -- equal tiling factor
        Zero exactly when all connected nodes agree on order and tiling for their
        shared dims (every edge fusable)."""
        total = torch.tensor(0.0)
        for e in self.graph["edges"]:
            info, rank_in, rank_out, _ = self._edge_ranks(e, params)
            if not info:
                continue
            tile_src = params.tile[e["src"]]
            tile_dst = params.tile[e["dst"]]
            for i, (pl, cl) in enumerate(info):
                order_gap = (rank_in[i] - rank_out[i]) ** 2
                tile_gap = (torch.log(tile_src[pl]) - torch.log(tile_dst[cl])) ** 2
                total = total + order_gap + tile_gap
        return total

    def _edge_fill(self, edge, params):
        """Differentiable buffer-fill count for an edge: number of producer
        output tiles that must be emitted before the consumer can start.

        fill = (BufSize / 2) / (elem_bytes * prod(producer tile sizes))
             = prod_d max(a_d,b_d)^S_d * D_d^(1-S_d) / prod_d a_d

        The /2 drops the ping-pong factor: the consumer starts once one buffer
        half is full. The constant elem_bytes cancels. fill in [~1, N_out(p)].

        Streamability gates on two things, matching StreamHLS's fusion test
        (ConvertMemRefsToFIFOs: an edge becomes a FIFO only when the producer
        store and consumer load have the same access pattern *and* affine map;
        otherwise the data is left in a materialized buffer):
          - order-match m_d: the *relative* order of the shared tensor dims, not
            their absolute nest positions. m_d compares the rank of dim d among
            the edge's dims in the producer nest to its rank in the consumer
            nest -- m_d -> 1 when the consumer reads dim d in the same relative
            order the producer emits it, -> 0 on a genuine reorder.
          - tile-match t_d: the producer and consumer tiling factors on dim d.
            t_d -> 1 when they are equal and -> 0 as they diverge; unequal tiles
            make the affine maps differ, so StreamHLS does not fuse and the dim
            (and everything inner to it) buffers (extent D instead of max(a,b)).
        Reuse loops -- outer consumer loops
        that don't index this tensor -- are not edge dims, so they shift
        absolute positions but never the relative ranks, and so never inflate
        fill. This is what lets a re-read (reuse) buffer fuse while a reorder
        (transpose-like) buffer serializes, despite identical BRAM size.
        """
        info, rank_in, rank_out, pos_in = self._edge_ranks(edge, params)
        if not info:
            return torch.tensor(1.0)  # no streamable dims to gate on: fill = 1
        tile_src = params.tile[edge["src"]]
        tile_dst = params.tile[edge["dst"]]
        trip_src = [float(l["trip_count"]) for l in self.nodes[edge["src"]]["loops"]]

        # Visit dims in producer stream order (outermost first).
        order = sorted(range(len(info)), key=lambda i: pos_in[i].item())

        cum_m = torch.tensor(1.0)  # prod of m_{d'} for d' <= d
        cum_t = torch.tensor(1.0)  # prod of t_{d'} for d' <= d
        fill_num = torch.tensor(1.0)
        a_prod = torch.tensor(1.0)
        for i in order:
            pl, cl = info[i]
            a = tile_src[pl]
            b = tile_dst[cl]
            D = trip_src[pl]  # == trip_dst[cl] by the extent-match filter above

            m = torch.exp(-((rank_in[i] - rank_out[i]) ** 2)
                          / (2.0 * self.sigma_r ** 2))
            t = torch.exp(-((torch.log(a) - torch.log(b)) ** 2)
                          / (2.0 * self.sigma_t ** 2))
            cum_m = cum_m * m
            cum_t = cum_t * t
            S = cum_m * cum_t  # streamability mask in [0,1]

            extent = torch.maximum(a, b) ** S * (torch.tensor(D) ** (1.0 - S))
            fill_num = fill_num * extent
            a_prod = a_prod * a

        return fill_num / a_prod

    def analyze(self, params):
        """Forward timing walk. Returns {total_cycles, per_node, per_edge}."""
        order = self._topo_order()
        intr = {nid: self._node_intrinsics(nid, params) for nid in order}

        st = {}
        per_edge = {}
        for nid in order:
            ins = self.in_edges[nid]
            if not ins:
                st[nid] = torch.tensor(0.0)
                continue
            cands = []
            for e in ins:
                p = e["src"]
                _, t_op_p, _ = intr[p]
                fill = self._edge_fill(e, params)
                per_edge[e["id"]] = fill
                cands.append(st[p] + fill * t_op_p)
            st[nid] = torch.stack(cands).max() if len(cands) > 1 else cands[0]

        per_node = {}
        lw = {}
        total = torch.tensor(0.0)
        for nid in order:
            n_out, t_op, ii = intr[nid]
            fw = st[nid] + t_op
            lw_n = st[nid] + n_out * t_op
            # Causality/rate clamp: a consumer cannot finish before its slowest
            # producer has emitted its last tile (which the consumer then takes
            # one T_op to process). Without this a node that is faster per
            # tile than its producer appears to finish before its inputs exist.
            for e in self.in_edges[nid]:
                lw_n = torch.maximum(lw_n, lw[e["src"]] + t_op)
            lw[nid] = lw_n
            per_node[nid] = {"st": st[nid], "fw": fw, "lw": lw_n,
                             "T_op": t_op, "N_out": n_out, "II": ii}
            if not self.out_edges[nid]:  # graph sink: contributes to total latency
                total = torch.maximum(total, lw_n)

        return {"total_cycles": total, "per_node": per_node, "per_edge": per_edge}


def load_graph(path):
    with open(path) as f:
        return json.load(f)


def _nearest_divisor(trip, value, limit):
    """Largest-fitting tiling factor: the divisor of `trip` that is <= `limit`
    and closest to the continuous `value` (ties broken toward the smaller)."""
    trip = int(round(trip))
    cap = min(trip, int(limit))
    divs = [d for d in range(1, cap + 1) if trip % d == 0]
    return min(divs, key=lambda d: (abs(d - value), d))


def leading_singleton_boundary_nodes(graph, mlir_path):
    """Graph nodes that access a leading-singleton function-argument tensor
    (rank >= 3 with a leading unit dim, e.g. a batch-1 `tensor<1x64x128>`).

    Such a tensor decays to a pointer at the top-level HLS interface
    (`float v[1][64][128]` -> `float (*)[64][128]`), so Vitis can partition at
    most one of its non-leading dims; tiling more than one loop of such a node
    emits a multi-dim array_partition on the arg that csynth rejects with
    "Pointer cast is not supported". 2D/1D args (and rank-3+ args without a
    leading unit dim) are unaffected, so they are not constrained.

    Shapes are read from the input MLIR's `func.func @forward(...) -> ...`
    signature (the func args / results carry the original tensor ranks, which the
    *_space.json graph does not). A boundary node (graph source or sink, the only
    nodes that touch func args) is matched to such an arg by its non-unit loop
    trip counts, since the unit dim is dropped from the loop nest."""
    import re

    text = open(mlir_path).read()
    m = re.search(r"func\.func @forward\((.*?)\)\s*->\s*(.*?)\s*\{", text, re.S)
    assert m, f"no 'func.func @forward' signature in {mlir_path}"
    targets = set()
    for shape in re.findall(r"tensor<([0-9x]+)x[a-z][a-z0-9]*>",
                            m.group(1) + " " + m.group(2)):
        dims = [int(d) for d in shape.split("x")]
        if len(dims) >= 3 and dims[0] == 1:
            targets.add(tuple(sorted(d for d in dims if d != 1)))
    if not targets:
        return set()

    interior = {e["src"] for e in graph["edges"]} & {e["dst"] for e in graph["edges"]}
    out = set()
    for n in graph["nodes"]:
        if n["id"] in interior:
            continue
        trips = tuple(sorted(int(l["trip_count"]) for l in n["loops"]
                             if l["trip_count"] != 1))
        if trips in targets:
            out.add(n["id"])
    return out


def _reconcile_fused_tiles(graph, out, params, tiling_limit, single_tile_nodes):
    """Snap each edge's shared dims to a single common tiling factor on both the
    producer and consumer loop, so a continuous near-fused edge survives rounding
    -- StreamHLS only fuses (storeMap == loadMap) when the shared dims are tiled
    identically, and rounding each node independently can split a near-equal pair
    (e.g. 7 -> 6 vs 8) and break the stream. The common factor is the divisor
    nearest the geometric mean of the two continuous tile sizes.

    single_tile_nodes are skipped: their interface partition is already pinned to
    the innermost loop and must not be overridden. A loop shared across several
    edges takes the last edge's value, which is correct for the linear fusion
    chains these graphs form."""
    onodes = {n["id"]: n for n in out["nodes"]}
    nodes = {n["id"]: n for n in graph["nodes"]}
    for e in graph["edges"]:
        src, dst = e["src"], e["dst"]
        if src in single_tile_nodes or dst in single_tile_nodes:
            continue
        trip_src = [l["trip_count"] for l in nodes[src]["loops"]]
        trip_dst = [l["trip_count"] for l in nodes[dst]["loops"]]
        for dim in e["dims"]:
            pl, cl = dim["producer_loop"], dim["consumer_loop"]
            if pl is None or cl is None:
                continue
            if not math.isclose(float(trip_src[pl]), float(trip_dst[cl])):
                continue
            geomean = math.sqrt(params.tile[src][pl].item()
                                * params.tile[dst][cl].item())
            common = _nearest_divisor(trip_src[pl], geomean, tiling_limit)
            onodes[src]["loops"][pl]["tiling_factor"] = common
            onodes[dst]["loops"][cl]["tiling_factor"] = common


def round_design(graph, params, tiling_limit, single_tile_nodes=(),
                 fuse_align=True):
    """Snap continuous DesignParams to the nearest valid design point and return
    a new graph dict (same schema as the *_space.json input). Soft permutations
    become the bijection maximizing the assignment score; each loop's tiling
    factor is rounded to a divisor of its trip count <= tiling_limit.

    When `fuse_align` (default), a final pass reconciles the tiling factors of
    each edge's shared dims to a single common divisor on both sides (see
    _reconcile_fused_tiles), so a continuous point the optimizer drove toward
    fusion is not split apart by independent per-loop rounding. The relative loop
    order needed for fusion is preserved automatically: the per-node bijection is
    the arg-max of soft permutations the fusion_penalty already pulls into
    agreement. With `fuse_align` off, tilings round independently and a mismatch
    simply leaves the data in a materialized on-chip buffer (still valid HLS, and
    charged by the model's tile-match streamability term in _edge_fill).

    `single_tile_nodes` (see leading_singleton_boundary_nodes) are nodes whose
    func-arg array cannot take a multi-dim interface partition. Vitis only
    accepts a partition on the *innermost* dim of such a pointer-decayed array
    (the working designs all tile the innermost dim; a middle dim raises "Pointer
    cast is not supported"). For these nodes we therefore tile only the last loop
    -- which indexes the innermost tensor dim -- carrying the largest requested
    factor of the node so its parallelism degree is preserved, and reset the
    rest to 1."""
    import copy
    import itertools

    single_tile_nodes = set(single_tile_nodes)
    out = copy.deepcopy(graph)
    for node in out["nodes"]:
        nid = node["id"]
        factors = [_nearest_divisor(l["trip_count"], params.tile[nid][i].item(),
                                    tiling_limit)
                   for i, l in enumerate(node["loops"])]
        if nid in single_tile_nodes and sum(f > 1 for f in factors) > 1:
            inner = len(factors) - 1  # last loop -> innermost tensor dim
            keep = _nearest_divisor(node["loops"][inner]["trip_count"],
                                    max(factors), tiling_limit)
            factors = [keep if i == inner else 1 for i in range(len(factors))]
        for i, l in enumerate(node["loops"]):
            l["tiling_factor"] = factors[i]
        P = params.soft_perm(nid).detach()
        n = P.shape[0]
        best = max(itertools.permutations(range(n)),
                   key=lambda perm: sum(P[i, perm[i]].item() for i in range(n)))
        node["permutation"] = list(best)
    if fuse_align:
        _reconcile_fused_tiles(graph, out, params, tiling_limit, single_tile_nodes)
    return out


if __name__ == "__main__":
    import sys

    base = ("designs/polybench/gemm2/gemm/mlir/intermediates/")
    paths = sys.argv[1:] or [base + "gemm_space.json", base + "gemm_space2.json"]
    for path in paths:
        graph = load_graph(path)
        params = DesignParams(graph)
        model = DFGTiming(graph)
        out = model.analyze(params)
        total = out["total_cycles"]
        print(f"\n=== {path} ===")
        for nid, info in sorted(out["per_node"].items()):
            print(f"  node {nid}: st={info['st'].item():.1f} "
                  f"fw={info['fw'].item():.1f} lw={info['lw'].item():.1f} "
                  f"T_op={info['T_op'].item():.3f} "
                  f"N_out={info['N_out'].item():.1f}")
        for eid, fill in sorted(out["per_edge"].items()):
            print(f"  edge {eid}: fill={fill.item():.2f}")
        print(f"  total_cycles = {total.item():.1f}")

        total.backward()
        g = params.tile[0].grad
        print(f"  d(total)/d(tile[node0]) = {g.tolist()}")
