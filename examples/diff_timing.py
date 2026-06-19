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

    fw(p) = st(p) + T_fire(p)             # first output tile
    lw(p) = st(p) + N_out(p) * T_fire(p)  # last output tile

and `fill in [1, N_out(p)]` slides the consumer start smoothly between them:

    st(n) = max over in-edges (p -> n) of  st(p) + fill_e * T_fire(p)

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

II = 1.0  # see module docstring: reduction-loop-innermost penalty not yet modeled

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

    INIT_SCALE = 4.0  # logit peak so the init soft permutation ~matches the JSON

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

    def _node_intrinsics(self, nid, params):
        """Return (N_out, T_fire) for a node, differentiable in tile sizes.

        StreamHLS pipelines the outer (tiled) loop band at II=1, which fully
        unrolls the inner tile loops (bound = tiling factor) into a spatial
        datapath. So a node's cycle count is the *outer* iteration product
        Pi(trip/tile), not the full Pi(trip): the tiling factor is the
        parallelization (unroll) knob and larger tiles lower latency.

        N_out  = product over output loops of trip/tile   (output tiles)
        T_fire = II * Pi(trip/tile over all loops) / N_out (outer cycles per tile)
               => lw = N_out * T_fire = II * Pi(trip/tile), tile-dependent.

        NOTE: partition_factor has no effect on timing -- array partitioning is
        only memory banking; the emitted partition factor follows the tiling
        factor regardless. The reduction-loop-innermost II penalty (II>1) is
        still not modeled (see module docstring).
        """
        loops = self.nodes[nid]["loops"]
        trip = [float(l["trip_count"]) for l in loops]
        tile = params.tile[nid]
        out_loops = self._output_loops(nid)

        n_out = _prod([trip[d] / tile[d] for d in out_loops])
        total_outer = _prod([trip[d] / tile[d] for d in range(len(loops))])
        t_fire = II * total_outer / n_out
        return n_out, t_fire

    def _edge_fill(self, edge, params):
        """Differentiable buffer-fill count for an edge: number of producer
        output tiles that must be emitted before the consumer can start.

        fill = (BufSize / 2) / (elem_bytes * prod(producer tile sizes))
             = prod_d max(a_d,b_d)^S_d * D_d^(1-S_d) / prod_d a_d

        The /2 drops the ping-pong factor: the consumer starts once one buffer
        half is full. The constant elem_bytes cancels. fill in [~1, N_out(p)].

        Streamability uses the *relative* order of the shared tensor dims, not
        their absolute nest positions. The order-match m_d compares the rank of
        dim d among the edge's dims in the producer nest to its rank in the
        consumer nest: m_d -> 1 when the consumer reads dim d in the same
        relative order the producer emits it (streams / fuses), and -> 0 on a
        genuine reorder (must materialize). Reuse loops -- outer consumer loops
        that don't index this tensor -- are not edge dims, so they shift
        absolute positions but never the relative ranks, and so never inflate
        fill. This is what lets a re-read (reuse) buffer fuse while a reorder
        (transpose-like) buffer serializes, despite identical BRAM size.
        """
        src, dst = edge["src"], edge["dst"]
        p_in = params.soft_perm(src)
        p_out = params.soft_perm(dst)
        tile_src = params.tile[src]
        tile_dst = params.tile[dst]
        trip_src = [float(l["trip_count"]) for l in self.nodes[src]["loops"]]
        trip_dst = [float(l["trip_count"]) for l in self.nodes[dst]["loops"]]

        # Expected nest position (0 = outermost) of each dim's producer and
        # consumer loop, and the (pl, cl) pair for tile/extent lookups.
        #
        # A dim is only considered when it maps 1:1 to a single loop on *both*
        # sides with matching extents. The two ways that fails are both reshapes
        # that linearize one tensor dim across several loops in row-major order:
        #   - a null mapping (producer_loop or consumer_loop is None): a
        #     non-single-induction-var access on that side (e.g. MHSA reads a
        #     producer dim of extent 128 as consumer loops 8x16, expr d0*16+d2);
        #   - matched loops whose extents disagree: the mirror case, where the
        #     producer emits dims 8 and 16 that the consumer reads as one loop
        #     of 128, so several producer dims map to the same consumer loop.
        # Such an access preserves emission order and streams, so the dim is left
        # out of the rank/extent product, contributing a neutral fill factor of
        # 1. (A true transpose/broadcast would also map to None, but Stream-HLS
        # only emits a dataflow graph for a legal applied point, where it
        # streams.)
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
            return torch.tensor(1.0)  # no streamable dims to gate on: fill = 1
        pos_in = torch.stack(pos_in)
        pos_out = torch.stack(pos_out)

        # Soft rank of each dim within the edge (count of dims at a smaller
        # position), in the producer and consumer nests respectively.
        def soft_rank(pos):
            diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # diff[i,j] = pos_i - pos_j
            gt = torch.sigmoid(diff / self.tau_r)
            return gt.sum(1) - torch.diagonal(gt)       # drop self (sigmoid(0)=.5)
        rank_in = soft_rank(pos_in)
        rank_out = soft_rank(pos_out)

        # Visit dims in producer stream order (outermost first).
        order = sorted(range(len(info)), key=lambda i: pos_in[i].item())

        cum_m = torch.tensor(1.0)  # prod of m_{d'} for d' <= d
        cum_t = torch.tensor(1.0)  # prod of t_{d'} for d' <  d
        fill_num = torch.tensor(1.0)
        a_prod = torch.tensor(1.0)
        for i in order:
            pl, cl = info[i]
            a = tile_src[pl]
            b = tile_dst[cl]
            D = trip_src[pl]  # == trip_dst[cl] by the extent-match filter above

            m = torch.exp(-((rank_in[i] - rank_out[i]) ** 2)
                          / (2.0 * self.sigma_r ** 2))
            cum_m = cum_m * m
            S = cum_m * cum_t  # streamability mask in [0,1]

            extent = torch.maximum(a, b) ** S * (torch.tensor(D) ** (1.0 - S))
            fill_num = fill_num * extent
            a_prod = a_prod * a

            t = torch.exp(-((torch.log(a) - torch.log(b)) ** 2)
                          / (2.0 * self.sigma_t ** 2))
            cum_t = cum_t * t

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
                _, t_fire_p = intr[p]
                fill = self._edge_fill(e, params)
                per_edge[e["id"]] = fill
                cands.append(st[p] + fill * t_fire_p)
            st[nid] = torch.stack(cands).max() if len(cands) > 1 else cands[0]

        per_node = {}
        lw = {}
        total = torch.tensor(0.0)
        for nid in order:
            n_out, t_fire = intr[nid]
            fw = st[nid] + t_fire
            lw_n = st[nid] + n_out * t_fire
            # Causality/rate clamp: a consumer cannot finish before its slowest
            # producer has emitted its last tile (which the consumer then takes
            # one T_fire to process). Without this a node that is faster per
            # tile than its producer appears to finish before its inputs exist.
            for e in self.in_edges[nid]:
                lw_n = torch.maximum(lw_n, lw[e["src"]] + t_fire)
            lw[nid] = lw_n
            per_node[nid] = {"st": st[nid], "fw": fw, "lw": lw_n,
                             "T_fire": t_fire, "N_out": n_out}
            if not self.out_edges[nid]:  # graph sink: contributes to total latency
                total = torch.maximum(total, lw_n)

        return {"total_cycles": total, "per_node": per_node, "per_edge": per_edge}


def load_graph(path):
    with open(path) as f:
        return json.load(f)


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
                  f"T_fire={info['T_fire'].item():.3f} "
                  f"N_out={info['N_out'].item():.1f}")
        for eid, fill in sorted(out["per_edge"].items()):
            print(f"  edge {eid}: fill={fill.item():.2f}")
        print(f"  total_cycles = {total.item():.1f}")

        total.backward()
        g = params.tile[0].grad
        print(f"  d(total)/d(tile[node0]) = {g.tolist()}")
