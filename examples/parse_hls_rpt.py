"""Parse Vitis HLS *.verbose.rpt files for per-node and top-level latency.

forward.verbose.rpt carries the top-level (dataflow) latency plus an Instance
table that maps each `nodeN_U0` instance to its module latency. nodeN.verbose.rpt
carries that node's own loop detail, from which the achieved initiation interval
is read. All latencies are in clock cycles. The HLS module index `nodeN` equals
the DFG node id, so the parsed dict keys line up with diff_timing's per_node.
"""

import re


def _first_data_int(lines, start):
    """First integer in the first '| <int> | ...' table row at/after `start`."""
    for ln in lines[start:]:
        m = re.match(r"\s*\|\s*(\d+)\s*\|", ln)
        if m:
            return int(m.group(1))
    assert False, "no latency data row found in report"


def parse_forward(path):
    """Return {'total': cycles, 'nodes': {node_id: module_latency_cycles}}."""
    with open(path) as f:
        lines = f.readlines()

    lat_i = next(i for i, l in enumerate(lines) if l.strip().startswith("+ Latency:"))
    total = _first_data_int(lines, lat_i)

    # The per-module Instance table lives in the Latency section; bound the scan
    # to it so later sections (utilization/binding tables that also list
    # nodeN_U0 with small counts) are not mistaken for latencies.
    inst_i = next(i for i, l in enumerate(lines) if "* Instance:" in l)
    end_i = next((i for i in range(inst_i + 1, len(lines)) if "====" in lines[i]),
                 len(lines))
    row = re.compile(r"\|\s*node(\d+)_U0\s*\|\s*node\d+\s*\|\s*(\d+)\s*\|")
    nodes = {}
    for l in lines[inst_i:end_i]:
        m = row.search(l)
        if m:
            nodes[int(m.group(1))] = int(m.group(2))
    assert nodes, f"no node instances parsed from {path}"
    return {"total": total, "nodes": nodes}


def parse_node_ii(path):
    """Achieved initiation interval of the node's pipelined loop, or None.

    Loop-detail rows begin with a Vitis nesting marker ('-' or '+'); the columns
    are: name, lat_min, lat_max, iter_lat, II_achieved, II_target, trip, pipelined.
    """
    with open(path) as f:
        lines = f.readlines()
    row = re.compile(
        r"\|\s*[-+]\s*[\w.]+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)\s*\|"
    )
    for l in lines:
        m = row.search(l)
        if m:
            return int(m.group(1))
    return None
