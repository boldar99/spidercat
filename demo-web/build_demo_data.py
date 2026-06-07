from __future__ import annotations

import json
import math
import re
from pathlib import Path

import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "demo"
COMPARISON_N_MIN = 8
COMPARISON_N_MAX = 50
COMPARISON_T_MIN = 2
COMPARISON_T_MAX = 7

SPIDERCAT_CIRCUITS = ROOT / "spidercat" / "circuits"
SPIDERCAT_GRAPHS = ROOT / "spidercat" / "circuits_data"
FLAG_CIRCUITS = ROOT / "spidercat" / "flag_at_origin_circuits"
MQT_CIRCUITS = ROOT / "spidercat" / "MQT_circuits"
SIM_DATA = ROOT / "spidercat" / "simulation_data"

RT_VALUES = {
    "1": 0.0,
    "2": 1.0 / 3.0,
    "3": 2.0 / 3.0,
    "4": 5.0 / 6.0,
    "5": 1.0,
}


def density_lower_bound(t: int) -> float:
    if t == 1:
        return math.inf
    a = math.ceil((t + 3) / 2)
    b = math.floor((t + 3) / 2)
    c = math.ceil((t - 3) / 2)
    d = math.floor((t - 3) / 2)
    return (a * b) / (a * b + c * b + d * a)


def minimum_e_and_v(n: int, t: int) -> tuple[int, int]:
    density = density_lower_bound(t)
    e_needed = math.ceil(n / density)
    adjustment = (3 - (e_needed % 3)) % 3
    e_final = e_needed + adjustment
    v_final = (2 * e_final) // 3
    return e_final, v_final


def canonical_edge(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def normalize_positions(raw_positions: dict[int, tuple[float, float]]) -> dict[int, dict[str, float]]:
    xs = [point[0] for point in raw_positions.values()]
    ys = [point[1] for point in raw_positions.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max(max_x - min_x, 1e-9)
    height = max(max_y - min_y, 1e-9)
    scale = max(width, height)

    normalized = {}
    for node, (x, y) in raw_positions.items():
        nx_pos = ((x - min_x) - width / 2) / scale
        ny_pos = ((y - min_y) - height / 2) / scale
        normalized[node] = {"x": round(nx_pos, 4), "y": round(ny_pos, 4)}
    return normalized


def layer_cnots(cnots: list[tuple[int, int]]) -> list[list[list[int]]]:
    next_free_layer: dict[int, int] = {}
    layers: list[list[list[int]]] = []
    for control, target in cnots:
        layer_index = max(next_free_layer.get(control, 0), next_free_layer.get(target, 0))
        while len(layers) <= layer_index:
            layers.append([])
        layers[layer_index].append([control, target])
        next_free_layer[control] = layer_index + 1
        next_free_layer[target] = layer_index + 1
    return layers


def parse_stim_circuit(text: str) -> tuple[list[tuple[int, int]], int]:
    cnots: list[tuple[int, int]] = []
    max_qubit = -1
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        op = tokens[0]
        payload = tokens[1:]
        numeric_targets: list[int] = []
        for token in payload:
            if token.startswith("rec["):
                continue
            try:
                value = int(token)
            except ValueError:
                continue
            numeric_targets.append(value)
            max_qubit = max(max_qubit, value)

        if op in {"CX", "CNOT"}:
            for index in range(0, len(numeric_targets), 2):
                cnots.append((numeric_targets[index], numeric_targets[index + 1]))
    return cnots, max_qubit + 1


def parse_qasm_circuit(text: str) -> tuple[list[tuple[int, int]], int]:
    register_offsets: dict[str, tuple[int, int]] = {}
    next_offset = 0
    cnots: list[tuple[int, int]] = []

    def resolve_qubit(token: str) -> int:
        match = re.fullmatch(r"([A-Za-z_]\w*)\[(\d+)\]", token.strip())
        if not match:
            raise ValueError(f"Unsupported qubit token: {token}")
        register, index_text = match.groups()
        offset, _ = register_offsets[register]
        return offset + int(index_text)

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue

        register_match = re.fullmatch(r"qreg\s+([A-Za-z_]\w*)\[(\d+)\];", line)
        if register_match:
            name, size_text = register_match.groups()
            size = int(size_text)
            register_offsets[name] = (next_offset, size)
            next_offset += size
            continue

        cnot_match = re.fullmatch(
            r"cx\s+([A-Za-z_]\w*\[\d+\])\s*,\s*([A-Za-z_]\w*\[\d+\]);",
            line,
        )
        if cnot_match:
            left, right = cnot_match.groups()
            cnots.append((resolve_qubit(left), resolve_qubit(right)))

    return cnots, next_offset


def metrics_from_cnots(cnots: list[tuple[int, int]], num_qubits: int, n: int) -> dict[str, object]:
    layers = layer_cnots(cnots)
    return {
        "numCx": len(cnots),
        "depth": len(layers) + 2,
        "numQubits": num_qubits,
        "numFlags": max(num_qubits - n, 0),
        "layers": layers,
    }


def parse_spidercat_metrics() -> dict[str, dict[str, object]]:
    metrics: dict[str, dict[str, object]] = {}
    pattern = re.compile(r"cat_state_t([^_]+)_n(\d+)_p1\.stim$")
    for path in sorted(SPIDERCAT_CIRCUITS.glob("cat_state_t*_n*_p1.stim")):
        match = pattern.search(path.name)
        if not match:
            continue
        t_text, n_text = match.groups()
        if t_text == "inf":
            continue
        n = int(n_text)
        t = int(t_text)
        if not (COMPARISON_N_MIN <= n <= COMPARISON_N_MAX and COMPARISON_T_MIN <= t <= COMPARISON_T_MAX):
            continue
        key = f"t{t}-n{n}"
        cnots, num_qubits = parse_stim_circuit(path.read_text())
        metric = metrics_from_cnots(cnots, num_qubits, n)
        metric.pop("layers", None)
        lower_bound = None
        lower_bound_proven = t <= 5
        if t >= 2:
            _, min_vertices = minimum_e_and_v(n, t)
            lower_bound = n + min_vertices + 1
        metric.update(
            {
                "n": n,
                "t": t,
                "sourcePath": path.relative_to(ROOT).as_posix(),
                "lowerBoundCnots": lower_bound,
                "matchesLowerBound": lower_bound is not None and metric["numCx"] == lower_bound,
                "lowerBoundProven": lower_bound_proven,
                "optimalVertexRatio": RT_VALUES.get(str(t)),
                "kind": "actual",
            }
        )
        metrics[key] = metric
    return metrics


def parse_flag_metrics() -> dict[str, dict[str, object]]:
    metrics: dict[str, dict[str, object]] = {}
    pattern = re.compile(r"d(\d+)-q(\d+)-GHZ\.qasm$")
    for path in sorted(FLAG_CIRCUITS.glob("d*-q*-GHZ.qasm")):
        match = pattern.search(path.name)
        if not match:
            continue
        d_text, n_text = match.groups()
        n = int(n_text)
        t = (int(d_text) - 1) // 2
        if not (COMPARISON_N_MIN <= n <= COMPARISON_N_MAX and COMPARISON_T_MIN <= t <= COMPARISON_T_MAX):
            continue
        key = f"t{t}-n{n}"
        cnots, num_qubits = parse_qasm_circuit(path.read_text())
        metric = metrics_from_cnots(cnots, num_qubits, n)
        metric.update(
            {
                "n": n,
                "t": t,
                "sourcePath": path.relative_to(ROOT).as_posix(),
                "kind": "actual",
            }
        )
        metrics[key] = metric
    return metrics


def parse_mqt_metrics() -> dict[str, dict[str, object]]:
    metrics: dict[str, dict[str, object]] = {}
    pattern = re.compile(r"ft_ghz_(\d+)_(\d+)\.stim$")
    for path in sorted(MQT_CIRCUITS.glob("ft_ghz_*_*.stim")):
        match = pattern.search(path.name)
        if not match:
            continue
        n_text, t_text = match.groups()
        n = int(n_text)
        t = int(t_text)
        if not (COMPARISON_N_MIN <= n <= COMPARISON_N_MAX and COMPARISON_T_MIN <= t <= COMPARISON_T_MAX):
            continue
        key = f"t{t}-n{n}"
        cnots, num_qubits = parse_stim_circuit(path.read_text())
        metric = metrics_from_cnots(cnots, num_qubits, n)
        metric.update(
            {
                "n": n,
                "t": t,
                "sourcePath": path.relative_to(ROOT).as_posix(),
                "kind": "actual",
            }
        )
        metrics[key] = metric
    return metrics


def parse_noise_snapshots() -> dict[str, dict[str, object]]:
    mapping = {
        "spidercat": SIM_DATA / "simulation_results_t_n_spider-cat_p1.json",
        "mqt": SIM_DATA / "simulation_results_t_n_MQT_p1.json",
        "flagAtOrigin": SIM_DATA / "simulation_results_t_n_flag-at-origin_p1.json",
    }
    snapshots: dict[str, dict[str, object]] = {name: {} for name in mapping}
    for method_name, path in mapping.items():
        rows = json.loads(path.read_text())
        for row in rows:
            if row["k"] != 0:
                continue
            key = f"t{row['t']}-n{row['n']}"
            snapshots[method_name][key] = {
                "p2": row["p"],
                "acceptanceRate": round(row["acceptance_rate"], 6),
                "cleanGivenAccepted": round(row["probability"], 6),
                "overallCleanRate": round(row["acceptance_rate"] * row["probability"], 6),
                "numAccepted": row["num_accepted"],
                "totalSamples": row["total_samples"],
            }
    return snapshots


def build_spider_graphs() -> dict[str, dict[str, object]]:
    graphs: dict[str, dict[str, object]] = {}
    pattern = re.compile(r"cat_state_t(\d+)_n(\d+)_p1\.json$")
    for path in sorted(SPIDERCAT_GRAPHS.glob("cat_state_t*_n*_p1.json")):
        match = pattern.search(path.name)
        if not match:
            continue
        t_text, n_text = match.groups()
        t = int(t_text)
        n = int(n_text)
        if not (COMPARISON_N_MIN <= n <= COMPARISON_N_MAX and COMPARISON_T_MIN <= t <= COMPARISON_T_MAX):
            continue
        key = f"t{t}-n{n}"
        payload = json.loads(path.read_text())

        graph = nx.from_edgelist(payload["G.edges"])
        forest = nx.from_edgelist(payload["forest"])
        positions = normalize_positions(nx.kamada_kawai_layout(graph))

        marks: dict[tuple[int, int], int] = {}
        for count_text, edges in payload["M_inv"].items():
            count = int(count_text)
            for edge in edges:
                marks[canonical_edge(edge[0], edge[1])] = count

        forest_edges = {canonical_edge(u, v) for u, v in forest.edges()}
        matching = {
            int(node_text): [canonical_edge(edge[0], edge[1]) for edge in edges]
            for node_text, edges in payload["matching"].items()
        }

        node_entries = [
            {
                "id": node,
                "x": positions[node]["x"],
                "y": positions[node]["y"],
                "degree": int(graph.degree(node)),
            }
            for node in sorted(graph.nodes())
        ]
        edge_entries = [
            {
                "u": min(u, v),
                "v": max(u, v),
                "markCount": marks.get(canonical_edge(u, v), 0),
                "inForest": canonical_edge(u, v) in forest_edges,
            }
            for u, v in sorted((canonical_edge(u, v) for u, v in graph.edges()))
        ]
        graphs[key] = {
            "n": n,
            "t": t,
            "numVertices": graph.number_of_nodes(),
            "numEdges": graph.number_of_edges(),
            "nodes": node_entries,
            "edges": edge_entries,
            "matching": {str(node): [[u, v] for u, v in edges] for node, edges in matching.items()},
            "sourcePath": path.relative_to(ROOT).as_posix(),
        }
    return graphs


def build_dataset() -> dict[str, object]:
    spider_metrics = parse_spidercat_metrics()
    flag_metrics = parse_flag_metrics()
    mqt_metrics = parse_mqt_metrics()
    graphs = build_spider_graphs()
    noise = parse_noise_snapshots()

    comparison_ns = list(range(COMPARISON_N_MIN, COMPARISON_N_MAX + 1))
    comparison_ts = list(range(COMPARISON_T_MIN, COMPARISON_T_MAX + 1))
    graph_ns_by_t: dict[str, list[int]] = {}
    for key, entry in graphs.items():
        graph_ns_by_t.setdefault(str(entry["t"]), []).append(entry["n"])
    for t_text in graph_ns_by_t:
        graph_ns_by_t[t_text] = sorted(set(graph_ns_by_t[t_text]))

    return {
        "meta": {
            "title": "SpiderCat CAT State Demo",
            "generatedFrom": {
                "paperAbs": "https://arxiv.org/abs/2603.05391",
                "paperPdf": "https://arxiv.org/pdf/2603.05391",
                "readmePath": "README.md",
                "spidercatCircuitsPath": "spidercat/circuits",
                "spidercatGraphsPath": "spidercat/circuits_data",
                "flagCircuitsPath": "spidercat/flag_at_origin_circuits",
                "mqtCircuitsPath": "spidercat/MQT_circuits",
                "simulationPath": "spidercat/simulation_data",
            },
        },
        "paper": {
            "recursive": {
                "theorem": "Theorem 3.1",
                "summary": "Recursive doubling trades a modest linear CNOT overhead for logarithmic-in-t depth.",
                "cnotFormula": "n(1 + log2(t + 1)) - 2(t + 1)",
                "ancillaFormula": "n / 2",
                "depthFormula": "2 log2(t) + 2",
            },
            "optimal": {
                "proposition": "Proposition 5.4",
                "theorem": "Theorem 5.5",
                "summary": "Generalised CNOT circuits can be lower-bounded by marked 3-regular graph vertex ratios r_t.",
                "cnotLowerBound": "n(r_t + 1) + 1",
                "rtValues": {name: round(value, 6) for name, value in RT_VALUES.items()},
            },
            "shallow": {
                "theorem": "Theorem 5.6",
                "summary": "A shallow construction keeps CNOT depth at 3 while staying linear in n.",
                "cnotFormula": "((29 r_t + 26) / 10) n",
                "ancillaFormula": "((12 r_t + 8) / 5) n",
                "depthFormula": "3",
            },
            "tableSummary": {
                "note": "Table 1 frames the design space as a trade-off between CNOT count, depth, and ancilla usage.",
            },
        },
        "methods": {
            "order": ["spidercat", "recursive", "shallow", "flagAtOrigin", "mqt"],
            "spidercat": {
                "label": "SpiderCat optimal",
                "kind": "repo",
                "optimize": "CNOT count",
                "paperHook": "Sections 5-6",
                "description": "Marked 3-regular graph plus spanning-forest extraction, with concrete circuits bundled in this repo.",
            },
            "recursive": {
                "label": "Recursive",
                "kind": "paper",
                "optimize": "Low depth in t",
                "paperHook": "Theorem 3.1",
                "description": "Recursively fuse smaller CAT states so the ZZ-check layers scale only logarithmically with t.",
            },
            "shallow": {
                "label": "Optimal shallow",
                "kind": "paper",
                "optimize": "Constant depth",
                "paperHook": "Theorem 5.6",
                "description": "Uses extra ancilla to hold depth at 3 while staying linear in n.",
            },
            "flagAtOrigin": {
                "label": "Flag-at-origin",
                "kind": "baseline",
                "optimize": "Simple flagged layout",
                "paperHook": "Repo benchmark",
                "description": "A bundled flagged baseline circuit family used in the repo's benchmark scripts.",
            },
            "mqt": {
                "label": "MQT",
                "kind": "baseline",
                "optimize": "Parallel benchmark baseline",
                "paperHook": "Repo benchmark",
                "description": "Another bundled benchmark family with relatively shallow clean circuits but more ancilla overhead.",
            },
        },
        "controls": {
            "defaultN": 20,
            "defaultT": 4,
            "comparisonNs": comparison_ns,
            "comparisonTs": comparison_ts,
            "graphNsByT": graph_ns_by_t,
        },
        "actualMetrics": {
            "spidercat": spider_metrics,
            "flagAtOrigin": flag_metrics,
            "mqt": mqt_metrics,
        },
        "simulationMetrics": noise,
        "spiderGraphs": graphs,
    }


def main() -> None:
    dataset = build_dataset()
    output = "window.SPIDERCAT_DEMO_DATA = " + json.dumps(dataset, indent=2) + ";\n"
    (DEMO_DIR / "data.js").write_text(output)
    print(f"Wrote {(DEMO_DIR / 'data.js').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
