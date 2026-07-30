"""Small executable example for the production ZX-to-Stim synthesizer.

The original prototype in this file assigned random half-edge colours and
contained an incomplete circuit builder.  The implementation now lives in
``spiderstate.zx_synthesis``; this module intentionally remains only as a
reproducible example for notebook users.
"""

from __future__ import annotations

import pyzx as zx
from pyzx.utils import EdgeType, VertexType

from spiderstate.zx_synthesis import synthesize_zx


def make_four_spider_example() -> zx.graph.base.BaseGraph:
    """Build the ``a,b,c,d`` cubic Hadamard-edge diagram from the sketch."""

    graph = zx.Graph()
    a, b, c, d = [
        graph.add_vertex(VertexType.Z, qubit=qubit, row=row)
        for qubit, row in ((0, 1), (1, 2), (0, 2), (1, 1))
    ]
    for u, v in ((a, c), (a, d), (b, c), (b, d)):
        graph.add_edge((u, v), EdgeType.HADAMARD)

    input_a = graph.add_vertex(VertexType.BOUNDARY, qubit=0, row=0)
    input_c = graph.add_vertex(VertexType.BOUNDARY, qubit=1, row=0)
    output_b = graph.add_vertex(VertexType.BOUNDARY, qubit=0, row=3)
    output_d = graph.add_vertex(VertexType.BOUNDARY, qubit=1, row=3)
    for boundary, spider in (
        (input_a, a),
        (input_c, c),
        (output_b, b),
        (output_d, d),
    ):
        graph.add_edge((boundary, spider), EdgeType.SIMPLE)

    graph.set_inputs((input_a, input_c))
    graph.set_outputs((output_b, output_d))
    return graph


def main() -> None:
    graph = make_four_spider_example()
    for strategy in ("gate_count", "depth"):
        result = synthesize_zx(graph, strategy=strategy)
        print(f"\n[{strategy}]")
        print(result.circuit)
        print(result.metrics)


if __name__ == "__main__":
    main()
