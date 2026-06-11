const data = window.SPIDERCAT_DEMO_DATA;

const METHOD_ACCENTS = {
  spidercat: "var(--spider)",
  recursive: "var(--recursive)",
  shallow: "var(--shallow)",
  flagAtOrigin: "var(--flag)",
  mqt: "var(--mqt)",
};

const KIND_LABELS = {
  repo: "Repo circuit",
  paper: "Paper theorem",
  baseline: "External baseline",
};

const state = {
  n: data.controls.defaultN,
  // `requestedT` is the raw slider value the user selected. `t` is the
  // *effective* fault weight actually used to look up bundled circuits, graph
  // data, metrics, captions, and exports. When the requested t lands in the
  // "implied" region (t >= floor(n / 2)) it is reduced "for free" to the
  // largest nontrivial value, so `t` may be smaller than `requestedT`.
  requestedT: data.controls.defaultT,
  t: data.controls.defaultT,
  selectedMethod: "spidercat",
  // User-chosen size of the recursive construction's base-case CAT seed blocks.
  // The effective seed is max(this, t + 1) so each fusion stays transversal.
  recursiveBase: 4,
  recursiveView: "schematic",
  shallowView: "schematic",
  spiderView: "graph",
  flagView: "circuit",
  mqtView: "circuit",
  exportFlags: false,
  graphPositionOverrides: {},
  circuitDragOverrides: {},
  // Per-(n,t) horizontal position overrides for ZZ-measurements in the recursive
  // ZX diagram, so users can drag overlapping measurements apart for readability.
  recursiveZxLayout: {},
  zoomScales: {},
};

// Distinct colours for the parallel ZZ-measurement layers. Each layer runs in
// CNOT depth 1 on a disjoint set of wires, echoing the paper's Figure 4.
const ZZ_LAYER_COLORS = [
  "#1f5fd0",
  "#e2622a",
  "#2e8b57",
  "#7c3aed",
  "#0f9d96",
  "#db2777",
  "#d97706",
  "#2563eb",
];

const refs = {
  nRange: document.getElementById("nRange"),
  nValue: document.getElementById("nValue"),
  tRange: document.getElementById("tRange"),
  tValue: document.getElementById("tValue"),
  stateSummary: document.getElementById("stateSummary"),
  methodCards: document.getElementById("methodCards"),
  detailTitle: document.getElementById("detailTitle"),
  detailSubtitle: document.getElementById("detailSubtitle"),
  visualLegend: document.getElementById("visualLegend"),
  visualHost: document.getElementById("visualHost"),
  visualCaption: document.getElementById("visualCaption"),
  detailInfo: document.getElementById("detailInfo"),
  comparisonLegend: document.getElementById("comparisonLegend"),
  comparisonHost: document.getElementById("comparisonHost"),
  comparisonCaption: document.getElementById("comparisonCaption"),
};

// A point (n, t) is nontrivial when floor(n / 2) > t. If the requested t lies
// in the implied region it can be reduced "for free" to the largest nontrivial
// value, so the demo falls back to that circuit instead of failing to render.
function effectiveTFor(n, t) {
  const maxNontrivial = Math.floor(n / 2) - 1;
  return Math.min(t, Math.max(1, maxNontrivial));
}

// Recompute the effective t from the current n and requested t. Call this
// whenever either control changes before re-rendering.
function syncEffectiveT() {
  state.t = effectiveTFor(state.n, state.requestedT);
}

// True when the requested t was reduced to a smaller effective t for lookup.
function isImpliedT() {
  return state.t !== state.requestedT;
}

// Human-readable explanation shown wherever the requested and effective t differ.
function impliedTNote() {
  if (!isImpliedT()) {
    return "";
  }
  return (
    `Requested t = ${state.requestedT}. For n = ${state.n} this is in the implied region, ` +
    `so the demo displays the t = ${state.t} circuit.`
  );
}

refs.nRange.value = String(state.n);
refs.tRange.value = String(state.requestedT);

refs.nRange.addEventListener("input", (event) => {
  state.n = Number(event.target.value);
  syncEffectiveT();
  render();
});

refs.tRange.addEventListener("input", (event) => {
  state.requestedT = Number(event.target.value);
  syncEffectiveT();
  render();
});

function keyOf(n, t) {
  return `t${t}-n${n}`;
}

function formatNumber(value, digits = 0) {
  if (value == null || Number.isNaN(value)) {
    return "n/a";
  }
  if (digits === 0) {
    return String(Math.round(value));
  }
  return value.toFixed(digits);
}

function formatPercent(value) {
  if (value == null || Number.isNaN(value)) {
    return "n/a";
  }
  return `${(value * 100).toFixed(1)}%`;
}

function recursiveEstimate(n, t) {
  const cnot = Math.max(0, Math.ceil(n * (1 + Math.log2(t + 1)) - 2 * (t + 1)));
  const depth = Math.max(2, Math.ceil(2 * Math.log2(Math.max(t, 1)) + 2));
  const ancillas = Math.ceil(n / 2);
  return {
    numCx: cnot,
    depth,
    ancillas,
    formulaLabel: data.paper.recursive.theorem,
    note: "Estimator from Theorem 3.1.",
    available: true,
  };
}

// Port of experimental/recursive_construction.py, generalised to a binary fusion
// tree whose leaves are CAT base-case seed states (the paper's seed blocks), so
// any target size n is representable. The seed size is max(4, t + 1) so each fuse
// can use (t + 1) transversal ZZ-measurements; n is split into contiguous leaf
// blocks of that size, and a balanced tree fuses adjacent blocks, each internal
// node measuring (t + 1) ZZ between the min-depth qubit on each side. This
// reproduces the .py's min-depth parallelisation and matches it exactly when n is
// a power-of-two multiple of the seed size. Returns ZZ-measurements in .py order.
const RECURSIVE_BASE_SIZE = 4;

// The base/seed CAT block size is a free user choice (the recursion's leaf size).
// Each fusion still performs t + 1 ZZ-measurements, but they reuse the block's
// qubits across rounds (see the round loop below), so the seed need NOT be as
// large as t + 1 — only a valid CAT block (>= 2 qubits). n / baseSize leaves result.
function recursiveBaseSize(requestedBase = RECURSIVE_BASE_SIZE) {
  return Math.max(2, Math.round(requestedBase));
}

function makeLeafBlocks(n, baseSize) {
  const numLeaves = Math.max(1, Math.ceil(n / baseSize));
  const leaves = [];
  let lo = 0;
  for (let i = 0; i < numLeaves; i += 1) {
    const hi = Math.floor((n * (i + 1)) / numLeaves);
    leaves.push({ lo, hi });
    lo = hi;
  }
  return leaves;
}

function buildFusionTree(leaves) {
  if (leaves.length === 1) {
    return { leaf: true, lo: leaves[0].lo, hi: leaves[0].hi, height: 0 };
  }
  const mid = Math.ceil(leaves.length / 2);
  const left = buildFusionTree(leaves.slice(0, mid));
  const right = buildFusionTree(leaves.slice(mid));
  return {
    leaf: false,
    lo: left.lo,
    hi: right.hi,
    mid: left.hi,
    left,
    right,
    height: 1 + Math.max(left.height, right.height),
  };
}

function collectInternalNodes(node, acc) {
  if (node.leaf) {
    return;
  }
  collectInternalNodes(node.left, acc);
  collectInternalNodes(node.right, acc);
  acc.push(node);
}

function argMinDepth(depths, lo, hi) {
  let best = lo;
  for (let q = lo + 1; q < hi; q += 1) {
    if (depths[q] < depths[best]) {
      best = q;
    }
  }
  return best;
}

function buildRecursiveConstruction(nRaw, tRaw, baseRaw = state.recursiveBase) {
  const n = Math.max(2, Math.round(nRaw));
  // FUSE-Nw fuses two sibling blocks with w = t + 1 ZZ-measurements, picking the
  // min-depth qubit on each side per round (qubits are reused across rounds, so
  // the seed size is independent of t). The leaf/seed size is the user's choice;
  // n splits into ceil(n / baseSize) base CAT blocks.
  const t = Math.max(0, Math.round(tRaw));
  const baseSize = recursiveBaseSize(baseRaw);

  const leaves = makeLeafBlocks(n, baseSize);
  const root = buildFusionTree(leaves);
  const internal = [];
  collectInternalNodes(root, internal);

  const byHeight = new Map();
  for (const node of internal) {
    if (!byHeight.has(node.height)) {
      byHeight.set(node.height, []);
    }
    byHeight.get(node.height).push(node);
  }

  const depths = new Array(n).fill(0);
  const measurements = [];
  let maxLayer = 0;
  const heights = [...byHeight.keys()].sort((a, b) => a - b);
  for (const height of heights) {
    const group = byHeight.get(height).slice().sort((a, b) => a.lo - b.lo);
    for (let round = 0; round <= t; round += 1) {
      for (const node of group) {
        const qL = argMinDepth(depths, node.lo, node.mid);
        const qR = argMinDepth(depths, node.mid, node.hi);
        const layer = Math.max(depths[qL], depths[qR]) + 1;
        depths[qL] = layer;
        depths[qR] = layer;
        maxLayer = Math.max(maxLayer, layer);
        measurements.push({ qL, qR, level: height, round, layer });
      }
    }
  }

  return {
    n,
    t,
    baseSize,
    leaves,
    root,
    measurements,
    numFusions: internal.length,
    totalZZ: measurements.length,
    maxLayer,
    levels: heights.length,
  };
}

function recursiveStimText(construction, useFlags) {
  const { n, t, baseSize, measurements, leaves } = construction;
  const lines = [
    `# Recursive fault-tolerant CAT^${n} state preparation (t = ${t})`,
    `# Generated from recursive_construction.py — CAT^${baseSize} base-case binary fusion tree`,
    `# ${leaves.length} base CAT blocks (size <= ${baseSize}) assumed already prepared`,
    `# ${measurements.length} ZZ-measurements fuse them up the tree`,
  ];
  if (useFlags) {
    const numFlags = measurements.length;
    const catOffset = numFlags;
    lines.push(`# qubits 0..${numFlags - 1} are flags; cat qubits are ${catOffset}..${catOffset + n - 1}`);
    let flagIdx = 0;
    for (const m of measurements) {
      lines.push(`CX ${catOffset + m.qL} ${flagIdx}`);
      lines.push(`CX ${catOffset + m.qR} ${flagIdx}`);
      lines.push(`MR ${flagIdx}`);
      flagIdx += 1;
    }
  } else {
    for (const m of measurements) {
      lines.push(`CZ ${m.qL} ${m.qR}`);
    }
  }
  return `${lines.join("\n")}\n`;
}

function downloadTextFile(filename, text) {
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function shallowEstimate(n, t) {
  const rt = data.paper.optimal.rtValues[String(t)];
  if (rt == null) {
    return {
      available: false,
      note: "The shallow theorem depends on known optimal r_t values, which the paper proves explicitly only up to t = 5.",
    };
  }

  return {
    available: true,
    numCx: Math.ceil(((29 * rt + 26) / 10) * n),
    depth: 3,
    ancillas: Math.ceil(((12 * rt + 8) / 5) * n),
    formulaLabel: data.paper.shallow.theorem,
    note: "Estimator from Theorem 5.6 using the paper's r_t values.",
  };
}

// Explicit Theorem 5.6 ("optimal shallow") circuit construction from a bundled
// marked 3-regular graph (G, M). The paragraph preceding the theorem describes
// it directly: put every vertex AND every mark on its own qubit triplet as a
// 3-qubit CAT state (CNOT depth 2), then fuse adjacent spiders along each graph
// edge with a Bell-basis measurement (raising CNOT depth to 3). The free leg of
// each mark spider is an output of the n-qubit CAT state.
//
// Expansion of (G, M): a 3-regular vertex becomes a 3-ary Z-spider; a mark on an
// edge becomes a boundary Z-spider whose extra leg is a data output and which
// splits its edge in two. So an edge (u, v) carrying k marks becomes the chain
//     u — m_1 — m_2 — ... — m_k — v
// contributing k + 1 fusions, and a vertex contributes one leg per incident edge.
//
// Qubit layout: data/output qubits 0..n-1 (one per mark) first, then ancilla
// "leg" qubits. Each spider is a GHZ block (H on its root leg, then a CNOT to
// each of the other two legs); each fusion is a Bell measurement CX p q; H p;
// M p; M q that consumes the two paired leg qubits. The Pauli corrections from
// the random measurement outcomes are tracked in the classical frame, so the
// data wires hold the cat state |0…0> + |1…1>.
function buildShallowConstruction(entry) {
  const totalMarks = entry.edges.reduce((sum, edge) => sum + (edge.markCount || 0), 0);

  // Data outputs occupy 0..totalMarks-1; ancilla leg qubits are allocated after.
  let nextAncilla = totalMarks;
  const allocAncilla = () => nextAncilla++;

  // Each incident edge-end consumes one fresh leg qubit on its vertex.
  const vertexLegs = new Map();
  function vertexLeg(id) {
    const qubit = allocAncilla();
    if (!vertexLegs.has(id)) {
      vertexLegs.set(id, []);
    }
    vertexLegs.get(id).push(qubit);
    return qubit;
  }

  const spiders = []; // { kind, root, legs: [q0, q1, q2] }
  const fusions = []; // [legA, legB] Bell-measured pairs
  let markCounter = 0;

  for (const edge of entry.edges) {
    const marks = edge.markCount || 0;
    const uLeg = vertexLeg(edge.u);
    const vLeg = vertexLeg(edge.v);
    if (marks === 0) {
      fusions.push([uLeg, vLeg]);
      continue;
    }
    // Build the mark chain u — m_1 — ... — m_k — v.
    let dangling = uLeg;
    for (let i = 0; i < marks; i += 1) {
      const output = markCounter++;
      const leftLeg = allocAncilla();
      const rightLeg = allocAncilla();
      // root = output so the data wire owns the GHZ |+> origin.
      spiders.push({ kind: "mark", root: output, legs: [output, leftLeg, rightLeg] });
      fusions.push([dangling, leftLeg]);
      dangling = rightLeg;
    }
    fusions.push([dangling, vLeg]);
  }

  for (const node of entry.nodes) {
    const legs = vertexLegs.get(node.id) || [];
    spiders.push({ kind: "vertex", root: legs[0], legs });
  }

  const numQubits = nextAncilla;
  const dataQubits = totalMarks;
  const cnotCount = spiders.reduce((sum, s) => sum + (s.legs.length - 1), 0) + fusions.length;

  // CNOT depth: GHZ blocks take depth 2 (root drives two CNOTs); every leg qubit
  // is then Bell-fused exactly once, so the fusion CNOTs all fit in one further
  // layer -> depth 3, matching the theorem.
  const depth = 3;

  return {
    n: dataQubits,
    t: entry.t,
    spiders,
    fusions,
    numQubits,
    dataQubits,
    cnotCount,
    depth,
    ancillaCount: numQubits - dataQubits,
    numVertices: entry.nodes.length,
    numMarkSpiders: totalMarks,
  };
}

function shallowStimText(construction, entry) {
  const { n, t, spiders, fusions, numQubits, cnotCount, ancillaCount } = construction;
  const rt = data.paper.optimal.rtValues[String(t)];
  const optimalCx = rt != null ? Math.ceil(((29 * rt + 26) / 10) * n) : null;

  const lines = [
    `# Theorem 5.6 "optimal shallow" CAT^${n} state preparation (t = ${t})`,
    `# Built from marked 3-regular graph ${entry.sourcePath || `t${t}-n${n}`}`,
    `#   ${construction.numVertices} vertices + ${construction.numMarkSpiders} marks`,
    `#   -> ${spiders.length} three-qubit CAT spiders fused by ${fusions.length} Bell measurements`,
    `# CNOT depth 3, ${cnotCount} CNOTs, ${ancillaCount} ancillae` +
      (optimalCx != null ? ` (theorem-optimal CNOT count ${optimalCx} after the direct-CNOT pass)` : ""),
    `# Data qubits 0..${n - 1} carry the output cat state; Pauli frame tracks the random Bell outcomes.`,
  ];

  // Initialise every qubit, then prepare each 3-qubit CAT spider (depth 2).
  const allQubits = Array.from({ length: numQubits }, (_, q) => q).join(" ");
  lines.push(`R ${allQubits}`);
  lines.push("TICK");
  for (const spider of spiders) {
    lines.push(`H ${spider.root}`);
  }
  for (const spider of spiders) {
    for (const leg of spider.legs) {
      if (leg !== spider.root) {
        lines.push(`CX ${spider.root} ${leg}`);
      }
    }
  }
  lines.push("TICK");
  // Fuse adjacent spiders with Bell-basis measurements (depth 3).
  for (const [a, b] of fusions) {
    lines.push(`CX ${a} ${b}`);
  }
  for (const [a, b] of fusions) {
    lines.push(`H ${a}`);
    lines.push(`M ${a} ${b}`);
  }
  return `${lines.join("\n")}\n`;
}

function getActualMetric(methodId, n, t) {
  return data.actualMetrics[methodId]?.[keyOf(n, t)] || null;
}

function getSpiderCircuit(n, t) {
  return data.spiderCircuits?.[keyOf(n, t)] || null;
}

function getMqtCircuit(n, t) {
  return data.mqtCircuits?.[keyOf(n, t)] || null;
}

// Parse the bundled SpiderCat Stim circuit into an ordered op list plus the
// initial single-qubit state of each wire. A leading H on a fresh qubit folds
// into a |+> ket (the cat-state origin); everything else starts in |0>.
function parseStimCircuit(text) {
  const ops = [];
  const initialKet = {};
  const touched = new Set();
  let maxQubit = -1;

  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#") || line.startsWith("DETECTOR")) {
      continue;
    }
    const tokens = line.split(/\s+/);
    const op = tokens[0];
    const nums = [];
    for (const token of tokens.slice(1)) {
      if (token.startsWith("rec[")) {
        continue;
      }
      const value = Number(token);
      if (Number.isInteger(value)) {
        nums.push(value);
        maxQubit = Math.max(maxQubit, value);
      }
    }

    if (op === "H") {
      for (const qubit of nums) {
        if (!touched.has(qubit)) {
          initialKet[qubit] = "+";
        } else {
          ops.push({ type: "h", qubit });
        }
        touched.add(qubit);
      }
    } else if (op === "CX" || op === "CNOT") {
      for (let index = 0; index + 1 < nums.length; index += 2) {
        const control = nums[index];
        const target = nums[index + 1];
        ops.push({ type: "cx", control, target });
        touched.add(control);
        touched.add(target);
      }
    } else if (op === "M" || op === "MZ" || op === "MX") {
      const basis = op === "MX" ? "X" : "Z";
      for (const qubit of nums) {
        ops.push({ type: "m", qubit, basis });
        touched.add(qubit);
      }
    }
  }

  const numQubits = maxQubit + 1;
  for (let qubit = 0; qubit < numQubits; qubit += 1) {
    if (!(qubit in initialKet)) {
      initialKet[qubit] = "0";
    }
  }
  return { ops, numQubits, initialKet };
}

// Layout tuned for the optimal-shallow circuit. Two rules beyond plain ASAP:
//   1. Every Hadamard is forced into a single shared column placed after the
//      last CNOT, so the H gates read as one clean layer.
//   2. Two CNOTs may only share a column when their vertical spans
//      [min(control, target), max(control, target)] are disjoint, so connectors
//      never draw on top of one another. This is interval-graph packing, so the
//      column count is the minimum needed to keep the CNOTs visually separated.
// The circuit gets wider as a result, which the scrollable viewport handles.
function scheduleShallowOps(gateOps, numQubits) {
  const freeAt = new Array(numQubits).fill(0);
  const columnSpans = []; // columnSpans[col] = array of [lo, hi] CNOT spans
  let maxCxCol = -1;

  for (const op of gateOps) {
    if (op.type !== "cx") {
      continue;
    }
    const lo = Math.min(op.control, op.target);
    const hi = Math.max(op.control, op.target);
    let col = Math.max(freeAt[op.control], freeAt[op.target]);
    for (;;) {
      const spans = columnSpans[col] || (columnSpans[col] = []);
      const overlaps = spans.some(([a, b]) => lo <= b && a <= hi);
      if (!overlaps) {
        spans.push([lo, hi]);
        break;
      }
      col += 1;
    }
    op.col = col;
    freeAt[op.control] = col + 1;
    freeAt[op.target] = col + 1;
    maxCxCol = Math.max(maxCxCol, col);
  }

  const hOps = gateOps.filter((op) => op.type === "h");
  const hCol = maxCxCol + 1;
  for (const op of hOps) {
    op.col = hCol;
  }
  return hOps.length ? hCol + 1 : maxCxCol + 1;
}

// ASAP layout with one extra rule: every gate occupies its full vertical span,
// so two gates may share a column only when their spans
// [min(qubit), max(qubit)] are disjoint. A CNOT spans its control..target; a
// single-qubit gate is a one-wire point. As a result no CNOT connector ever
// crosses another gate in the same column, at the cost of more columns (which
// the scrollable viewport handles). Per-wire op ordering is preserved via freeAt.
function scheduleNonOverlappingOps(gateOps, numQubits) {
  const freeAt = new Array(numQubits).fill(0);
  const columnSpans = []; // columnSpans[col] = array of [lo, hi] spans
  let maxCol = 0;

  for (const op of gateOps) {
    const qubits = op.type === "cx" ? [op.control, op.target] : [op.qubit];
    const lo = Math.min(...qubits);
    const hi = Math.max(...qubits);
    let col = 0;
    for (const qubit of qubits) {
      col = Math.max(col, freeAt[qubit]);
    }
    for (;;) {
      const spans = columnSpans[col] || (columnSpans[col] = []);
      const overlaps = spans.some(([a, b]) => lo <= b && a <= hi);
      if (!overlaps) {
        spans.push([lo, hi]);
        break;
      }
      col += 1;
    }
    op.col = col;
    for (const qubit of qubits) {
      freeAt[qubit] = col + 1;
    }
    maxCol = Math.max(maxCol, col);
  }
  return maxCol + 1;
}

function downloadText(fileName, text, mime = "text/plain") {
  const blob = new Blob([text], { type: `${mime};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function getNoiseMetric(methodId, n, t) {
  return data.simulationMetrics[methodId]?.[keyOf(n, t)] || null;
}

function getNearestSpiderGraph(t, n) {
  const exactKey = keyOf(n, t);
  if (data.spiderGraphs[exactKey]) {
    return { entry: data.spiderGraphs[exactKey], exact: true, targetN: n };
  }

  const options = data.controls.graphNsByT[String(t)] || [];
  if (!options.length) {
    return null;
  }

  let nearest = options[0];
  let bestDistance = Math.abs(options[0] - n);
  for (const candidate of options.slice(1)) {
    const distance = Math.abs(candidate - n);
    if (distance < bestDistance) {
      nearest = candidate;
      bestDistance = distance;
    }
  }
  return {
    entry: data.spiderGraphs[keyOf(nearest, t)],
    exact: false,
    targetN: nearest,
  };
}

function buildMethodModel(methodId) {
  const methodMeta = data.methods[methodId];

  if (methodId === "recursive") {
    const estimate = recursiveEstimate(state.n, state.t);
    return {
      id: methodId,
      accent: METHOD_ACCENTS[methodId],
      kindLabel: KIND_LABELS[methodMeta.kind],
      ...methodMeta,
      available: true,
      metrics: {
        numCx: estimate.numCx,
        depth: estimate.depth,
        ancillas: estimate.ancillas,
      },
      note: estimate.note,
      formulaLabel: estimate.formulaLabel,
      estimated: true,
    };
  }

  if (methodId === "shallow") {
    const estimate = shallowEstimate(state.n, state.t);
    return {
      id: methodId,
      accent: METHOD_ACCENTS[methodId],
      kindLabel: KIND_LABELS[methodMeta.kind],
      ...methodMeta,
      available: estimate.available,
      metrics: estimate.available
        ? {
            numCx: estimate.numCx,
            depth: estimate.depth,
            ancillas: estimate.ancillas,
          }
        : null,
      note: estimate.note,
      formulaLabel: estimate.formulaLabel || data.paper.shallow.theorem,
      estimated: true,
    };
  }

  const actual = getActualMetric(methodId, state.n, state.t);
  const noise = getNoiseMetric(methodId, state.n, state.t);
  const spiderGraph = methodId === "spidercat" ? getNearestSpiderGraph(state.t, state.n) : null;
  let note = "Bundled repo circuit.";

  if (methodId === "spidercat") {
    if (actual?.lowerBoundCnots != null) {
      if (actual.matchesLowerBound) {
        note = actual.lowerBoundProven
          ? "Exact circuit matches the density-based lower bound used in the repo and paper-backed for t <= 5."
          : "Circuit matches the repo's density-based lower bound estimate.";
      } else {
        const gap = actual.numCx - actual.lowerBoundCnots;
        note = actual.lowerBoundProven
          ? `${gap} CNOT${gap === 1 ? "" : "s"} above the density-based lower bound.`
          : `${gap} CNOT${gap === 1 ? "" : "s"} above the repo's lower-bound estimate.`;
      }
    } else if (spiderGraph && !spiderGraph.exact) {
      note = `No exact graph file at n = ${state.n}, so the explorer uses the nearest available SpiderCat instance at n = ${spiderGraph.targetN}.`;
    }
  }

  return {
    id: methodId,
    accent: METHOD_ACCENTS[methodId],
    kindLabel: KIND_LABELS[methodMeta.kind],
    ...methodMeta,
    available: Boolean(actual),
    metrics: actual
      ? {
          numCx: actual.numCx,
          depth: actual.depth,
          ancillas: actual.numFlags,
        }
      : null,
    actual,
    noise,
    spiderGraph,
    note: actual ? note : "No bundled circuit for this exact (n, t) point.",
    estimated: false,
  };
}

function buildHighlights(models) {
  const available = models.filter((model) => model.available && model.metrics);
  if (!available.length) {
    return;
  }
  const bestCx = Math.min(...available.map((model) => model.metrics.numCx));
  const bestDepth = Math.min(...available.map((model) => model.metrics.depth));
  const bestAnc = Math.min(...available.map((model) => model.metrics.ancillas));

  for (const model of models) {
    const highlights = [];
    if (!model.available || !model.metrics) {
      model.highlights = highlights;
      continue;
    }
    if (model.metrics.numCx === bestCx) {
      highlights.push("lowest CNOT");
    }
    if (model.metrics.depth === bestDepth) {
      highlights.push("lowest depth");
    }
    if (model.metrics.ancillas === bestAnc) {
      highlights.push("fewest ancillae");
    }
    if (model.id === "spidercat" && model.actual?.matchesLowerBound) {
      highlights.push("bound matched");
    }
    model.highlights = highlights;
  }
}

function renderSummary(models) {
  const impliedSuffix = isImpliedT() ? ` ${impliedTNote()}` : "";
  const available = models.filter((model) => model.available && model.metrics);
  if (!available.length) {
    refs.stateSummary.textContent = `No methods are available at n = ${state.n}, t = ${state.t}.${impliedSuffix}`;
    return;
  }

  const bestCx = [...available].sort((left, right) => left.metrics.numCx - right.metrics.numCx)[0];
  const bestDepth = [...available].sort((left, right) => left.metrics.depth - right.metrics.depth)[0];
  const spider = models.find((model) => model.id === "spidercat");
  const extra =
    spider && spider.available && spider.actual?.matchesLowerBound
      ? " SpiderCat hits the bundled lower bound here."
      : "";

  refs.stateSummary.textContent =
    `At n = ${state.n}, t = ${state.t}, ${bestCx.label} is the cheapest in CNOT count while ${bestDepth.label} is the shallowest construction.${extra}${impliedSuffix}`;
}

function cardHtml(model) {
  const unavailableCard =
    model.id === "shallow"
      ? {
          pill: "Theorem-limited",
          title: "Unavailable for this t",
          body: "The shallow estimator is only wired up where the demo has a known r_t value.",
          hint: "Try t = 2 through t = 5.",
        }
      : {
          pill: "Repo-limited",
          title: "No exact bundle here",
          body: `This demo does not include a ${model.label.toLowerCase()} circuit at n = ${state.n}, t = ${state.t}.`,
          hint: "Try a nearby bundled point or switch constructions.",
        };

  const metricHtml = model.metrics
    ? `
      <div class="metrics-grid">
        <div class="metric-box">
          <span class="metric-label">CNOTs</span>
          <strong>${formatNumber(model.metrics.numCx)}</strong>
        </div>
        <div class="metric-box">
          <span class="metric-label">Depth</span>
          <strong>${formatNumber(model.metrics.depth)}</strong>
        </div>
        <div class="metric-box">
          <span class="metric-label">Anc.</span>
          <strong>${formatNumber(model.metrics.ancillas)}</strong>
        </div>
      </div>
    `
    : `
      <div class="availability-box">
        <span class="availability-pill">${unavailableCard.pill}</span>
        <strong>${unavailableCard.title}</strong>
        <p>${unavailableCard.body}</p>
        <span class="availability-hint">${unavailableCard.hint}</span>
      </div>
    `;

  const highlights = (model.highlights || [])
    .map((highlight) => `<span class="highlight-pill">${highlight}</span>`)
    .join("");

  const linksHtml = (model.links || [])
    .map(
      (link) =>
        `<a class="method-link" href="${link.url}" target="_blank" rel="noreferrer" onclick="event.stopPropagation()">${link.label} &rarr;</a>`,
    )
    .join("");

  const noiseHtml =
    model.noise != null
      ? `<span>p = ${model.noise.p2.toFixed(2)} snapshot: accept ${formatPercent(model.noise.acceptanceRate)}, clean|accepted ${formatPercent(model.noise.cleanGivenAccepted)}.</span>`
      : "<span>No bundled noise snapshot at this exact point.</span>";

  return `
    <article
      class="method-card ${state.selectedMethod === model.id ? "selected" : ""} ${model.available ? "" : "unavailable"}"
      data-method="${model.id}"
      style="border-top: 6px solid ${model.accent};"
    >
      <div class="method-header">
        <div>
          <h3>${model.label}</h3>
          <span class="kind-pill">${model.kindLabel}</span>
        </div>
      </div>
      <p class="method-copy">${model.description}</p>
      ${metricHtml}
      <div class="highlight-strip">${highlights}</div>
      <div class="method-footer">
        <span>${model.paperHook}</span>
        <span>${model.note}</span>
        ${model.estimated ? `<span>${model.formulaLabel}</span>` : noiseHtml}
        ${linksHtml ? `<div class="method-links">${linksHtml}</div>` : ""}
      </div>
    </article>
  `;
}

function renderCards(models) {
  const groups = [
    {
      title: "SpiderCat constructions",
      blurb: "The three constructions introduced in the SpiderCat paper.",
      models: models.filter((model) => data.methods[model.id].kind !== "baseline"),
    },
    {
      title: "External baselines",
      blurb: "Prior-work circuits bundled for comparison.",
      models: models.filter((model) => data.methods[model.id].kind === "baseline"),
    },
  ];

  const visibleGroups = groups.filter((group) => group.models.length);

  // Keep each header interleaved with its own cards in the DOM so narrow
  // screens stack naturally as grouped sections. On wide screens the CSS pins
  // headers to the top row (each spanning only its own columns) and the cards
  // to the row beneath, so all method cards land on a single row.
  let columnCursor = 1;
  refs.methodCards.innerHTML = visibleGroups
    .map((group) => {
      const span = group.models.length;
      const head = `<div class="method-group-head" style="grid-column: ${columnCursor} / span ${span};">
          <h3>${group.title}</h3>
          <p>${group.blurb}</p>
        </div>`;
      columnCursor += span;
      return head + group.models.map(cardHtml).join("");
    })
    .join("");

  refs.methodCards.querySelectorAll(".method-card").forEach((card) => {
    card.addEventListener("click", () => {
      state.selectedMethod = card.dataset.method;
      render();
      // The detail view lives below the cards, so switching a construction
      // changes nothing in view. Scroll it into sight on every selection.
      document.querySelector(".detail-section")?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });
}

function clearVisual() {
  refs.visualLegend.innerHTML = "";
  refs.visualHost.innerHTML = "";
  refs.visualCaption.textContent = "";
}

function legendPills(items, target = refs.visualLegend) {
  target.innerHTML = items
    .map(
      (item) =>
        `<span class="inline-pill"><span style="display:inline-block;width:0.8rem;height:0.8rem;border-radius:999px;background:${item.color};"></span>${item.label}</span>`,
    )
    .join("");
}

function svgNode(name, attrs = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attrs)) {
    node.setAttribute(key, String(value));
  }
  return node;
}

function clampNumber(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function renderZoomableSvg(svg, zoomKey, options = {}) {
  const minScale = options.minScale ?? 1;
  const maxScale = options.maxScale ?? 3;
  const step = options.step ?? 0.25;
  const hintText = options.hint || "Use the zoom controls if the figure feels too small.";
  // When provided, the stage is sized in real pixels (natural width × zoom)
  // instead of a percentage of the viewport, so an oversized figure overflows
  // and both scrollbars appear for panning rather than being squeezed to fit.
  const naturalSize = options.naturalSize || null;
  const host = options.host || refs.visualHost;

  if (state.zoomScales[zoomKey] == null) {
    state.zoomScales[zoomKey] = 1;
  }

  const figure = document.createElement("div");
  figure.className = "zoomable-figure";

  const toolbar = document.createElement("div");
  toolbar.className = "zoom-toolbar";

  const hint = document.createElement("p");
  hint.className = "zoom-hint";
  hint.textContent = hintText;
  toolbar.appendChild(hint);

  const controls = document.createElement("div");
  controls.className = "zoom-controls";

  const zoomOut = document.createElement("button");
  zoomOut.type = "button";
  zoomOut.className = "zoom-button";
  zoomOut.setAttribute("aria-label", "Zoom out");
  zoomOut.textContent = "-";

  const zoomReset = document.createElement("button");
  zoomReset.type = "button";
  zoomReset.className = "zoom-button zoom-readout";
  zoomReset.setAttribute("aria-label", "Reset zoom");

  const zoomIn = document.createElement("button");
  zoomIn.type = "button";
  zoomIn.className = "zoom-button";
  zoomIn.setAttribute("aria-label", "Zoom in");
  zoomIn.textContent = "+";

  controls.append(zoomOut, zoomReset, zoomIn);
  toolbar.appendChild(controls);

  const viewport = document.createElement("div");
  viewport.className = "zoom-viewport";

  const stage = document.createElement("div");
  stage.className = "zoom-stage";
  if (naturalSize) {
    // Let the stage shrink below the viewport width and center when it fits.
    stage.style.minWidth = "0";
    stage.style.marginInline = "auto";
  }
  stage.appendChild(svg);
  viewport.appendChild(stage);

  function applyZoom() {
    const scale = clampNumber(state.zoomScales[zoomKey], minScale, maxScale);
    state.zoomScales[zoomKey] = scale;
    if (naturalSize) {
      stage.style.width = `${Math.round(naturalSize.width * scale)}px`;
    } else {
      stage.style.width = `${scale * 100}%`;
    }
    zoomReset.textContent = `${Math.round(scale * 100)}%`;
    zoomOut.disabled = scale <= minScale + 1e-9;
    zoomIn.disabled = scale >= maxScale - 1e-9;
    zoomReset.disabled = Math.abs(scale - 1) < 1e-9;
  }

  zoomOut.addEventListener("click", () => {
    state.zoomScales[zoomKey] = clampNumber(state.zoomScales[zoomKey] - step, minScale, maxScale);
    applyZoom();
  });

  zoomIn.addEventListener("click", () => {
    state.zoomScales[zoomKey] = clampNumber(state.zoomScales[zoomKey] + step, minScale, maxScale);
    applyZoom();
  });

  zoomReset.addEventListener("click", () => {
    state.zoomScales[zoomKey] = 1;
    applyZoom();
  });

  viewport.addEventListener(
    "wheel",
    (event) => {
      if (!event.ctrlKey && !event.metaKey) {
        return;
      }
      event.preventDefault();
      const direction = event.deltaY < 0 ? 1 : -1;
      state.zoomScales[zoomKey] = clampNumber(
        state.zoomScales[zoomKey] + direction * step,
        minScale,
        maxScale,
      );
      applyZoom();
    },
    { passive: false },
  );

  applyZoom();
  figure.append(toolbar, viewport);
  host.appendChild(figure);
}

function renderSpiderGraph(model) {
  clearVisual();
  const graphBundle = model.spiderGraph;
  if (!graphBundle) {
    refs.visualHost.innerHTML = `<div class="empty-state">No SpiderCat graph is bundled for t = ${state.t}.</div>`;
    refs.visualCaption.textContent = "The graph explorer only covers SpiderCat instances saved in spidercat/circuits_data.";
    return;
  }
  if (!graphBundle.exact) {
    refs.visualHost.innerHTML = `<div class="empty-state">No exact SpiderCat graph is bundled at n = ${state.n}, t = ${state.t}, so there is nothing to draw.</div>`;
    refs.visualCaption.textContent =
      "The graph view renders the exact repo instance, available only at bundled (n, t) points. Try a nearby (n, t).";
    return;
  }

  legendPills([
    { color: "var(--spider-fill)", label: "Z spider (qubit)" },
    { color: "var(--forest)", label: "spanning forest edge" },
    { color: "rgba(20, 33, 61, 0.28)", label: "non-forest edge" },
    { color: "var(--mark)", label: "mark location" },
  ]);

  const entry = graphBundle.entry;
  const width = 980;
  const height = 660;
  const svg = svgNode("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": `${entry.n}-qubit SpiderCat graph for t = ${entry.t}`,
  });

  const rawXs = entry.nodes.map((node) => node.x);
  const rawYs = entry.nodes.map((node) => node.y);
  const minX = Math.min(...rawXs);
  const maxX = Math.max(...rawXs);
  const minY = Math.min(...rawYs);
  const maxY = Math.max(...rawYs);
  const rawWidth = Math.max(maxX - minX, 0.001);
  const rawHeight = Math.max(maxY - minY, 0.001);
  const padding = 90;
  const scale = Math.min((width - padding * 2) / rawWidth, (height - padding * 2) / rawHeight);
  const graphWidth = rawWidth * scale;
  const graphHeight = rawHeight * scale;
  const offsetX = (width - graphWidth) / 2;
  const offsetY = (height - graphHeight) / 2;
  const nodeRadius = entry.nodes.length <= 20 ? 18 : entry.nodes.length <= 32 ? 15 : 12;
  const labelFontSize = entry.nodes.length <= 20 ? 16 : 13;
  const markRadius = nodeRadius * 0.58;
  const multiMarkRadius = nodeRadius * 0.82;
  const graphId = `spidercat-${entry.t}-${entry.n}`;
  if (!state.graphPositionOverrides[graphId]) {
    state.graphPositionOverrides[graphId] = {};
  }
  const savedPositions = state.graphPositionOverrides[graphId];
  for (const node of entry.nodes) {
    if (!savedPositions[node.id]) {
      savedPositions[node.id] = {
        x: offsetX + (node.x - minX) * scale,
        y: offsetY + (node.y - minY) * scale,
      };
    }
  }

  const edgeLayer = svgNode("g");
  const nodeLayer = svgNode("g");
  const edgeVisuals = [];
  const nodeVisuals = new Map();
  let activeDrag = null;

  function positionsMap() {
    return new Map(entry.nodes.map((node) => [node.id, savedPositions[node.id]]));
  }

  function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }

  function graphBounds() {
    const margin = nodeRadius + 12;
    return {
      minX: margin,
      maxX: width - margin,
      minY: margin,
      maxY: height - margin,
    };
  }

  function eventPointInSvg(event) {
    const ctm = svg.getScreenCTM();
    if (!ctm) {
      return null;
    }
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    return point.matrixTransform(ctm.inverse());
  }

  for (const edge of entry.edges) {
    const edgeGroup = svgNode("g");
    const line = svgNode("line", {
      stroke: edge.inForest ? "var(--forest)" : "rgba(20, 33, 61, 0.28)",
      "stroke-width": edge.inForest ? 9 : 4,
      "stroke-linecap": "round",
    });
    edgeGroup.appendChild(line);

    let markCircle = null;
    let markText = null;
    if (edge.markCount > 0) {
      markCircle = svgNode("circle", {
        r: edge.markCount > 1 ? multiMarkRadius : markRadius,
        fill: "var(--mark)",
        stroke: "#fff",
        "stroke-width": 3,
      });
      edgeGroup.appendChild(markCircle);

      if (edge.markCount > 1) {
        markText = svgNode("text", {
          "text-anchor": "middle",
          "font-size": Math.max(12, labelFontSize - 2),
          "font-weight": 700,
          fill: "#fff",
        });
        markText.textContent = String(edge.markCount);
        edgeGroup.appendChild(markText);
      }
    }

    edgeLayer.appendChild(edgeGroup);
    edgeVisuals.push({ edge, line, markCircle, markText });
  }

  const showLabels = entry.nodes.length <= 24;
  for (const node of entry.nodes) {
    const nodeGroup = svgNode("g", {
      class: "graph-node-handle",
      "data-node-id": node.id,
      tabindex: "0",
      role: "button",
      "aria-label": `Move vertex ${node.id}`,
    });
    nodeGroup.appendChild(
      svgNode("circle", {
        cx: 0,
        cy: 0,
        r: nodeRadius,
        // Z spider convention (cf. poster): light-green fill, thin dark-green outline
        fill: "var(--spider-fill)",
        stroke: "var(--spider-stroke)",
        "stroke-width": 2.5,
      }),
    );

    let label = null;
    if (showLabels) {
      label = svgNode("text", {
        "text-anchor": "middle",
        "font-size": labelFontSize,
        "font-weight": 600,
        fill: "var(--ink)",
      });
      label.textContent = String(node.id);
      nodeGroup.appendChild(label);
    }

    nodeGroup.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      activeDrag = {
        nodeId: node.id,
        pointerId: event.pointerId,
      };
      nodeGroup.classList.add("dragging");
      svg.classList.add("dragging-graph");
      nodeGroup.setPointerCapture(event.pointerId);
      updateDraggedNode(event);
    });

    nodeLayer.appendChild(nodeGroup);
    nodeVisuals.set(node.id, { group: nodeGroup, label });
  }

  svg.appendChild(edgeLayer);
  svg.appendChild(nodeLayer);

  function updateGraphVisuals() {
    const positions = positionsMap();
    for (const { edge, line, markCircle, markText } of edgeVisuals) {
      const from = positions.get(edge.u);
      const to = positions.get(edge.v);
      line.setAttribute("x1", from.x);
      line.setAttribute("y1", from.y);
      line.setAttribute("x2", to.x);
      line.setAttribute("y2", to.y);

      if (markCircle) {
        const mx = (from.x + to.x) / 2;
        const my = (from.y + to.y) / 2;
        markCircle.setAttribute("cx", mx);
        markCircle.setAttribute("cy", my);
        if (markText) {
          markText.setAttribute("x", mx);
          markText.setAttribute("y", my + 5);
        }
      }
    }

    for (const node of entry.nodes) {
      const point = positions.get(node.id);
      const visual = nodeVisuals.get(node.id);
      visual.group.setAttribute("transform", `translate(${point.x} ${point.y})`);

      if (visual.label) {
        const dx = point.x - width / 2;
        const dy = point.y - height / 2;
        const mag = Math.max(Math.hypot(dx, dy), 1);
        const labelDistance = nodeRadius + 14;
        visual.label.setAttribute("x", (dx / mag) * labelDistance);
        visual.label.setAttribute("y", (dy / mag) * labelDistance + 5);
      }
    }
  }

  function finishDrag(pointerId) {
    if (!activeDrag || activeDrag.pointerId !== pointerId) {
      return;
    }
    const visual = nodeVisuals.get(activeDrag.nodeId);
    if (visual) {
      visual.group.classList.remove("dragging");
      if (visual.group.hasPointerCapture(pointerId)) {
        visual.group.releasePointerCapture(pointerId);
      }
    }
    svg.classList.remove("dragging-graph");
    activeDrag = null;
  }

  function updateDraggedNode(event) {
    if (!activeDrag || activeDrag.pointerId !== event.pointerId) {
      return;
    }
    const point = eventPointInSvg(event);
    if (!point) {
      return;
    }
    const bounds = graphBounds();
    savedPositions[activeDrag.nodeId] = {
      x: clamp(point.x, bounds.minX, bounds.maxX),
      y: clamp(point.y, bounds.minY, bounds.maxY),
    };
    updateGraphVisuals();
  }

  svg.addEventListener("pointermove", updateDraggedNode);
  svg.addEventListener("pointerup", (event) => {
    finishDrag(event.pointerId);
  });
  svg.addEventListener("pointercancel", (event) => {
    finishDrag(event.pointerId);
  });
  svg.addEventListener("pointerleave", (event) => {
    if (event.buttons === 0) {
      finishDrag(event.pointerId);
    }
  });

  updateGraphVisuals();

  refs.visualHost.appendChild(svg);
  refs.visualCaption.textContent = graphBundle.exact
    ? `Exact SpiderCat graph instance for n = ${entry.n}, t = ${entry.t}. Drag any vertex to explore different embeddings of the same marked 3-regular graph while keeping the forest and mark structure fixed.`
    : `No exact SpiderCat graph is bundled at n = ${state.n}, so this panel shows the nearest available instance at n = ${entry.n}, t = ${entry.t}. Drag any vertex to explore the layout.`;
}

function renderSpiderCircuit(model) {
  clearVisual();
  const circuit = getSpiderCircuit(state.n, state.t);
  if (!circuit) {
    refs.visualHost.innerHTML = `<div class="empty-state">No bundled SpiderCat circuit at n = ${state.n}, t = ${state.t}, so there is nothing to draw.</div>`;
    refs.visualCaption.textContent =
      "The circuit view renders the exact repo circuit, available only at bundled (n, t) points. Try a nearby (n, t).";
    return;
  }

  legendPills([
    { color: "var(--data-wire)", label: "data qubit (output)" },
    { color: "var(--flag-wire)", label: "flag / ancilla" },
    { color: "var(--ink)", label: "CNOT (• control / ⊕ target)" },
    { color: "var(--spider-stroke)", label: "Z measurement" },
  ]);

  const { ops, numQubits, initialKet } = parseStimCircuit(circuit.stim);
  // Schedule only the gates; every measurement is terminal for its wire (verified
  // across all bundled circuits), so they all sit in one final column.
  const gateOps = ops.filter((op) => op.type !== "m");
  const measureOps = ops.filter((op) => op.type === "m");
  // Pack gates so no two CNOTs (or a CNOT and another gate) overlap in a column.
  const numCols = scheduleNonOverlappingOps(gateOps, numQubits);
  const dataQubits = state.n;

  // Comfortable row spacing now that the viewport scrolls instead of squeezing
  // every wire into the visible height.
  const rowGap = numQubits <= 24 ? 26 : numQubits <= 48 ? 18 : numQubits <= 120 ? 13 : 10;
  const colGap = 32;
  const leftPad = 92;
  const rightPad = 36;
  const topPad = 30;
  const bottomPad = 30;

  const colX = (col) => leftPad + col * colGap + colGap / 2;
  const rowY = (qubit) => topPad + qubit * rowGap;

  const dotR = Math.min(4.2, rowGap * 0.26);
  const targetR = Math.min(6, rowGap * 0.34);
  const meterW = Math.min(20, colGap * 0.62);
  const meterH = Math.min(15, rowGap * 0.78);

  // All measurements live in one column past the last gate.
  const meterX = colX(numCols);
  const wireEndX = meterX + meterW / 2 + 8;
  const width = wireEndX + rightPad;
  const height = topPad + numQubits * rowGap + bottomPad;

  // Drag bounds: keep CNOT endpoints inside the gate area, left of the meters.
  const dragMin = leftPad + 8;
  const dragMax = meterX - meterW / 2 - 10;

  const circuitId = `spidercircuit-${state.t}-${state.n}`;
  if (!state.circuitDragOverrides[circuitId]) {
    state.circuitDragOverrides[circuitId] = {};
  }
  const overrides = state.circuitDragOverrides[circuitId];

  const svg = svgNode("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": `SpiderCat optimal circuit for n = ${state.n}, t = ${state.t}`,
  });

  const wireLayer = svgNode("g");
  const gateLayer = svgNode("g");
  const handleLayer = svgNode("g");

  // Wires + ket / index labels.
  for (let qubit = 0; qubit < numQubits; qubit += 1) {
    const y = rowY(qubit);
    const isData = qubit < dataQubits;
    wireLayer.appendChild(
      svgNode("line", {
        x1: leftPad,
        y1: y,
        x2: wireEndX,
        y2: y,
        stroke: isData ? "var(--data-wire)" : "var(--flag-wire)",
        "stroke-width": isData ? 1.8 : 1.3,
        "stroke-opacity": isData ? 0.6 : 0.45,
      }),
    );

    if (numQubits <= 64 || isData || qubit % 4 === 0) {
      const index = svgNode("text", {
        x: 14,
        y: y + 4,
        "font-size": Math.max(9, Math.min(11, rowGap * 0.5)),
        fill: "var(--muted)",
      });
      index.textContent = isData ? `q${qubit}` : `a${qubit - dataQubits}`;
      wireLayer.appendChild(index);
    }

    const ket = svgNode("text", {
      x: leftPad - 10,
      y: y + 4,
      "font-size": Math.max(10, Math.min(13, rowGap * 0.56)),
      "font-weight": 600,
      "text-anchor": "end",
      fill: "var(--ink)",
    });
    ket.textContent = initialKet[qubit] === "+" ? "|+⟩" : "|0⟩";
    wireLayer.appendChild(ket);
  }

  // Hadamard boxes (fixed position).
  for (const op of gateOps) {
    if (op.type !== "h") {
      continue;
    }
    const x = colX(op.col);
    const y = rowY(op.qubit);
    const boxW = Math.min(18, colGap * 0.56);
    const boxH = Math.min(16, rowGap * 0.8);
    gateLayer.appendChild(
      svgNode("rect", {
        x: x - boxW / 2,
        y: y - boxH / 2,
        width: boxW,
        height: boxH,
        rx: 3,
        fill: "#ffffff",
        stroke: "var(--ink)",
        "stroke-width": 1.5,
      }),
    );
    const label = svgNode("text", {
      x,
      y: y + boxH * 0.3,
      "font-size": Math.min(12, boxH * 0.8),
      "font-weight": 700,
      "text-anchor": "middle",
      fill: "var(--ink)",
    });
    label.textContent = "H";
    gateLayer.appendChild(label);
  }

  // Measurement meters, all aligned in the final column.
  for (const op of measureOps) {
    const y = rowY(op.qubit);
    gateLayer.appendChild(
      svgNode("rect", {
        x: meterX - meterW / 2,
        y: y - meterH / 2,
        width: meterW,
        height: meterH,
        rx: 3,
        fill: "#ffffff",
        stroke: "var(--spider-stroke)",
        "stroke-width": 1.5,
      }),
    );
    const arcR = meterW * 0.3;
    const arcY = y + meterH * 0.16;
    gateLayer.appendChild(
      svgNode("path", {
        d: `M ${meterX - arcR} ${arcY} A ${arcR} ${arcR} 0 0 1 ${meterX + arcR} ${arcY}`,
        fill: "none",
        stroke: "var(--spider-stroke)",
        "stroke-width": 1.3,
      }),
    );
    gateLayer.appendChild(
      svgNode("line", {
        x1: meterX,
        y1: arcY,
        x2: meterX + arcR * 0.78,
        y2: arcY - arcR * 0.82,
        stroke: "var(--spider-stroke)",
        "stroke-width": 1.3,
      }),
    );
    if (op.basis === "X") {
      const basisLabel = svgNode("text", {
        x: meterX + meterW * 0.34,
        y: y - meterH * 0.18,
        "font-size": Math.min(8, meterH * 0.5),
        "font-weight": 700,
        "text-anchor": "middle",
        fill: "var(--spider-stroke)",
      });
      basisLabel.textContent = "X";
      gateLayer.appendChild(basisLabel);
    }
  }

  // CNOTs: control dot + target ⊕ joined by a connector. Each endpoint can be
  // dragged horizontally along its own wire so overlapping gates can be spread.
  function eventPointInSvg(event) {
    const ctm = svg.getScreenCTM();
    if (!ctm) {
      return null;
    }
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    return point.matrixTransform(ctm.inverse());
  }

  let gateIndex = 0;
  const gateVisuals = [];
  let activeDrag = null;
  const hitR = Math.max(9, targetR + 4);

  for (const op of gateOps) {
    if (op.type !== "cx") {
      continue;
    }
    const index = gateIndex;
    gateIndex += 1;
    const gate = { index, defaultX: colX(op.col) };
    const yc = rowY(op.control);
    const yt = rowY(op.target);

    const connector = svgNode("line", {
      stroke: "var(--ink)",
      "stroke-width": 1.7,
      "stroke-linecap": "round",
    });
    gateLayer.appendChild(connector);

    const controlHandle = svgNode("g", {
      class: "circuit-gate-handle",
      tabindex: "0",
      role: "button",
      "aria-label": `CNOT ${index} control on q${op.control}`,
    });
    controlHandle.appendChild(svgNode("circle", { cx: 0, cy: 0, r: dotR, fill: "var(--ink)" }));
    controlHandle.appendChild(svgNode("circle", { cx: 0, cy: 0, r: hitR, fill: "transparent" }));

    const targetHandle = svgNode("g", {
      class: "circuit-gate-handle",
      tabindex: "0",
      role: "button",
      "aria-label": `CNOT ${index} target on q${op.target}`,
    });
    targetHandle.appendChild(
      svgNode("circle", { cx: 0, cy: 0, r: targetR, fill: "#ffffff", stroke: "var(--ink)", "stroke-width": 1.7 }),
    );
    targetHandle.appendChild(
      svgNode("line", { x1: -targetR, y1: 0, x2: targetR, y2: 0, stroke: "var(--ink)", "stroke-width": 1.7 }),
    );
    targetHandle.appendChild(
      svgNode("line", { x1: 0, y1: -targetR, x2: 0, y2: targetR, stroke: "var(--ink)", "stroke-width": 1.7 }),
    );
    targetHandle.appendChild(svgNode("circle", { cx: 0, cy: 0, r: hitR, fill: "transparent" }));

    handleLayer.appendChild(controlHandle);
    handleLayer.appendChild(targetHandle);

    function update() {
      const override = overrides[index] || {};
      const cx = override.cx ?? gate.defaultX;
      const tx = override.tx ?? gate.defaultX;
      connector.setAttribute("x1", cx);
      connector.setAttribute("y1", yc);
      connector.setAttribute("x2", tx);
      connector.setAttribute("y2", yt);
      controlHandle.setAttribute("transform", `translate(${cx} ${yc})`);
      targetHandle.setAttribute("transform", `translate(${tx} ${yt})`);
    }

    function startDrag(endpoint, handle, event) {
      event.preventDefault();
      activeDrag = { index, endpoint, pointerId: event.pointerId, update };
      handle.classList.add("dragging");
      svg.classList.add("dragging-graph");
      handle.setPointerCapture(event.pointerId);
    }

    controlHandle.addEventListener("pointerdown", (event) => startDrag("cx", controlHandle, event));
    targetHandle.addEventListener("pointerdown", (event) => startDrag("tx", targetHandle, event));
    // Double-click an endpoint to snap this CNOT back to its scheduled column.
    controlHandle.addEventListener("dblclick", () => {
      delete overrides[index];
      update();
    });
    targetHandle.addEventListener("dblclick", () => {
      delete overrides[index];
      update();
    });

    gateVisuals.push({ controlHandle, targetHandle, update });
    update();
  }

  function moveActiveDrag(event) {
    if (!activeDrag || activeDrag.pointerId !== event.pointerId) {
      return;
    }
    const point = eventPointInSvg(event);
    if (!point) {
      return;
    }
    const x = clampNumber(point.x, dragMin, dragMax);
    overrides[activeDrag.index] = { ...(overrides[activeDrag.index] || {}), [activeDrag.endpoint]: x };
    activeDrag.update();
  }

  function endDrag(event) {
    if (!activeDrag || activeDrag.pointerId !== event.pointerId) {
      return;
    }
    for (const visual of gateVisuals) {
      visual.controlHandle.classList.remove("dragging");
      visual.targetHandle.classList.remove("dragging");
    }
    svg.classList.remove("dragging-graph");
    activeDrag = null;
  }

  svg.addEventListener("pointermove", moveActiveDrag);
  svg.addEventListener("pointerup", endDrag);
  svg.addEventListener("pointercancel", endDrag);

  svg.appendChild(wireLayer);
  svg.appendChild(gateLayer);
  svg.appendChild(handleLayer);

  renderZoomableSvg(svg, circuitId, {
    minScale: 0.2,
    maxScale: 6,
    naturalSize: { width, height },
    hint: "Large circuits render at full size — drag the scrollbars to pan up/down and left/right. Drag a CNOT's control (•) or target (⊕) to nudge it; double-click to reset. Ctrl-scroll to zoom; zoom out to see the whole circuit.",
  });

  refs.visualCaption.textContent =
    `Exact SpiderCat optimal circuit for n = ${state.n}, t = ${state.t} (${gateOps.filter((op) => op.type === "cx").length} CNOTs, ${measureOps.length} measurements), rendered from the bundled ${circuit.fileName}. The origin spider starts in |+⟩; all flag/ancilla wires (a0, a1, …) are measured in the Z basis at the end, and the data wires (q0 … q${state.n - 1}) carry the output cat state. The CNOTs are spread across columns so no two connectors overlap, which can make the diagram wide — drag the scrollbars to pan, or zoom out for the full view. Use Export above to download the Stim source.`;
}

function renderSchedule(metric, dataQubits, accentColor, caption, zoomKey = "schedule") {
  clearVisual();
  legendPills([
    { color: "var(--data-wire)", label: "data qubit" },
    { color: "var(--flag-wire)", label: "flag / ancilla" },
    { color: "var(--ink)", label: "CNOT (control • / target ⊕)" },
  ]);

  const width = Math.max(860, 120 + metric.layers.length * 34);
  const rowGap = 16;
  const height = Math.max(360, 80 + metric.numQubits * rowGap);
  const leftPad = 88;
  const topPad = 34;
  const colGap = 30;
  const svg = svgNode("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": "CNOT layer diagram",
  });

  for (let layer = 0; layer < metric.layers.length; layer += 1) {
    const x = leftPad + layer * colGap;
    if (layer % 2 === 0) {
      svg.appendChild(
        svgNode("rect", {
          x: x - 11,
          y: 18,
          width: colGap,
          height: height - 36,
          fill: "rgba(20, 33, 61, 0.035)",
          rx: 8,
        }),
      );
    }
  }

  for (let qubit = 0; qubit < metric.numQubits; qubit += 1) {
    const y = topPad + qubit * rowGap;
    const color = qubit < dataQubits ? "var(--data-wire)" : "var(--flag-wire)";
    svg.appendChild(
      svgNode("line", {
        x1: leftPad - 8,
        y1: y,
        x2: width - 24,
        y2: y,
        stroke: color,
        "stroke-width": qubit < dataQubits ? 1.8 : 1.3,
        "stroke-opacity": qubit < dataQubits ? 0.35 : 0.28,
      }),
    );

    if (metric.numQubits <= 32 || qubit === 0 || qubit === dataQubits - 1 || qubit % 5 === 0) {
      const label = svgNode("text", {
        x: 16,
        y: y + 4,
        "font-size": 11,
        fill: "var(--muted)",
      });
      label.textContent = qubit < dataQubits ? `q${qubit}` : `a${qubit - dataQubits}`;
      svg.appendChild(label);
    }
  }

  metric.layers.forEach((pairs, layerIndex) => {
    const x = leftPad + layerIndex * colGap;
    const layerLabel = svgNode("text", {
      x,
      y: 14,
      "font-size": 10,
      "text-anchor": "middle",
      fill: "var(--muted)",
    });
    if (metric.layers.length <= 18 || layerIndex % 2 === 0) {
      layerLabel.textContent = String(layerIndex + 1);
      svg.appendChild(layerLabel);
    }

    pairs.forEach(([control, target]) => {
      const y1 = topPad + control * rowGap;
      const y2 = topPad + target * rowGap;
      // CNOT notation (cf. poster): solid control dot --- vertical line --- (+) target,
      // drawn in monochrome ink. Qubit role stays encoded on the wires themselves.
      const gateColor = "var(--ink)";
      svg.appendChild(
        svgNode("line", {
          x1: x,
          y1,
          x2: x,
          y2,
          stroke: gateColor,
          "stroke-width": 1.8,
          "stroke-linecap": "round",
        }),
      );
      // control: filled dot
      svg.appendChild(
        svgNode("circle", {
          cx: x,
          cy: y1,
          r: 4,
          fill: gateColor,
        }),
      );
      // target: (+) — open circle with a cross
      const targetR = 5.6;
      svg.appendChild(
        svgNode("circle", {
          cx: x,
          cy: y2,
          r: targetR,
          fill: "#ffffff",
          stroke: gateColor,
          "stroke-width": 1.8,
        }),
      );
      svg.appendChild(
        svgNode("line", {
          x1: x - targetR,
          y1: y2,
          x2: x + targetR,
          y2: y2,
          stroke: gateColor,
          "stroke-width": 1.8,
        }),
      );
      svg.appendChild(
        svgNode("line", {
          x1: x,
          y1: y2 - targetR,
          x2: x,
          y2: y2 + targetR,
          stroke: gateColor,
          "stroke-width": 1.8,
        }),
      );
    });
  });

  renderZoomableSvg(svg, zoomKey, {
    maxScale: 3.5,
    hint: "Zoom in to inspect individual layers and qubit labels. Ctrl-scroll also works.",
  });
  refs.visualCaption.textContent = caption;
}

function prependViewToggle(options, current, onSelect) {
  const toggle = document.createElement("div");
  toggle.className = "view-toggle";
  options.forEach((option) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `view-toggle-button${option.id === current ? " active" : ""}`;
    button.textContent = option.label;
    button.setAttribute("aria-pressed", String(option.id === current));
    button.addEventListener("click", () => {
      if (option.id !== current) {
        onSelect(option.id);
      }
    });
    toggle.appendChild(button);
  });
  refs.visualHost.insertBefore(toggle, refs.visualHost.firstChild);
}

function appendRecursiveExport() {
  const construction = buildRecursiveConstruction(state.n, state.t);

  const panel = document.createElement("div");
  panel.className = "export-panel";

  const summaryText = (flags) =>
    `Stim circuit: CAT^${construction.n}, t = ${construction.t}, base CAT^${construction.baseSize} — ` +
    `${construction.totalZZ} ZZ-measurements across ${construction.levels} fusion levels` +
    `${flags ? `, ${construction.totalZZ} flag qubits` : ""}.`;

  const summary = document.createElement("p");
  summary.className = "export-summary";
  summary.textContent = summaryText(state.exportFlags);
  panel.appendChild(summary);

  const controls = document.createElement("div");
  controls.className = "export-controls";

  // Base-case CAT size: the leaf/seed block the binary fusion tree starts from
  // (>= 2 qubits, independent of t). Commit on release so the live drag doesn't
  // tear down this panel mid-gesture (render() rebuilds it).
  const minBase = 2;
  const maxBase = 16;
  const baseLabel = document.createElement("label");
  baseLabel.className = "export-base";
  const baseText = document.createElement("span");
  baseText.className = "export-base-label";
  const baseValue = document.createElement("strong");
  baseValue.textContent = `CAT^${construction.baseSize}`;
  baseText.append("Base size ", baseValue);
  const baseInput = document.createElement("input");
  baseInput.type = "range";
  baseInput.min = String(minBase);
  baseInput.max = String(maxBase);
  baseInput.step = "1";
  baseInput.value = String(construction.baseSize);
  baseInput.addEventListener("input", () => {
    baseValue.textContent = `CAT^${baseInput.value}`;
  });
  baseInput.addEventListener("change", () => {
    state.recursiveBase = Number(baseInput.value);
    render();
  });
  baseLabel.append(baseText, baseInput);
  controls.appendChild(baseLabel);

  const flagsLabel = document.createElement("label");
  flagsLabel.className = "export-flags";
  const flagsInput = document.createElement("input");
  flagsInput.type = "checkbox";
  flagsInput.checked = state.exportFlags;
  flagsInput.addEventListener("change", () => {
    state.exportFlags = flagsInput.checked;
    summary.textContent = summaryText(flagsInput.checked);
  });
  const flagsText = document.createElement("span");
  flagsText.textContent = "Flag qubits (CNOT + MR)";
  flagsLabel.append(flagsInput, flagsText);
  controls.appendChild(flagsLabel);

  const fileName = () =>
    `recursive_cat_n${construction.n}_t${construction.t}_base${construction.baseSize}${state.exportFlags ? "_flagged" : ""}.stim`;

  const downloadButton = document.createElement("button");
  downloadButton.type = "button";
  downloadButton.className = "export-button";
  downloadButton.textContent = "Download .stim";
  downloadButton.addEventListener("click", () => {
    downloadTextFile(fileName(), recursiveStimText(construction, state.exportFlags));
  });
  controls.appendChild(downloadButton);

  const copyButton = document.createElement("button");
  copyButton.type = "button";
  copyButton.className = "export-button ghost";
  copyButton.textContent = "Copy";
  copyButton.addEventListener("click", () => {
    const text = recursiveStimText(construction, state.exportFlags);
    const done = () => {
      copyButton.textContent = "Copied!";
      copyButton.classList.add("copied");
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done).catch(() => {
        copyButton.textContent = "Copy failed";
      });
    } else {
      done();
    }
  });
  controls.appendChild(copyButton);

  panel.appendChild(controls);
  refs.visualHost.appendChild(panel);
}

function buildRecursiveZXSvg(construction) {
  const { n, t, measurements, maxLayer } = construction;

  const rowGap = clampNumber(Math.round(620 / Math.max(n, 1)), 12, 30);
  const topPad = 46;
  const bottomPad = 30;
  const height = topPad + (n - 1) * rowGap + bottomPad;

  const abstractWidth = 150;
  const seedX = abstractWidth + 52;
  const wiresStart = seedX + 30;
  const layerGap = clampNumber(Math.round(760 / Math.max(maxLayer, 1)), 26, 60);
  const firstLayerX = wiresStart + 30;
  const rightPad = 40;

  const rowY = (qubit) => topPad + qubit * rowGap;
  const nodeR = clampNumber(rowGap * 0.22, 3, 5.5);

  // Default layout that keeps ZZ-measurements from overlapping: within each
  // scheduling layer, measurements whose vertical spans intersect are placed in
  // separate sub-columns (greedy interval partitioning), so no two connectors
  // ever sit on top of one another before the user drags anything.
  const subGap = Math.max(nodeR * 2 + 5, 11);
  const slotByIdx = new Array(measurements.length).fill(0);
  const layerSlotCount = new Map();
  {
    const byLayer = new Map();
    measurements.forEach((m, idx) => {
      if (!byLayer.has(m.layer)) {
        byLayer.set(m.layer, []);
      }
      byLayer.get(m.layer).push(idx);
    });
    const spanLo = (idx) => Math.min(measurements[idx].qL, measurements[idx].qR);
    const spanHi = (idx) => Math.max(measurements[idx].qL, measurements[idx].qR);
    for (const [layer, idxs] of byLayer.entries()) {
      const sorted = idxs.slice().sort((a, b) => spanLo(a) - spanLo(b));
      // slotEnds[s] = highest qubit row currently occupied in sub-column s.
      const slotEnds = [];
      for (const idx of sorted) {
        let slot = 0;
        while (slot < slotEnds.length && slotEnds[slot] >= spanLo(idx)) {
          slot += 1;
        }
        slotByIdx[idx] = slot;
        slotEnds[slot] = spanHi(idx);
      }
      layerSlotCount.set(layer, slotEnds.length);
    }
  }

  // Lay out each layer's left edge cumulatively, leaving room for its widest
  // fan-out of sub-columns plus a constant gap before the next layer.
  const layerColX = new Map();
  let layerCursor = firstLayerX;
  for (let layer = 1; layer <= Math.max(maxLayer, 1); layer += 1) {
    layerColX.set(layer, layerCursor);
    const slots = layerSlotCount.get(layer) || 1;
    layerCursor += (slots - 1) * subGap + layerGap;
  }
  const width = layerCursor - layerGap + rightPad;

  const defaultMeasX = (idx) =>
    (layerColX.get(measurements[idx].layer) || firstLayerX) + slotByIdx[idx] * subGap;

  const svg = svgNode("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": `ZX diagram of the recursive CAT^${n} preparation at t = ${t}`,
  });

  // Abstract |CAT^n> block: a triangle that is "defined as" the explicit network.
  const cy = height / 2;
  const triTop = clampNumber(cy - 46, topPad, height - 60);
  const triBot = clampNumber(cy + 46, topPad + 60, height - bottomPad);
  const triLeft = 28;
  const triRight = abstractWidth - 36;
  const triangle = svgNode("polygon", {
    points: `${triRight},${triTop} ${triRight},${triBot} ${triLeft},${cy}`,
    fill: "rgba(217, 93, 57, 0.08)",
    stroke: "var(--recursive)",
    "stroke-width": 1.6,
    "stroke-linejoin": "round",
  });
  svg.appendChild(triangle);

  const ket = svgNode("text", {
    x: (triLeft + triRight) / 2 + 6,
    y: cy + 4,
    "font-size": 13,
    "font-weight": 700,
    "text-anchor": "middle",
    fill: "var(--ink)",
  });
  ket.textContent = `|CAT^${n}⟩`;
  svg.appendChild(ket);

  const leafSpiderY = (leaf) => rowY((leaf.lo + leaf.hi - 1) / 2);

  // The triangle is the LHS of an equation |CAT^n> ≜ [network]; it is NOT wired
  // into the network. A few short stubs + vertical dots denote its n abstract
  // output wires, and the "defined-as" symbol separates it from the construction.
  const stubX = triRight + 18;
  for (const dy of [-9, 0, 9]) {
    svg.appendChild(
      svgNode("line", {
        x1: triRight,
        y1: cy + dy,
        x2: stubX,
        y2: cy + dy,
        stroke: "rgba(20, 33, 61, 0.35)",
        "stroke-width": 1.2,
      }),
    );
  }
  const wiresDots = svgNode("text", {
    x: stubX + 9,
    y: cy + 5,
    "font-size": 15,
    "font-weight": 700,
    "text-anchor": "middle",
    fill: "var(--muted)",
  });
  wiresDots.textContent = "⋮";
  svg.appendChild(wiresDots);

  const defEq = svgNode("text", {
    x: (stubX + 18 + seedX) / 2,
    y: cy + 5,
    "font-size": 18,
    "font-weight": 700,
    "text-anchor": "middle",
    fill: "var(--muted)",
  });
  defEq.textContent = "≜";
  svg.appendChild(defEq);

  // explicit qubit wires
  for (let q = 0; q < n; q += 1) {
    const y = rowY(q);
    svg.appendChild(
      svgNode("line", {
        x1: wiresStart,
        y1: y,
        x2: width - rightPad / 2,
        y2: y,
        stroke: "rgba(20, 33, 61, 0.45)",
        "stroke-width": 1.3,
      }),
    );
  }

  // Base-case seed spiders: one Z-spider per leaf, fanning out (a claw)
  // to each qubit of its block — the paper's prepared seed CAT state.
  for (const leaf of construction.leaves) {
    const spiderY = leafSpiderY(leaf);
    for (let q = leaf.lo; q < leaf.hi; q += 1) {
      svg.appendChild(
        svgNode("line", {
          x1: seedX,
          y1: spiderY,
          x2: wiresStart,
          y2: rowY(q),
          stroke: "var(--spider-stroke)",
          "stroke-width": 1.2,
        }),
      );
    }
    svg.appendChild(
      svgNode("circle", {
        cx: seedX,
        cy: spiderY,
        r: nodeR + 1.5,
        fill: "var(--spider-fill)",
        stroke: "var(--spider-stroke)",
        "stroke-width": 1.4,
      }),
    );
  }

  // ZZ-measurements: a coloured vertical connector with a Z-spider at each end,
  // placed at the layer (= scheduling depth) it executes in. Each measurement is
  // draggable horizontally along the wires so users can pull apart measurements
  // that share a time slice. Overrides persist per (n, t) in state.
  const layoutKey = `${n}-${t}`;
  if (!state.recursiveZxLayout[layoutKey]) {
    state.recursiveZxLayout[layoutKey] = {};
  }
  const overrides = state.recursiveZxLayout[layoutKey];
  const dragMinX = wiresStart + 6;
  const dragMaxX = width - rightPad / 2 - 4;
  const measVisuals = [];
  let activeDrag = null;

  function measX(idx) {
    return overrides[idx] != null
      ? clampNumber(overrides[idx], dragMinX, dragMaxX)
      : defaultMeasX(idx);
  }

  function eventXInSvg(event) {
    const ctm = svg.getScreenCTM();
    if (!ctm) {
      return null;
    }
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    return point.matrixTransform(ctm.inverse()).x;
  }

  function applyMeasX(idx) {
    const x = measX(idx);
    const visual = measVisuals[idx];
    visual.line.setAttribute("x1", x);
    visual.line.setAttribute("x2", x);
    visual.hit.setAttribute("x1", x);
    visual.hit.setAttribute("x2", x);
    visual.circles.forEach((circle) => circle.setAttribute("cx", x));
  }

  measurements.forEach((m, idx) => {
    const yTop = rowY(Math.min(m.qL, m.qR));
    const yBot = rowY(Math.max(m.qL, m.qR));
    // Colour by scheduling layer: every measurement in a layer is on disjoint
    // wires, so one colour = one CNOT-depth-1 time slice.
    const color = ZZ_LAYER_COLORS[(m.layer - 1) % ZZ_LAYER_COLORS.length];

    const group = svgNode("g", {
      class: "zx-meas-handle",
      "data-meas": idx,
      role: "button",
      tabindex: "0",
      "aria-label": `Move layer-${m.layer} ZZ-measurement between qubits ${m.qL} and ${m.qR}`,
    });

    // wide transparent hit target makes the thin connector easy to grab
    const hit = svgNode("line", {
      x1: 0,
      y1: yTop,
      x2: 0,
      y2: yBot,
      stroke: "transparent",
      "stroke-width": 16,
      "stroke-linecap": "round",
    });
    group.appendChild(hit);

    const line = svgNode("line", {
      x1: 0,
      y1: yTop,
      x2: 0,
      y2: yBot,
      stroke: color,
      "stroke-width": 2.4,
      "stroke-linecap": "round",
    });
    group.appendChild(line);

    const circles = [m.qL, m.qR].map((q) => {
      const circle = svgNode("circle", {
        cx: 0,
        cy: rowY(q),
        r: nodeR,
        fill: "var(--spider-fill)",
        stroke: color,
        "stroke-width": 1.4,
      });
      group.appendChild(circle);
      return circle;
    });

    group.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      activeDrag = { idx, pointerId: event.pointerId };
      group.classList.add("dragging");
      svg.classList.add("dragging-zx");
      group.setPointerCapture(event.pointerId);
    });

    svg.appendChild(group);
    measVisuals.push({ line, hit, circles, group });
    applyMeasX(idx);
  });

  function moveActiveDrag(event) {
    if (!activeDrag || activeDrag.pointerId !== event.pointerId) {
      return;
    }
    const x = eventXInSvg(event);
    if (x == null) {
      return;
    }
    overrides[activeDrag.idx] = clampNumber(x, dragMinX, dragMaxX);
    applyMeasX(activeDrag.idx);
  }

  function endActiveDrag(event) {
    if (!activeDrag || activeDrag.pointerId !== event.pointerId) {
      return;
    }
    const visual = measVisuals[activeDrag.idx];
    visual.group.classList.remove("dragging");
    if (visual.group.hasPointerCapture(event.pointerId)) {
      visual.group.releasePointerCapture(event.pointerId);
    }
    svg.classList.remove("dragging-zx");
    activeDrag = null;
  }

  svg.addEventListener("pointermove", moveActiveDrag);
  svg.addEventListener("pointerup", endActiveDrag);
  svg.addEventListener("pointercancel", endActiveDrag);

  return svg;
}

// Mix two hex colours; amount 0 -> base, 1 -> target.
function mixHex(base, target, amount) {
  const parse = (hex) => {
    const h = hex.replace("#", "");
    return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
  };
  const [r1, g1, b1] = parse(base);
  const [r2, g2, b2] = parse(target);
  const mix = (a, b) => Math.round(a + (b - a) * amount);
  const hex = (v) => v.toString(16).padStart(2, "0");
  return `#${hex(mix(r1, r2))}${hex(mix(g1, g2))}${hex(mix(b1, b2))}`;
}

// A distinct shade of a layer's base colour for the i-th of `count` ZZ-
// measurements in that layer: lighter shades first, darkening toward near-black,
// so every shade stays legible on the white canvas while reading as one hue.
function zzShade(base, i, count) {
  if (count <= 1) {
    return base;
  }
  const frac = i / (count - 1); // 0 -> lightest, 1 -> darkest
  return frac <= 0.5
    ? mixHex(base, "#ffffff", (0.5 - frac) * 2 * 0.34)
    : mixHex(base, "#0b1020", (frac - 0.5) * 2 * 0.5);
}

// Simplified scheme: one column band per CNOT-depth layer, but inside each band
// every transversal ZZ-measurement is drawn as its own vertical edge connecting
// exactly the two wires it fuses (a dot at each end), so it's clear which wire
// pairs with which. Overlapping measurements are packed into sub-columns, and
// each measurement gets a distinct shade of its layer's colour — the two wires
// of one ZZ-measurement share a tone, different measurements use different tones.
// Built from the same fusion-tree port as the schematic.
function renderRecursiveSimplified(model) {
  clearVisual();
  const construction = buildRecursiveConstruction(state.n, state.t);
  const { n, t, measurements, maxLayer } = construction;

  legendPills(
    Array.from({ length: maxLayer }, (_, i) => ({
      color: ZZ_LAYER_COLORS[i % ZZ_LAYER_COLORS.length],
      label: `Layer ${i + 1}`,
    })),
  );

  // Group the individual ZZ-measurements by layer (each keeps its own wire pair).
  const layerMeas = new Map();
  for (const m of measurements) {
    if (!layerMeas.has(m.layer)) {
      layerMeas.set(m.layer, []);
    }
    layerMeas.get(m.layer).push({ lo: Math.min(m.qL, m.qR), hi: Math.max(m.qL, m.qR) });
  }
  const layers = [...layerMeas.keys()].sort((a, b) => a - b);

  // Per layer: pack measurements into non-overlapping vertical tracks (greedy
  // interval colouring) so stacked edges don't sit on the same sub-column, and
  // assign each a distinct shade of the layer colour.
  const layerPlan = new Map();
  for (const layer of layers) {
    const ms = layerMeas
      .get(layer)
      .slice()
      .sort((a, b) => a.lo - b.lo || a.hi - b.hi);
    const base = ZZ_LAYER_COLORS[(layer - 1) % ZZ_LAYER_COLORS.length];
    const trackEnds = []; // last hi placed on each track
    ms.forEach((mm, i) => {
      let track = trackEnds.findIndex((end) => end < mm.lo);
      if (track === -1) {
        track = trackEnds.length;
        trackEnds.push(mm.hi);
      } else {
        trackEnds[track] = mm.hi;
      }
      mm.track = track;
      mm.color = zzShade(base, i, ms.length);
    });
    layerPlan.set(layer, { ms, tracks: Math.max(1, trackEnds.length), base });
  }

  const wireGap = 26;
  const subColGap = 22; // px between sub-columns within a layer band
  const layerPad = 40; // gap between adjacent layer bands
  const leftPad = 60;
  const topPad = 40;
  const rightPad = 40;
  const bottomPad = 34;

  // Lay the layer bands out left-to-right, each as wide as its track count needs.
  const bandX = new Map();
  let cursor = leftPad;
  for (const layer of layers) {
    const tracks = layerPlan.get(layer).tracks;
    const bandWidth = tracks * subColGap;
    bandX.set(layer, { start: cursor, width: bandWidth });
    cursor += bandWidth + layerPad;
  }
  const width = Math.max(leftPad + 80, cursor - layerPad + rightPad);
  const height = topPad + (n - 1) * wireGap + bottomPad;
  const yOf = (q) => topPad + q * wireGap;
  const subX = (layer, track) => bandX.get(layer).start + (track + 0.5) * subColGap;

  const svg = svgNode("svg", {
    viewBox: `0 0 ${width} ${height}`,
    width,
    height,
    "aria-label": `Simplified recursive CAT^${n} scheme: each ZZ-measurement as a shaded vertical edge between the two wires it fuses`,
  });

  // Horizontal qubit wires with labels on the left.
  for (let q = 0; q < n; q += 1) {
    svg.appendChild(svgNode("line", {
      x1: leftPad - 18, y1: yOf(q), x2: width - rightPad + 12, y2: yOf(q),
      stroke: "rgba(20, 33, 61, 0.22)", "stroke-width": 1,
    }));
    const label = svgNode("text", {
      x: leftPad - 24, y: yOf(q) + 4,
      "text-anchor": "end", "font-size": 11, fill: "var(--muted)",
    });
    label.textContent = `q${q}`;
    svg.appendChild(label);
  }

  // One shaded vertical edge per ZZ-measurement, connecting its two wires.
  for (const layer of layers) {
    const { ms, base, tracks } = layerPlan.get(layer);
    for (const mm of ms) {
      const x = subX(layer, mm.track);
      svg.appendChild(svgNode("line", {
        x1: x, y1: yOf(mm.lo), x2: x, y2: yOf(mm.hi),
        stroke: mm.color, "stroke-width": 3.4, "stroke-linecap": "round",
      }));
      for (const q of [mm.lo, mm.hi]) {
        svg.appendChild(svgNode("circle", {
          cx: x, cy: yOf(q), r: 4.2, fill: mm.color,
        }));
      }
    }
    const band = bandX.get(layer);
    const head = svgNode("text", {
      x: band.start + (tracks * subColGap) / 2, y: topPad - 16,
      "text-anchor": "middle", "font-size": 11, "font-weight": 600, fill: base,
    });
    head.textContent = `Layer ${layer}`;
    svg.appendChild(head);
  }

  renderZoomableSvg(svg, "recursive-simplified", {
    minScale: 0.3,
    maxScale: 5,
    naturalSize: { width, height },
    hint: "Each shaded vertical edge is one transversal ZZ-measurement linking the two wires it fuses; shades of a layer's colour tell its individual measurements apart. If it runs off-screen, drag the scrollbars to pan, or zoom out.",
  });

  refs.visualCaption.textContent =
    `Simplified scheme for the recursive CAT^${n} preparation at t = ${t}. ` +
    `The ${measurements.length} transversal ZZ-measurements are grouped into ${maxLayer} CNOT-depth ` +
    `layer${maxLayer === 1 ? "" : "s"}, one column band each. Within a band every ZZ-measurement is a ` +
    `separate vertical edge joining the two wires it fuses, drawn in its own shade of the layer's colour so ` +
    `wires sharing a measurement share a tone.`;
}

// In-browser port of pyzx's recursive_unfuse_FE (zxcalc/pyzx unfuse_FE_rules.py,
// arXiv:2506.17181), so the fault-equivalent ZX diagram is generated live for any
// n (the slider goes to 500 — too many/too-large graphs to precompute). The unfuse
// is driven with w = t + 1 gadgets ((t + 1)-FE decomposition); pyzx's vertex/edge
// counts and degree sequence match when called with the same gadget count w.
// Vertex types: 0 boundary, 1 Z, 2 X. Edge types: 1 simple, 2 hadamard.
const ZX_BOUNDARY = 0;
const ZX_Z = 1;
const ZX_SIMPLE = 1;

class UnfuseGraph {
  constructor() {
    this.next = 0;
    this.ty = new Map();
    this.q = new Map();
    this.r = new Map();
    this.adj = new Map(); // id -> Map(neighbor -> edgeType); insertion order preserved
  }
  addVertex(type, qubit = -1, row = -1) {
    const v = this.next++;
    this.ty.set(v, type);
    this.q.set(v, qubit);
    this.r.set(v, row);
    this.adj.set(v, new Map());
    return v;
  }
  addEdge(a, b, et = ZX_SIMPLE) {
    this.adj.get(a).set(b, et);
    this.adj.get(b).set(a, et);
  }
  removeVertex(v) {
    for (const nb of this.adj.get(v).keys()) {
      this.adj.get(nb).delete(v);
    }
    this.adj.delete(v);
    this.ty.delete(v);
    this.q.delete(v);
    this.r.delete(v);
  }
  neighbors(v) {
    return [...this.adj.get(v).keys()];
  }
  degree(v) {
    return this.adj.get(v).size;
  }
  vertices() {
    return [...this.ty.keys()];
  }
}

// All permutations of arr (arr length <= 5 here: only used for degree-4/5 unfuse).
function permutationsOf(arr) {
  if (arr.length <= 1) {
    return [arr.slice()];
  }
  const out = [];
  for (let i = 0; i < arr.length; i += 1) {
    const rest = arr.slice(0, i).concat(arr.slice(i + 1));
    for (const p of permutationsOf(rest)) {
      out.push([arr[i], ...p]);
    }
  }
  return out;
}

// Brute-force min-cost assignment (pyzx's _linear_sum_assignment_itertools).
function minCostAssignment(cost) {
  const rows = cost.length;
  const cols = rows ? cost[0].length : 0;
  let best = Infinity;
  let bestPerm = [];
  for (const perm of permutationsOf([...Array(cols).keys()].slice(0, cols))) {
    if (perm.length < rows) continue;
    let s = 0;
    for (let i = 0; i < rows; i += 1) {
      s += cost[i][perm[i]];
    }
    if (s < best) {
      best = s;
      bestPerm = perm.slice(0, rows);
    }
  }
  return bestPerm;
}

function bestPairing(g, neighbors, newVerts) {
  const m = neighbors.length;
  if (m === 0) return [];
  const cost = Array.from({ length: m }, () => new Array(m).fill(0));
  for (let i = 0; i < m; i += 1) {
    const nq = g.q.get(neighbors[i]);
    const nr = g.r.get(neighbors[i]);
    for (let j = 0; j < m; j += 1) {
      cost[i][j] = Math.hypot(g.q.get(newVerts[j]) - nq, g.r.get(newVerts[j]) - nr);
    }
  }
  return minCostAssignment(cost);
}

function squareCoords(q, r) {
  const d = 0.5;
  return [[q - d, r - d], [q + d, r - d], [q + d, r + d], [q - d, r + d]];
}

function nCycleCoords(N, q, r) {
  const radius = (0.75 * N) / 5;
  const coords = [];
  for (let i = 0; i < N; i += 1) {
    const angle = (2 * Math.PI * i) / N + Math.PI;
    coords.push([q + radius * Math.cos(angle), r - radius * Math.sin(angle)]);
  }
  return coords;
}

// Unfuse a spider into a polygon of new spiders (degree-4 square, degree-5 pentagon).
function unfusePolygon(g, v, coordsFn) {
  const vType = g.ty.get(v);
  const neighs = g.neighbors(v);
  const originalEdge = new Map(neighs.map((n) => [n, g.adj.get(v).get(n)]));
  const coords = coordsFn(g.q.get(v), g.r.get(v));
  const newVs = coords.map(([qc, rc]) => g.addVertex(vType, qc, rc));
  for (let i = 0; i < newVs.length; i += 1) {
    g.addEdge(newVs[i], newVs[(i + 1) % newVs.length]);
  }
  const assignment = bestPairing(g, neighs, newVs);
  neighs.forEach((nb, i) => g.addEdge(nb, newVs[assignment[i]], originalEdge.get(nb)));
  g.removeVertex(v);
}

function splitNeighbors(g, neighbors) {
  const sorted = neighbors.slice().sort((a, b) => g.q.get(a) - g.q.get(b));
  const mid = Math.floor(sorted.length / 2);
  return [sorted.slice(0, mid), sorted.slice(mid)];
}

// Core 2n-degree unfusion: two inner spiders plus w parity-check gadgets.
function unfuse2nCore(g, v, w) {
  const vType = g.ty.get(v);
  const [group1, group2] = splitNeighbors(g, g.neighbors(v));
  const degreeN = group1.length;
  const all = group1.concat(group2);
  const startFrom = Math.min(...all.map((n) => g.r.get(n))) - 1;
  const posQ1 = group1.reduce((s, n) => s + g.q.get(n), 0) / group1.length;
  const posQ2 = group2.reduce((s, n) => s + g.q.get(n), 0) / group2.length;

  const inner1 = g.addVertex(vType, posQ1, startFrom - degreeN - 1);
  const inner2 = g.addVertex(vType, posQ2, startFrom - degreeN - 1);

  const pairs = Math.min(group1.length, group2.length);
  for (let i = 0; i < pairs; i += 1) {
    const n1 = group1[i];
    const n2 = group2[i];
    if (w == null || w >= i + 1) {
      const v1 = g.addVertex(vType, g.q.get(n1), startFrom - i);
      const v2 = g.addVertex(vType, g.q.get(n2), startFrom - i);
      g.addEdge(v1, n1);
      g.addEdge(v2, n2);
      g.addEdge(v1, v2);
      g.addEdge(inner1, v1);
      g.addEdge(inner2, v2);
    } else {
      g.addEdge(inner1, n1);
      g.addEdge(inner2, n2);
    }
  }
  if (group2.length > group1.length) {
    g.addEdge(inner2, group2[group2.length - 1]);
  }
  g.removeVertex(v);
  return [inner1, inner2];
}

function recursiveUnfuse(g, v, w) {
  const degree = g.degree(v);
  if (degree <= 3) return;
  if (degree === 4) {
    unfusePolygon(g, v, squareCoords);
    return;
  }
  if (degree === 5) {
    unfusePolygon(g, v, (q, r) => nCycleCoords(5, q, r));
    return;
  }
  const [inner1, inner2] = unfuse2nCore(g, v, w);
  recursiveUnfuse(g, inner1, w);
  recursiveUnfuse(g, inner2, w);
}

// Build the unfused fault-equivalent ZX diagram for CAT^n at fault weight t and
// return it as a {v, e, o, nSpiders} structure for buildUnfuseZXSvg.
function recursiveZxStructure(nRaw, tRaw) {
  const n = Math.max(4, Math.round(nRaw));
  const t = Math.max(1, Math.round(tRaw));
  const g = new UnfuseGraph();
  const centre = g.addVertex(ZX_Z, (n - 1) / 2, 0);
  const outputs = [];
  for (let i = 0; i < n; i += 1) {
    const b = g.addVertex(ZX_BOUNDARY, i, 2);
    g.addEdge(centre, b);
    outputs.push(b);
  }
  // Tolerating t faults needs a (t + 1)-FE decomposition: each unfuse emits
  // w = t + 1 ZZ-measurement gadgets (e.g. five at t = 4), matching the
  // transversal fusion used in buildRecursiveConstruction.
  recursiveUnfuse(g, centre, t + 1);

  const ids = g.vertices();
  const index = new Map(ids.map((id, i) => [id, i]));
  const v = ids.map((id) => [g.ty.get(id), g.q.get(id), g.r.get(id)]);
  const seen = new Set();
  const e = [];
  for (const a of ids) {
    for (const [b, et] of g.adj.get(a)) {
      const key = index.get(a) < index.get(b)
        ? `${index.get(a)},${index.get(b)}`
        : `${index.get(b)},${index.get(a)}`;
      if (!seen.has(key)) {
        seen.add(key);
        e.push([index.get(a), index.get(b), et]);
      }
    }
  }
  return {
    v,
    e,
    o: outputs.map((id) => index.get(id)),
    nSpiders: ids.filter((id) => g.ty.get(id) !== ZX_BOUNDARY).length,
  };
}

// Draw a pyzx graph structure {v:[[type,qubit,row]], e:[[a,b,edgeType]], o:[...]}.
// pyzx uses `qubit` as the vertical axis and `row` as the horizontal axis, so we
// map x <- row and y <- qubit and keep pyzx's own coordinates (hence layout).
function buildUnfuseZXSvg(struct) {
  const XS = 78; // px per row unit (horizontal)
  const YS = 30; // px per qubit unit (vertical)
  const pad = 36;
  const rows = struct.v.map((vt) => vt[2]);
  const qubits = struct.v.map((vt) => vt[1]);
  const rMin = Math.min(...rows);
  const qMin = Math.min(...qubits);
  const width = (Math.max(...rows) - rMin) * XS + pad * 2;
  const height = (Math.max(...qubits) - qMin) * YS + pad * 2;
  const px = (vt) => pad + (vt[2] - rMin) * XS;
  const py = (vt) => pad + (vt[1] - qMin) * YS;

  const svg = svgNode("svg", {
    viewBox: `0 0 ${width} ${height}`,
    width,
    height,
    "aria-label": `Fault-equivalent ZX diagram from pyzx recursive_unfuse_FE`,
  });

  // Edges first so spiders draw on top.
  for (const [a, b, et] of struct.e) {
    const va = struct.v[a];
    const vb = struct.v[b];
    const line = svgNode("line", {
      x1: px(va), y1: py(va), x2: px(vb), y2: py(vb),
      stroke: "#000000",
      "stroke-width": 1.6,
      "stroke-dasharray": et === 2 ? "5 4" : "none", // Hadamard edge dashed
    });
    svg.appendChild(line);
    if (et === 2) {
      // Hadamard box at the edge midpoint.
      const mx = (px(va) + px(vb)) / 2;
      const my = (py(va) + py(vb)) / 2;
      svg.appendChild(svgNode("rect", {
        x: mx - 5, y: my - 5, width: 10, height: 10,
        fill: "#ffd166", stroke: "#14213d", "stroke-width": 1,
      }));
    }
  }

  const outputs = new Set(struct.o);
  struct.v.forEach((vt, i) => {
    const type = vt[0];
    if (type === 0) {
      // Boundary (output wire end): small open marker.
      svg.appendChild(svgNode("circle", {
        cx: px(vt), cy: py(vt), r: 3.5,
        fill: outputs.has(i) ? "#14213d" : "#ffffff",
        stroke: "#14213d", "stroke-width": 1.2,
      }));
      return;
    }
    // Z-spider green, X-spider red (only Z appears in this construction).
    svg.appendChild(svgNode("circle", {
      cx: px(vt), cy: py(vt), r: 7,
      fill: type === 2 ? "#f3b6b6" : "var(--spider-fill)",
      stroke: type === 2 ? "#b3261e" : "var(--recursive)",
      "stroke-width": 1.6,
    }));
  });

  return svg;
}

function renderRecursiveZX(model) {
  clearVisual();

  const struct = recursiveZxStructure(state.n, state.t);

  legendPills([
    { color: "var(--spider-fill)", label: "Z-spider (degree <= 3)" },
    { color: "#14213d", label: "output boundary" },
  ]);

  const svg = buildUnfuseZXSvg(struct);
  // Render at natural pixel size so a large diagram overflows the viewport and
  // both scrollbars appear (drag them to pan) instead of being squeezed to fit.
  const [, , vbWidth, vbHeight] = (svg.getAttribute("viewBox") || "0 0 900 480")
    .split(/\s+/)
    .map(Number);
  renderZoomableSvg(svg, "recursive-zx", {
    minScale: 0.3,
    maxScale: 5,
    naturalSize: { width: vbWidth, height: vbHeight },
    hint: "Fault-equivalent ZX diagram generated by pyzx recursive_unfuse_FE. If it runs off-screen, drag the scrollbars to pan, or zoom out for the whole figure.",
  });

  const n = Math.round(state.n);
  const t = Math.round(state.t);
  refs.visualCaption.textContent =
    `Fault-equivalent ZX diagram for CAT^${n} at t = ${t}, generated by an in-browser port of ` +
    `pyzx's recursive_unfuse_FE (zxcalc/pyzx unfuse_FE_rules.py, arXiv:2506.17181): a single ` +
    `degree-${n} Z-spider is recursively unfused with w = t + 1 = ${t + 1} ZZ-measurement ` +
    `gadgets per fusion into ` +
    `${struct.nSpiders} degree-<=3 Z-spiders. The n output legs are on the right. ` +
    `Generated live, so it scales with the n slider.`;
}

function renderRecursiveSchematic(model) {
  clearVisual();
  legendPills([
    { color: "var(--recursive)", label: "recursive fusion step" },
    { color: "rgba(20, 33, 61, 0.16)", label: "smaller CAT block" },
  ]);

  const construction = buildRecursiveConstruction(state.n, state.t);
  const { root, baseSize } = construction;

  // Walk the ACTUAL fusion tree, tagging each node with a leaf-slot index (its x)
  // and its height (the row it sits in). Leaves are height 0 at the top; the root
  // is deepest and becomes the final output. Leaf count is ceil(n / baseSize), so
  // unbalanced trees (counts that aren't powers of two) show the true block count
  // per round — e.g. n = 20, base 4 gives 5 seeds, not a padded 8.
  const allNodes = [];
  let leafCursor = 0;
  function place(node) {
    if (node.leaf) {
      node._slot = leafCursor;
      leafCursor += 1;
      node._h = 0;
    } else {
      place(node.left);
      place(node.right);
      node._slot = (node.left._slot + node.right._slot) / 2;
      node._h = node.height;
    }
    allNodes.push(node);
  }
  place(root);

  const numLeaves = leafCursor;
  const maxHeight = root._h;
  const internalNodes = allNodes.filter((node) => !node.leaf);

  const laneLeft = 192;
  const laneRight = 64;
  const colWidth = 132;
  const width = Math.max(980, laneLeft + laneRight + numLeaves * colWidth);
  const usableWidth = width - laneLeft - laneRight;
  const topPad = 92;
  const laneGap = 130;
  const boxHeight = 58;
  const height = topPad + maxHeight * laneGap + 96;

  const xOf = (slot) => laneLeft + (usableWidth * (slot + 0.5)) / numLeaves;
  const yOf = (h) => topPad + h * laneGap;

  const svg = svgNode("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": "Recursive CAT state fusion schematic",
  });

  function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }

  function appendBoxLabel(x, y, lines, isFinal) {
    const text = svgNode("text", {
      x,
      y: y - (lines.length - 1) * 7,
      "font-size": isFinal ? 13 : 11.5,
      "font-weight": isFinal ? 700 : 600,
      "text-anchor": "middle",
      fill: "var(--ink)",
    });
    lines.forEach((line, index) => {
      const span = svgNode("tspan", { x, dy: index === 0 ? 0 : 16 });
      span.textContent = line;
      text.appendChild(span);
    });
    svg.appendChild(text);
  }

  // One lane background + title per height row.
  const rowCounts = new Array(maxHeight + 1).fill(0);
  allNodes.forEach((node) => { rowCounts[node._h] += 1; });
  for (let h = 0; h <= maxHeight; h += 1) {
    const y = yOf(h);
    const isOutput = h === maxHeight;
    svg.appendChild(svgNode("rect", {
      x: 18,
      y: y - 48,
      width: width - 36,
      height: 88,
      rx: 26,
      fill: isOutput ? "rgba(217, 93, 57, 0.08)" : "rgba(255, 255, 255, 0.32)",
      stroke: isOutput ? "rgba(217, 93, 57, 0.18)" : "rgba(20, 33, 61, 0.06)",
      "stroke-width": 1.2,
    }));

    const title = svgNode("text", { x: 42, y: y - 8, "font-size": 12, "font-weight": 700, fill: "var(--ink)" });
    title.textContent = h === 0 ? "Seed blocks" : isOutput ? "Output" : `Fusion round ${h}`;
    svg.appendChild(title);

    const subtitle = svgNode("text", { x: 42, y: y + 12, "font-size": 11, fill: "var(--muted)" });
    subtitle.textContent =
      h === 0
        ? `${numLeaves} base CAT^${baseSize} block${numLeaves === 1 ? "" : "s"}`
        : isOutput
          ? "final fault-tolerant CAT state"
          : `${rowCounts[h]} merged CAT block${rowCounts[h] === 1 ? "" : "s"}`;
    svg.appendChild(subtitle);
  }

  // Fusion connectors: each internal node fuses its two children with (t+1) ZZ
  // checks. Children may sit several rows up (a block that waits its turn), so a
  // connector can span more than one lane.
  internalNodes.forEach((node) => {
    const parentX = xOf(node._slot);
    const parentTop = yOf(node._h) - boxHeight / 2;
    const junctionY = parentTop - 26;
    [node.left, node.right].forEach((child) => {
      const childX = xOf(child._slot);
      const childBottom = yOf(child._h) + boxHeight / 2;
      svg.appendChild(svgNode("path", {
        d: `M ${childX} ${childBottom} C ${childX} ${junctionY - 18}, ${parentX} ${junctionY - 18}, ${parentX} ${junctionY}`,
        fill: "none",
        stroke: "var(--recursive)",
        "stroke-width": 3.5,
        "stroke-linecap": "round",
      }));
    });
    svg.appendChild(svgNode("line", {
      x1: parentX, y1: junctionY, x2: parentX, y2: parentTop,
      stroke: "var(--recursive)", "stroke-width": 3.5, "stroke-linecap": "round",
    }));
    svg.appendChild(svgNode("circle", {
      cx: parentX, cy: junctionY, r: 4.5, fill: "var(--recursive)", stroke: "#fff", "stroke-width": 1.4,
    }));
    const labelWidth = 94;
    svg.appendChild(svgNode("rect", {
      x: parentX - labelWidth / 2, y: junctionY + 5, width: labelWidth, height: 22, rx: 11,
      fill: "rgba(255, 250, 241, 0.96)", stroke: "rgba(217, 93, 57, 0.2)", "stroke-width": 1,
    }));
    const label = svgNode("text", {
      x: parentX, y: junctionY + 20, "font-size": 11.5, "font-weight": 700, "text-anchor": "middle", fill: "var(--recursive)",
    });
    label.textContent = `${state.t + 1} ZZ checks`;
    svg.appendChild(label);
  });

  // Blocks: seeds, merged sub-blocks, and the final output (the root).
  allNodes.forEach((node) => {
    const x = xOf(node._slot);
    const y = yOf(node._h);
    const isOutput = node === root;
    const widthBox = isOutput ? 340 : clamp(usableWidth / numLeaves - 22, 96, 150);
    svg.appendChild(svgNode("rect", {
      x: x - widthBox / 2,
      y: y - boxHeight / 2,
      width: widthBox,
      height: boxHeight,
      rx: 22,
      fill: isOutput ? "rgba(217, 93, 57, 0.16)" : "rgba(20, 33, 61, 0.06)",
      stroke: isOutput ? "var(--recursive)" : "rgba(20, 33, 61, 0.18)",
      "stroke-width": isOutput ? 2.6 : 1.5,
    }));
    appendBoxLabel(x, y + 2, isOutput ? ["final", `CAT_${state.n}`] : ["CAT", "sub-block"], isOutput);
  });

  renderZoomableSvg(svg, "recursive-schematic", {
    minScale: 0.3,
    maxScale: 3,
    naturalSize: { width, height },
    hint: "Recursive fusion tree. If it runs off-screen, drag the scrollbars to pan, or zoom out for the whole figure.",
  });

  refs.visualCaption.textContent =
    `Recursive paper construction for CAT^${state.n} at t = ${state.t}: ${numLeaves} base CAT^${baseSize} ` +
    `seed block${numLeaves === 1 ? "" : "s"} fuse up a binary tree over ${maxHeight} round${maxHeight === 1 ? "" : "s"}, ` +
    `each fusion using ${state.t + 1} transversal ZZ checks. Set the seed size with the Base size slider below.`;
}

// Pull the headline metric (CNOT count or depth) for a method at a given (n, t).
// The two paper constructions expose closed-form estimators over any n, so they
// draw as continuous curves; the bundled-circuit methods only return a value at
// their saved (n, t) points, leaving natural gaps in the scatter.
function scalabilityMetricAt(methodId, n, t, field) {
  if (methodId === "recursive") {
    return recursiveEstimate(n, t)[field];
  }
  if (methodId === "shallow") {
    const estimate = shallowEstimate(n, t);
    return estimate.available ? estimate[field] : null;
  }
  const actual = getActualMetric(methodId, n, t);
  if (!actual) {
    return null;
  }
  if (field === "depth") {
    return actual.depth;
  }
  if (field === "ancillas") {
    return actual.numFlags;
  }
  return actual.numCx;
}

// "Compare Different Methods" section: overlay how CNOT count, CNOT depth, and
// ancilla count grow with the target size n for every construction at the current
// fault weight t. The recursive construction's depth is 2*log2(t) + 2 — constant
// in n — so its curve reads as a flat line while the baselines climb, the
// scalability story Theorem 3.1 makes, while its CNOT count stays near-linear.
function renderMethodComparison() {
  refs.comparisonLegend.innerHTML = "";
  refs.comparisonHost.innerHTML = "";
  refs.comparisonCaption.textContent = "";

  const SERIES = data.methods.order.map((id) => ({
    id,
    label: data.methods[id].label,
    color: METHOD_ACCENTS[id],
  }));

  legendPills(
    SERIES.map((series) => ({ color: series.color, label: series.label })),
    refs.comparisonLegend,
  );

  const ns = data.controls.comparisonNs;
  const nMin = ns[0];
  const nMax = ns[ns.length - 1];
  const t = state.t;

  const width = 980;
  const panelHeight = 300;
  const panelGap = 56;
  const height = panelHeight * 3 + panelGap * 2 + 40;
  const plotLeft = 84;
  const plotRight = width - 220; // leave room for the inline series labels
  const plotW = plotRight - plotLeft;

  const svg = svgNode("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": `Scalability of every construction in n at t = ${t}`,
  });

  const xOfN = (n) => plotLeft + ((n - nMin) / (nMax - nMin)) * plotW;

  function niceCeil(value) {
    if (value <= 0) {
      return 1;
    }
    const magnitude = 10 ** Math.floor(Math.log10(value));
    for (const step of [1, 2, 2.5, 5, 10]) {
      const candidate = step * magnitude;
      if (candidate >= value) {
        return candidate;
      }
    }
    return 10 * magnitude;
  }

  function drawPanel(top, field, title, subtitle) {
    const bottom = top + panelHeight;
    const plotTop = top + 30;
    const plotBottom = bottom - 44;
    const plotH = plotBottom - plotTop;

    // Gather each method's points and the panel's shared y-range.
    const seriesPoints = SERIES.map((series) => {
      const points = [];
      for (const n of ns) {
        const value = scalabilityMetricAt(series.id, n, t, field);
        if (value != null && Number.isFinite(value)) {
          points.push({ n, value });
        }
      }
      return { ...series, points };
    });

    const maxValue = Math.max(
      1,
      ...seriesPoints.flatMap((series) => series.points.map((point) => point.value)),
    );
    const yMax = niceCeil(maxValue);
    const yOf = (value) => plotBottom - (value / yMax) * plotH;

    // Panel title + subtitle.
    const titleNode = svgNode("text", {
      x: plotLeft,
      y: top + 12,
      "font-size": 15,
      "font-weight": 700,
      fill: "var(--ink)",
    });
    titleNode.textContent = title;
    svg.appendChild(titleNode);

    const subtitleNode = svgNode("text", {
      x: plotLeft,
      y: top + 28,
      "font-size": 11.5,
      fill: "var(--muted)",
    });
    subtitleNode.textContent = subtitle;
    svg.appendChild(subtitleNode);

    // Horizontal gridlines + y tick labels.
    const yTicks = 4;
    for (let i = 0; i <= yTicks; i += 1) {
      const value = (yMax * i) / yTicks;
      const y = yOf(value);
      svg.appendChild(
        svgNode("line", {
          x1: plotLeft,
          y1: y,
          x2: plotRight,
          y2: y,
          stroke: "rgba(20, 33, 61, 0.1)",
          "stroke-width": i === 0 ? 1.4 : 1,
        }),
      );
      const tick = svgNode("text", {
        x: plotLeft - 10,
        y: y + 4,
        "font-size": 10.5,
        "text-anchor": "end",
        fill: "var(--muted)",
      });
      tick.textContent = String(Math.round(value));
      svg.appendChild(tick);
    }

    // X axis ticks + labels.
    const xTickNs = ns.filter((n, index) => index % 6 === 0 || n === nMax);
    for (const n of xTickNs) {
      const x = xOfN(n);
      svg.appendChild(
        svgNode("line", {
          x1: x,
          y1: plotBottom,
          x2: x,
          y2: plotBottom + 5,
          stroke: "rgba(20, 33, 61, 0.4)",
          "stroke-width": 1,
        }),
      );
      const label = svgNode("text", {
        x,
        y: plotBottom + 18,
        "font-size": 10.5,
        "text-anchor": "middle",
        fill: "var(--muted)",
      });
      label.textContent = String(n);
      svg.appendChild(label);
    }

    const xAxisLabel = svgNode("text", {
      x: (plotLeft + plotRight) / 2,
      y: plotBottom + 36,
      "font-size": 11.5,
      "font-weight": 600,
      "text-anchor": "middle",
      fill: "var(--ink)",
    });
    xAxisLabel.textContent = "target size n";
    svg.appendChild(xAxisLabel);

    // One polyline + dotted markers per method, with an inline right-hand label.
    for (const series of seriesPoints) {
      if (!series.points.length) {
        continue;
      }
      const emphasised = series.id === "recursive";
      const pointsAttr = series.points
        .map((point) => `${xOfN(point.n)},${yOf(point.value)}`)
        .join(" ");
      svg.appendChild(
        svgNode("polyline", {
          points: pointsAttr,
          fill: "none",
          stroke: series.color,
          "stroke-width": emphasised ? 3.6 : 2,
          "stroke-linecap": "round",
          "stroke-linejoin": "round",
          "stroke-opacity": emphasised ? 1 : 0.85,
          // Sparse bundled-circuit families read better dashed, so the eye does
          // not mistake the interpolation for measured intermediate points.
          "stroke-dasharray": series.points.length < ns.length ? "6 5" : "none",
        }),
      );
      for (const point of series.points) {
        svg.appendChild(
          svgNode("circle", {
            cx: xOfN(point.n),
            cy: yOf(point.value),
            r: emphasised ? 3 : 2.2,
            fill: series.color,
          }),
        );
      }

      const last = series.points[series.points.length - 1];
      const labelNode = svgNode("text", {
        x: plotRight + 12,
        y: yOf(last.value) + 4,
        "font-size": emphasised ? 12 : 11,
        "font-weight": emphasised ? 700 : 600,
        fill: series.color,
      });
      labelNode.textContent = series.label;
      svg.appendChild(labelNode);
    }
  }

  drawPanel(
    20,
    "numCx",
    "CNOT count vs n",
    "The recursive count grows only near-linearly: n·(1 + log2(t+1)) − 2(t+1).",
  );
  drawPanel(
    20 + panelHeight + panelGap,
    "depth",
    "CNOT depth vs n",
    `The recursive depth 2·log2(t)+2 stays flat in n at t = ${t}; the baselines climb with n.`,
  );
  drawPanel(
    20 + (panelHeight + panelGap) * 2,
    "ancillas",
    "Ancilla count vs n",
    "Ancilla usage per construction: recursive needs n/2, while the shallow trade buys depth 3 with more ancillae.",
  );

  renderZoomableSvg(svg, `method-comparison-${t}`, {
    host: refs.comparisonHost,
    maxScale: 3,
    hint: "Solid curves are closed-form paper estimators; dashed curves are bundled-circuit families plotted at their saved (n, t) points.",
  });

  refs.comparisonCaption.textContent =
    `Method comparison at t = ${t}, across CNOT count, CNOT depth, and ancilla count. The recursive construction (Theorem 3.1) holds CNOT depth constant in n — its defining scalability advantage — while keeping the CNOT count near-linear. Bundled baselines (SpiderCat, flag-at-origin, MQT) are charted only where the repo ships an exact circuit, so their curves are sparser.`;
}

function renderShallowSchematic(model) {
  clearVisual();
  legendPills([
    { color: "var(--shallow)", label: "depth layer 1" },
    { color: "#60a5fa", label: "depth layer 2" },
    { color: "#93c5fd", label: "depth layer 3" },
  ]);

  const width = 980;
  const height = 560;
  const svg = svgNode("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": "Constant-depth shallow construction schematic",
  });

  const nodes = 12;
  const leftPad = 160;
  const rightPad = 70;
  const topPad = 98;
  const rowGap = 142;
  const cardHeight = 112;
  const guideTop = topPad - 34;
  const guideBottom = topPad + rowGap * 2 + 82;
  const xs = Array.from(
    { length: nodes },
    (_, index) => leftPad + (index * (width - leftPad - rightPad)) / (nodes - 1),
  );
  const ys = Array.from({ length: 3 }, (_, index) => topPad + index * rowGap + 42);
  const layerColors = ["var(--shallow)", "#60a5fa", "#93c5fd"];
  const layerTitles = [
    ["Layer 1", "adjacent disjoint pairs"],
    ["Layer 2", "offset disjoint pairs"],
    ["Layer 3", "longer-span disjoint pairs"],
  ];
  const matchings = [
    [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
    [[0, 2], [1, 3], [4, 6], [5, 7], [8, 10], [9, 11]],
    [[0, 4], [1, 5], [2, 6], [3, 7], [8, 11], [9, 10]],
  ];

  const intro = svgNode("text", {
    x: 28,
    y: 34,
    "font-size": 13,
    "font-weight": 700,
    fill: "var(--ink)",
  });
  intro.textContent = "Same qubit register reused in all 3 CNOT layers";
  svg.appendChild(intro);

  const introSub = svgNode("text", {
    x: 28,
    y: 54,
    "font-size": 11.5,
    fill: "var(--muted)",
  });
  introSub.textContent = "Each row is one depth layer; every qubit touches exactly one edge per row.";
  svg.appendChild(introSub);

  xs.forEach((x, index) => {
    svg.appendChild(
      svgNode("line", {
        x1: x,
        y1: guideTop,
        x2: x,
        y2: guideBottom,
        stroke: "rgba(20, 33, 61, 0.11)",
        "stroke-width": 1.2,
        "stroke-dasharray": "4 8",
      }),
    );

    const label = svgNode("text", {
      x,
      y: guideBottom + 24,
      "font-size": 11.5,
      "font-weight": 600,
      "text-anchor": "middle",
      fill: "var(--muted)",
    });
    label.textContent = `q${index}`;
    svg.appendChild(label);
  });

  ys.forEach((y, layerIndex) => {
    svg.appendChild(
      svgNode("rect", {
        x: 18,
        y: y - 56,
        width: width - 36,
        height: cardHeight,
        rx: 24,
        fill: "rgba(255, 255, 255, 0.36)",
        stroke: "rgba(20, 33, 61, 0.07)",
        "stroke-width": 1.2,
      }),
    );

    const title = svgNode("text", {
      x: 34,
      y: y - 8,
      "font-size": 13,
      "font-weight": 700,
      fill: "var(--ink)",
    });
    title.textContent = layerTitles[layerIndex][0];
    svg.appendChild(title);

    const subtitle = svgNode("text", {
      x: 34,
      y: y + 12,
      "font-size": 11.5,
      fill: "var(--muted)",
    });
    subtitle.textContent = layerTitles[layerIndex][1];
    svg.appendChild(subtitle);

    svg.appendChild(
      svgNode("line", {
        x1: xs[0],
        y1: y,
        x2: xs[xs.length - 1],
        y2: y,
        stroke: "rgba(20, 33, 61, 0.12)",
        "stroke-width": 2,
      }),
    );

    matchings[layerIndex].forEach(([left, right], pairIndex) => {
      const span = right - left;
      const x1 = xs[left];
      const x2 = xs[right];
      const controlX = (x1 + x2) / 2;
      const lift = 18 + span * 14 + (pairIndex % 2) * 4;
      svg.appendChild(
        svgNode("path", {
          d: `M ${x1} ${y} Q ${controlX} ${y - lift} ${x2} ${y}`,
          fill: "none",
          stroke: layerColors[layerIndex],
          "stroke-width": 5.5,
          "stroke-linecap": "round",
        }),
      );
    });

    xs.forEach((x, index) => {
      svg.appendChild(
        svgNode("circle", {
          cx: x,
          cy: y,
          r: 8,
          // Z spider convention: light-green fill (row color kept on the outline)
          fill: "var(--spider-fill)",
          stroke: layerColors[layerIndex],
          "stroke-width": 2.5,
        }),
      );
    });

    const badge = svgNode("rect", {
      x: width - 144,
      y: y - 17,
      width: 102,
      height: 24,
      rx: 12,
      fill: "rgba(255, 250, 241, 0.95)",
      stroke: layerColors[layerIndex],
      "stroke-width": 1.2,
    });
    svg.appendChild(badge);

    const badgeText = svgNode("text", {
      x: width - 93,
      y: y,
      "font-size": 11,
      "font-weight": 700,
      "text-anchor": "middle",
      fill: layerColors[layerIndex],
    });
    badgeText.textContent = "6 disjoint CNOTs";
    svg.appendChild(badgeText);
  });

  renderZoomableSvg(svg, "shallow", {
    maxScale: 3,
    hint: "Zoom in to follow the three matchings on the shared qubit register. Ctrl-scroll also works.",
  });
  refs.visualCaption.textContent =
    `Illustrative 12-qubit slice of the paper's shallow construction. The same qubit ordering is reused in every row, and each row is one disjoint matching executed in a single CNOT depth layer. The theorem's full construction adds ancilla overhead and chooses these matchings so fault tolerance is preserved.`;
}

// The exact Theorem 5.6 circuit, derived from the bundled marked 3-regular graph
// for the current (n, t): every vertex/mark becomes a 3-qubit CAT spider and
// adjacent spiders are fused with Bell measurements, giving a genuine depth-3
// preparation. Reuses getNearestSpiderGraph so it lights up wherever a graph is
// bundled, snapping to the closest available n when there is no exact instance.
function getShallowBundle() {
  const bundle = getNearestSpiderGraph(state.t, state.n);
  if (!bundle) {
    return null;
  }
  const construction = buildShallowConstruction(bundle.entry);
  return { ...bundle, construction, stim: shallowStimText(construction, bundle.entry) };
}

function shallowFileName(construction) {
  return `shallow_cat_n${construction.n}_t${construction.t}.stim`;
}

function renderShallowCircuit(model) {
  clearVisual();
  const bundle = getShallowBundle();
  if (!bundle) {
    refs.visualHost.innerHTML = `<div class="empty-state">No marked 3-regular graph is bundled for t = ${state.t}, so the shallow circuit cannot be constructed.</div>`;
    refs.visualCaption.textContent =
      "The shallow circuit is extracted from the bundled SpiderCat graphs. Try t = 2 through t = 7.";
    return;
  }

  legendPills([
    { color: "var(--data-wire)", label: "data qubit (output)" },
    { color: "var(--flag-wire)", label: "ancilla leg (measured out)" },
    { color: "var(--ink)", label: "CNOT (• control / ⊕ target)" },
    { color: "var(--spider-stroke)", label: "Bell measurement" },
  ]);

  const { construction, exact } = bundle;
  const { ops, numQubits, initialKet } = parseStimCircuit(bundle.stim);
  // Schedule only the gates; every measurement is terminal for its wire, so they
  // all sit in one final column.
  const gateOps = ops.filter((op) => op.type !== "m");
  const measureOps = ops.filter((op) => op.type === "m");
  // Shallow-specific layout: all Hadamards in one column, CNOTs packed so their
  // vertical spans never overlap (see scheduleShallowOps).
  const numCols = scheduleShallowOps(gateOps, numQubits);
  const dataQubits = construction.dataQubits;

  // Comfortable row spacing now that the viewport scrolls instead of squeezing
  // every wire into the visible height.
  const rowGap = numQubits <= 24 ? 26 : numQubits <= 60 ? 18 : numQubits <= 140 ? 13 : 10;
  const colGap = 30;
  const leftPad = 96;
  const rightPad = 36;
  const topPad = 28;
  const bottomPad = 28;

  const colX = (col) => leftPad + col * colGap + colGap / 2;
  const rowY = (qubit) => topPad + qubit * rowGap;

  const dotR = Math.min(3.6, rowGap * 0.26);
  const targetR = Math.min(5.2, rowGap * 0.34);
  const meterW = Math.min(18, colGap * 0.62);
  const meterH = Math.min(13, rowGap * 0.8);

  // All measurements live in one column past the last gate.
  const meterX = colX(numCols);
  const wireEndX = meterX + meterW / 2 + 8;
  const width = wireEndX + rightPad;
  const height = topPad + numQubits * rowGap + bottomPad;

  // Drag bounds: keep CNOT endpoints inside the gate area, left of the meters.
  const dragMin = leftPad + 8;
  const dragMax = meterX - meterW / 2 - 10;

  const circuitId = `shallowcircuit-${state.t}-${state.n}`;
  if (!state.circuitDragOverrides[circuitId]) {
    state.circuitDragOverrides[circuitId] = {};
  }
  const overrides = state.circuitDragOverrides[circuitId];

  const svg = svgNode("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": `Optimal shallow CAT circuit for n = ${construction.n}, t = ${construction.t}`,
  });

  const wireLayer = svgNode("g");
  const gateLayer = svgNode("g");
  const handleLayer = svgNode("g");

  // Wires + a single "q12 |0>"-style label tucked against the left of each wire.
  for (let qubit = 0; qubit < numQubits; qubit += 1) {
    const y = rowY(qubit);
    const isData = qubit < dataQubits;
    wireLayer.appendChild(
      svgNode("line", {
        x1: leftPad,
        y1: y,
        x2: wireEndX,
        y2: y,
        stroke: isData ? "var(--data-wire)" : "var(--flag-wire)",
        "stroke-width": isData ? 1.8 : 1.2,
        "stroke-opacity": isData ? 0.6 : 0.4,
      }),
    );

    if (numQubits <= 96 || isData) {
      const labelSize = Math.max(9, Math.min(12, rowGap * 0.56));
      const label = svgNode("text", {
        x: leftPad - 8,
        y: y + labelSize * 0.34,
        "font-size": labelSize,
        "text-anchor": "end",
      });
      const idSpan = svgNode("tspan", { fill: "var(--muted)", "font-weight": 400 });
      idSpan.textContent = `${isData ? `q${qubit}` : `a${qubit - dataQubits}`} `;
      const ketSpan = svgNode("tspan", { fill: "var(--ink)", "font-weight": 600 });
      ketSpan.textContent = initialKet[qubit] === "+" ? "|+⟩" : "|0⟩";
      label.append(idSpan, ketSpan);
      wireLayer.appendChild(label);
    }
  }

  // Hadamard boxes (fixed position).
  for (const op of gateOps) {
    if (op.type !== "h") {
      continue;
    }
    const x = colX(op.col);
    const y = rowY(op.qubit);
    const boxW = Math.min(16, colGap * 0.52);
    const boxH = Math.min(14, rowGap * 0.8);
    gateLayer.appendChild(
      svgNode("rect", {
        x: x - boxW / 2,
        y: y - boxH / 2,
        width: boxW,
        height: boxH,
        rx: 3,
        fill: "#ffffff",
        stroke: "var(--ink)",
        "stroke-width": 1.4,
      }),
    );
    const label = svgNode("text", {
      x,
      y: y + boxH * 0.32,
      "font-size": Math.min(11, boxH * 0.82),
      "font-weight": 700,
      "text-anchor": "middle",
      fill: "var(--ink)",
    });
    label.textContent = "H";
    gateLayer.appendChild(label);
  }

  // Bell-measurement meters, aligned in the final column.
  for (const op of measureOps) {
    const y = rowY(op.qubit);
    gateLayer.appendChild(
      svgNode("rect", {
        x: meterX - meterW / 2,
        y: y - meterH / 2,
        width: meterW,
        height: meterH,
        rx: 3,
        fill: "#ffffff",
        stroke: "var(--spider-stroke)",
        "stroke-width": 1.4,
      }),
    );
    const arcR = meterW * 0.3;
    const arcY = y + meterH * 0.16;
    gateLayer.appendChild(
      svgNode("path", {
        d: `M ${meterX - arcR} ${arcY} A ${arcR} ${arcR} 0 0 1 ${meterX + arcR} ${arcY}`,
        fill: "none",
        stroke: "var(--spider-stroke)",
        "stroke-width": 1.2,
      }),
    );
    gateLayer.appendChild(
      svgNode("line", {
        x1: meterX,
        y1: arcY,
        x2: meterX + arcR * 0.78,
        y2: arcY - arcR * 0.82,
        stroke: "var(--spider-stroke)",
        "stroke-width": 1.2,
      }),
    );
  }

  // CNOTs: control dot + target ⊕ joined by a connector. Each endpoint can be
  // dragged horizontally along its own wire so overlapping depth-3 gates can be
  // spread out for readability.
  function eventPointInSvg(event) {
    const ctm = svg.getScreenCTM();
    if (!ctm) {
      return null;
    }
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    return point.matrixTransform(ctm.inverse());
  }

  let gateIndex = 0;
  const gateVisuals = [];
  let activeDrag = null;
  const hitR = Math.max(8, targetR + 4);

  for (const op of gateOps) {
    if (op.type !== "cx") {
      continue;
    }
    const index = gateIndex;
    gateIndex += 1;
    const gate = { index, defaultX: colX(op.col) };
    const yc = rowY(op.control);
    const yt = rowY(op.target);

    const connector = svgNode("line", {
      stroke: "var(--ink)",
      "stroke-width": 1.5,
      "stroke-linecap": "round",
    });
    gateLayer.appendChild(connector);

    const controlHandle = svgNode("g", {
      class: "circuit-gate-handle",
      tabindex: "0",
      role: "button",
      "aria-label": `CNOT ${index} control on wire ${op.control}`,
    });
    controlHandle.appendChild(svgNode("circle", { cx: 0, cy: 0, r: dotR, fill: "var(--ink)" }));
    controlHandle.appendChild(svgNode("circle", { cx: 0, cy: 0, r: hitR, fill: "transparent" }));

    const targetHandle = svgNode("g", {
      class: "circuit-gate-handle",
      tabindex: "0",
      role: "button",
      "aria-label": `CNOT ${index} target on wire ${op.target}`,
    });
    targetHandle.appendChild(
      svgNode("circle", { cx: 0, cy: 0, r: targetR, fill: "#ffffff", stroke: "var(--ink)", "stroke-width": 1.5 }),
    );
    targetHandle.appendChild(
      svgNode("line", { x1: -targetR, y1: 0, x2: targetR, y2: 0, stroke: "var(--ink)", "stroke-width": 1.5 }),
    );
    targetHandle.appendChild(
      svgNode("line", { x1: 0, y1: -targetR, x2: 0, y2: targetR, stroke: "var(--ink)", "stroke-width": 1.5 }),
    );
    targetHandle.appendChild(svgNode("circle", { cx: 0, cy: 0, r: hitR, fill: "transparent" }));

    handleLayer.appendChild(controlHandle);
    handleLayer.appendChild(targetHandle);

    function update() {
      const override = overrides[index] || {};
      const cx = override.cx ?? gate.defaultX;
      const tx = override.tx ?? gate.defaultX;
      connector.setAttribute("x1", cx);
      connector.setAttribute("y1", yc);
      connector.setAttribute("x2", tx);
      connector.setAttribute("y2", yt);
      controlHandle.setAttribute("transform", `translate(${cx} ${yc})`);
      targetHandle.setAttribute("transform", `translate(${tx} ${yt})`);
    }

    function startDrag(endpoint, handle, event) {
      event.preventDefault();
      activeDrag = { index, endpoint, pointerId: event.pointerId, update };
      handle.classList.add("dragging");
      svg.classList.add("dragging-graph");
      handle.setPointerCapture(event.pointerId);
    }

    controlHandle.addEventListener("pointerdown", (event) => startDrag("cx", controlHandle, event));
    targetHandle.addEventListener("pointerdown", (event) => startDrag("tx", targetHandle, event));
    // Double-click an endpoint to snap this CNOT back to its scheduled column.
    controlHandle.addEventListener("dblclick", () => {
      delete overrides[index];
      update();
    });
    targetHandle.addEventListener("dblclick", () => {
      delete overrides[index];
      update();
    });

    gateVisuals.push({ controlHandle, targetHandle, update });
    update();
  }

  function moveActiveDrag(event) {
    if (!activeDrag || activeDrag.pointerId !== event.pointerId) {
      return;
    }
    const point = eventPointInSvg(event);
    if (!point) {
      return;
    }
    const x = clampNumber(point.x, dragMin, dragMax);
    overrides[activeDrag.index] = { ...(overrides[activeDrag.index] || {}), [activeDrag.endpoint]: x };
    activeDrag.update();
  }

  function endDrag(event) {
    if (!activeDrag || activeDrag.pointerId !== event.pointerId) {
      return;
    }
    for (const visual of gateVisuals) {
      visual.controlHandle.classList.remove("dragging");
      visual.targetHandle.classList.remove("dragging");
    }
    svg.classList.remove("dragging-graph");
    activeDrag = null;
  }

  svg.addEventListener("pointermove", moveActiveDrag);
  svg.addEventListener("pointerup", endDrag);
  svg.addEventListener("pointercancel", endDrag);

  svg.appendChild(wireLayer);
  svg.appendChild(gateLayer);
  svg.appendChild(handleLayer);

  renderZoomableSvg(svg, circuitId, {
    minScale: 0.2,
    maxScale: 8,
    naturalSize: { width, height },
    hint: "Large circuits render at full size — drag the scrollbars to pan up/down and left/right. Drag a CNOT's control (•) or target (⊕) to nudge it; double-click to reset. Ctrl-scroll to zoom; zoom out to see the whole circuit.",
  });

  const rt = data.paper.optimal.rtValues[String(construction.t)];
  const optimalCx = rt != null ? Math.ceil(((29 * rt + 26) / 10) * construction.n) : null;
  const snap = exact
    ? ""
    : ` No exact graph is bundled at n = ${state.n}, so this uses the nearest available instance at n = ${construction.n}.`;
  refs.visualCaption.textContent =
    `Explicit Theorem 5.6 depth-3 circuit for n = ${construction.n}, t = ${construction.t}, built from the marked 3-regular graph ` +
    `(${construction.numVertices} vertices + ${construction.numMarkSpiders} marks → ${construction.spiders.length} three-qubit CAT spiders fused by ${construction.fusions.length} Bell measurements). ` +
    `${construction.cnotCount} CNOTs in CNOT depth 3 on ${construction.numQubits} qubits (${construction.ancillaCount} ancillae)` +
    (optimalCx != null ? `; the theorem's direct-CNOT pass lowers this to ${optimalCx} CNOTs.` : ".") +
    ` Data wires q0…q${construction.n - 1} carry the output cat state; a0, a1, … are ancilla legs consumed by the Bell measurements.${snap} ` +
    `The Hadamards are laid out in a single layer and the CNOTs are spread across columns so no two connectors overlap, which makes the diagram wide — drag the scrollbars to pan, or zoom out for the full view. Use Export above to download the Stim source.`;
}

function appendShallowExport() {
  const bundle = getShallowBundle();
  if (!bundle) {
    return;
  }
  const { construction, stim } = bundle;

  const panel = document.createElement("div");
  panel.className = "export-panel";

  const summary = document.createElement("p");
  summary.className = "export-summary";
  summary.textContent =
    `Stim circuit: optimal-shallow CAT^${construction.n}, t = ${construction.t} — ` +
    `${construction.spiders.length} three-qubit CAT spiders, ${construction.fusions.length} Bell measurements, ` +
    `${construction.cnotCount} CNOTs in depth 3.`;
  panel.appendChild(summary);

  const controls = document.createElement("div");
  controls.className = "export-controls";

  const downloadButton = document.createElement("button");
  downloadButton.type = "button";
  downloadButton.className = "export-button";
  downloadButton.textContent = "Download .stim";
  downloadButton.addEventListener("click", () => {
    downloadTextFile(shallowFileName(construction), stim);
  });
  controls.appendChild(downloadButton);

  const copyButton = document.createElement("button");
  copyButton.type = "button";
  copyButton.className = "export-button ghost";
  copyButton.textContent = "Copy";
  copyButton.addEventListener("click", () => {
    const done = () => {
      copyButton.textContent = "Copied!";
      copyButton.classList.add("copied");
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(stim).then(done).catch(() => {
        copyButton.textContent = "Copy failed";
      });
    } else {
      done();
    }
  });
  controls.appendChild(copyButton);

  panel.appendChild(controls);
  refs.visualHost.appendChild(panel);
}

function flagFileName(metric) {
  return `flag_at_origin_cat_n${metric.n}_t${metric.t}.stim`;
}

// Reconstruct a Stim source from the bundled flag-at-origin CNOT layers. The
// repo QASM is just `h q[0]` + a CNOT fan-out from the origin data qubit; here
// we add the Z-basis flag readout that completes the flagged preparation.
function flagStimText(metric) {
  const { n, t, numQubits, numFlags, layers, sourcePath } = metric;
  const origin = numFlags; // global index of q[0], the origin spider in |+>
  const numCx = layers.reduce((sum, layer) => sum + layer.length, 0);

  const lines = [
    `# Flag-at-origin GHZ / CAT^${n} state preparation (t = ${t})`,
    `# Source: ${sourcePath || `flag d${2 * t + 1}-q${n}`}`,
    `# Qubits 0..${numFlags - 1} are flags f0..f${numFlags - 1} (Z-measured at the end);`,
    `#   qubits ${numFlags}..${numQubits - 1} are data q0..q${n - 1} (output cat state), origin = q0 in |+>.`,
    `# ${numCx} CNOTs across ${layers.length} entangling layers.`,
  ];

  const allQubits = Array.from({ length: numQubits }, (_, q) => q).join(" ");
  lines.push(`R ${allQubits}`);
  lines.push(`H ${origin}`);
  lines.push("TICK");
  for (const layer of layers) {
    if (!layer.length) {
      continue;
    }
    const pairs = layer.map(([control, target]) => `${control} ${target}`).join(" ");
    lines.push(`CX ${pairs}`);
    lines.push("TICK");
  }
  if (numFlags > 0) {
    const flagIdx = Array.from({ length: numFlags }, (_, i) => i).join(" ");
    lines.push(`M ${flagIdx}`); // flag readout in the Z basis
  }
  return `${lines.join("\n")}\n`;
}

function renderFlagCircuit(model) {
  clearVisual();
  const metric = model.actual;
  if (!metric || !metric.layers) {
    renderEmptyVisual(
      `No bundled flag-at-origin circuit is available for n = ${state.n}, t = ${state.t}.`,
      "The circuit view renders the exact repo QASM, available only at bundled (n, t) points. Try a nearby (n, t).",
    );
    return;
  }

  legendPills([
    { color: "var(--data-wire)", label: "data qubit (output)" },
    { color: "var(--flag-wire)", label: "flag (measured out)" },
    { color: "var(--ink)", label: "CNOT (• control / ⊕ target)" },
    { color: "var(--spider-stroke)", label: "Z measurement" },
  ]);

  const { numQubits, numFlags, layers } = metric;
  const origin = numFlags; // q[0], the origin spider in |+>

  // Flatten the layers into a draggable CNOT list. Column 0 holds the origin
  // Hadamard; entangling layer i sits in column i + 1.
  const cxGates = [];
  layers.forEach((pairs, layerIndex) => {
    pairs.forEach(([control, target]) => {
      cxGates.push({ control, target, col: layerIndex + 1 });
    });
  });
  const numCols = layers.length + 1;

  const rowGap = numQubits <= 24 ? 24 : numQubits <= 60 ? 15 : numQubits <= 140 ? 10 : 7;
  const colGap = 30;
  const leftPad = 96;
  const rightPad = 36;
  const topPad = 28;
  const bottomPad = 28;

  const colX = (col) => leftPad + col * colGap + colGap / 2;
  const rowY = (qubit) => topPad + qubit * rowGap;

  const dotR = Math.min(4, rowGap * 0.26);
  const targetR = Math.min(5.6, rowGap * 0.34);
  const meterW = Math.min(18, colGap * 0.62);
  const meterH = Math.min(13, rowGap * 0.8);

  // Flag readout lives in one column past the last gate.
  const meterX = colX(numCols);
  const wireEndX = meterX + meterW / 2 + 8;
  const width = wireEndX + rightPad;
  const height = topPad + numQubits * rowGap + bottomPad;

  // Drag bounds: keep CNOT endpoints inside the gate area, left of the meters.
  const dragMin = leftPad + 8;
  const dragMax = meterX - meterW / 2 - 10;

  const circuitId = `flagcircuit-${state.t}-${state.n}`;
  if (!state.circuitDragOverrides[circuitId]) {
    state.circuitDragOverrides[circuitId] = {};
  }
  const overrides = state.circuitDragOverrides[circuitId];

  const svg = svgNode("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": `Flag-at-origin CAT circuit for n = ${state.n}, t = ${state.t}`,
  });

  const wireLayer = svgNode("g");
  const gateLayer = svgNode("g");
  const handleLayer = svgNode("g");

  // Wires + labels. Flags are global indices 0..numFlags-1; data are the rest.
  for (let qubit = 0; qubit < numQubits; qubit += 1) {
    const y = rowY(qubit);
    const isData = qubit >= numFlags;
    wireLayer.appendChild(
      svgNode("line", {
        x1: leftPad,
        y1: y,
        x2: wireEndX,
        y2: y,
        stroke: isData ? "var(--data-wire)" : "var(--flag-wire)",
        "stroke-width": isData ? 1.8 : 1.2,
        "stroke-opacity": isData ? 0.6 : 0.4,
      }),
    );

    if (numQubits <= 96 || isData) {
      const labelSize = Math.max(9, Math.min(12, rowGap * 0.56));
      const label = svgNode("text", {
        x: leftPad - 8,
        y: y + labelSize * 0.34,
        "font-size": labelSize,
        "text-anchor": "end",
      });
      const idSpan = svgNode("tspan", { fill: "var(--muted)", "font-weight": 400 });
      idSpan.textContent = `${isData ? `q${qubit - numFlags}` : `f${qubit}`} `;
      const ketSpan = svgNode("tspan", { fill: "var(--ink)", "font-weight": 600 });
      ketSpan.textContent = qubit === origin ? "|+⟩" : "|0⟩";
      label.append(idSpan, ketSpan);
      wireLayer.appendChild(label);
    }
  }

  // Origin Hadamard box (fixed position, column 0).
  {
    const x = colX(0);
    const y = rowY(origin);
    const boxW = Math.min(16, colGap * 0.52);
    const boxH = Math.min(14, rowGap * 0.8);
    gateLayer.appendChild(
      svgNode("rect", {
        x: x - boxW / 2,
        y: y - boxH / 2,
        width: boxW,
        height: boxH,
        rx: 3,
        fill: "#ffffff",
        stroke: "var(--ink)",
        "stroke-width": 1.4,
      }),
    );
    const label = svgNode("text", {
      x,
      y: y + boxH * 0.32,
      "font-size": Math.min(11, boxH * 0.82),
      "font-weight": 700,
      "text-anchor": "middle",
      fill: "var(--ink)",
    });
    label.textContent = "H";
    gateLayer.appendChild(label);
  }

  // Flag Z-measurement meters, aligned in the final column.
  for (let qubit = 0; qubit < numFlags; qubit += 1) {
    const y = rowY(qubit);
    gateLayer.appendChild(
      svgNode("rect", {
        x: meterX - meterW / 2,
        y: y - meterH / 2,
        width: meterW,
        height: meterH,
        rx: 3,
        fill: "#ffffff",
        stroke: "var(--spider-stroke)",
        "stroke-width": 1.4,
      }),
    );
    const arcR = meterW * 0.3;
    const arcY = y + meterH * 0.16;
    gateLayer.appendChild(
      svgNode("path", {
        d: `M ${meterX - arcR} ${arcY} A ${arcR} ${arcR} 0 0 1 ${meterX + arcR} ${arcY}`,
        fill: "none",
        stroke: "var(--spider-stroke)",
        "stroke-width": 1.2,
      }),
    );
    gateLayer.appendChild(
      svgNode("line", {
        x1: meterX,
        y1: arcY,
        x2: meterX + arcR * 0.78,
        y2: arcY - arcR * 0.82,
        stroke: "var(--spider-stroke)",
        "stroke-width": 1.2,
      }),
    );
  }

  // CNOTs: control dot + target ⊕ joined by a connector. Each endpoint can be
  // dragged horizontally along its own wire so overlapping gates can be spread.
  function eventPointInSvg(event) {
    const ctm = svg.getScreenCTM();
    if (!ctm) {
      return null;
    }
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    return point.matrixTransform(ctm.inverse());
  }

  const gateVisuals = [];
  let activeDrag = null;
  const hitR = Math.max(9, targetR + 4);

  cxGates.forEach((op, index) => {
    const gate = { index, defaultX: colX(op.col) };
    const yc = rowY(op.control);
    const yt = rowY(op.target);

    const connector = svgNode("line", {
      stroke: "var(--ink)",
      "stroke-width": 1.7,
      "stroke-linecap": "round",
    });
    gateLayer.appendChild(connector);

    const controlHandle = svgNode("g", {
      class: "circuit-gate-handle",
      tabindex: "0",
      role: "button",
      "aria-label": `CNOT ${index} control`,
    });
    controlHandle.appendChild(svgNode("circle", { cx: 0, cy: 0, r: dotR, fill: "var(--ink)" }));
    controlHandle.appendChild(svgNode("circle", { cx: 0, cy: 0, r: hitR, fill: "transparent" }));

    const targetHandle = svgNode("g", {
      class: "circuit-gate-handle",
      tabindex: "0",
      role: "button",
      "aria-label": `CNOT ${index} target`,
    });
    targetHandle.appendChild(
      svgNode("circle", { cx: 0, cy: 0, r: targetR, fill: "#ffffff", stroke: "var(--ink)", "stroke-width": 1.7 }),
    );
    targetHandle.appendChild(
      svgNode("line", { x1: -targetR, y1: 0, x2: targetR, y2: 0, stroke: "var(--ink)", "stroke-width": 1.7 }),
    );
    targetHandle.appendChild(
      svgNode("line", { x1: 0, y1: -targetR, x2: 0, y2: targetR, stroke: "var(--ink)", "stroke-width": 1.7 }),
    );
    targetHandle.appendChild(svgNode("circle", { cx: 0, cy: 0, r: hitR, fill: "transparent" }));

    handleLayer.appendChild(controlHandle);
    handleLayer.appendChild(targetHandle);

    function update() {
      const override = overrides[index] || {};
      const cx = override.cx ?? gate.defaultX;
      const tx = override.tx ?? gate.defaultX;
      connector.setAttribute("x1", cx);
      connector.setAttribute("y1", yc);
      connector.setAttribute("x2", tx);
      connector.setAttribute("y2", yt);
      controlHandle.setAttribute("transform", `translate(${cx} ${yc})`);
      targetHandle.setAttribute("transform", `translate(${tx} ${yt})`);
    }

    function startDrag(endpoint, handle, event) {
      event.preventDefault();
      activeDrag = { index, endpoint, pointerId: event.pointerId, update };
      handle.classList.add("dragging");
      svg.classList.add("dragging-graph");
      handle.setPointerCapture(event.pointerId);
    }

    controlHandle.addEventListener("pointerdown", (event) => startDrag("cx", controlHandle, event));
    targetHandle.addEventListener("pointerdown", (event) => startDrag("tx", targetHandle, event));
    // Double-click an endpoint to snap this CNOT back to its scheduled column.
    controlHandle.addEventListener("dblclick", () => {
      delete overrides[index];
      update();
    });
    targetHandle.addEventListener("dblclick", () => {
      delete overrides[index];
      update();
    });

    gateVisuals.push({ controlHandle, targetHandle, update });
    update();
  });

  function moveActiveDrag(event) {
    if (!activeDrag || activeDrag.pointerId !== event.pointerId) {
      return;
    }
    const point = eventPointInSvg(event);
    if (!point) {
      return;
    }
    const x = clampNumber(point.x, dragMin, dragMax);
    overrides[activeDrag.index] = { ...(overrides[activeDrag.index] || {}), [activeDrag.endpoint]: x };
    activeDrag.update();
  }

  function endDrag(event) {
    if (!activeDrag || activeDrag.pointerId !== event.pointerId) {
      return;
    }
    for (const visual of gateVisuals) {
      visual.controlHandle.classList.remove("dragging");
      visual.targetHandle.classList.remove("dragging");
    }
    svg.classList.remove("dragging-graph");
    activeDrag = null;
  }

  svg.addEventListener("pointermove", moveActiveDrag);
  svg.addEventListener("pointerup", endDrag);
  svg.addEventListener("pointercancel", endDrag);

  svg.appendChild(wireLayer);
  svg.appendChild(gateLayer);
  svg.appendChild(handleLayer);

  renderZoomableSvg(svg, circuitId, {
    maxScale: 6,
    hint: "Drag a CNOT's control (•) or target (⊕) sideways to spread out overlapping gates; double-click an endpoint to reset it. Ctrl-scroll to zoom.",
  });

  const sourceName = metric.sourcePath ? metric.sourcePath.split("/").pop() : "QASM";
  refs.visualCaption.textContent =
    `Exact flag-at-origin CAT^${metric.n} circuit for n = ${state.n}, t = ${state.t} (${metric.numCx} CNOTs across ${layers.length} entangling layers), rendered from the bundled ${sourceName}. The origin data qubit q0 starts in |+⟩ and fans out across the flag (f0, f1, …) and data (q0 … q${metric.n - 1}) wires; the flags are measured in the Z basis at the end. Because several CNOTs share each time slice, drag any control or target sideways to declutter; double-click an endpoint to reset. Use Export above to download the Stim source.`;
}

function appendFlagExport(metric) {
  if (!metric || !metric.layers) {
    return;
  }
  const stim = flagStimText(metric);

  const panel = document.createElement("div");
  panel.className = "export-panel";

  const summary = document.createElement("p");
  summary.className = "export-summary";
  summary.textContent =
    `Stim circuit: flag-at-origin CAT^${metric.n}, t = ${metric.t} — ` +
    `${metric.numCx} CNOTs across ${metric.layers.length} entangling layers, ${metric.numFlags} flag qubits.`;
  panel.appendChild(summary);

  const controls = document.createElement("div");
  controls.className = "export-controls";

  const downloadButton = document.createElement("button");
  downloadButton.type = "button";
  downloadButton.className = "export-button";
  downloadButton.textContent = "Download .stim";
  downloadButton.addEventListener("click", () => {
    downloadTextFile(flagFileName(metric), stim);
  });
  controls.appendChild(downloadButton);

  const copyButton = document.createElement("button");
  copyButton.type = "button";
  copyButton.className = "export-button ghost";
  copyButton.textContent = "Copy";
  copyButton.addEventListener("click", () => {
    const done = () => {
      copyButton.textContent = "Copied!";
      copyButton.classList.add("copied");
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(stim).then(done).catch(() => {
        copyButton.textContent = "Copy failed";
      });
    } else {
      done();
    }
  });
  controls.appendChild(copyButton);

  panel.appendChild(controls);
  refs.visualHost.appendChild(panel);
}

// Draggable circuit view for the MQT baseline. MQT qubits are laid out data-first
// (q0..q_{n-1}) with ancilla/flag qubits afterwards, mirroring the bundled Stim.
// Only the entangling CNOT layers are drawn; the full circuit (origin Hadamard and
// flag measurements) lives in the bundled Stim source offered by appendMqtExport.
function renderMqtCircuit(model) {
  clearVisual();
  const metric = model.actual;
  if (!metric || !metric.layers) {
    renderEmptyVisual(
      `No bundled MQT circuit is available for n = ${state.n}, t = ${state.t}.`,
      "The circuit view renders the exact repo Stim circuit, available only at bundled (n, t) points. Try a nearby (n, t).",
    );
    return;
  }

  legendPills([
    { color: "var(--data-wire)", label: "data qubit (output)" },
    { color: "var(--flag-wire)", label: "ancilla / flag" },
    { color: "var(--ink)", label: "CNOT (• control / ⊕ target)" },
  ]);

  const { numQubits, layers } = metric;
  const dataQubits = state.n;

  // Flatten the layers into a draggable CNOT list. Each entangling layer is
  // meant to run in parallel, but two CNOTs in the same layer whose vertical
  // spans [min(control, target), max(control, target)] overlap would draw their
  // connectors on top of one another. So within each layer we interval-pack the
  // CNOTs into sub-columns (the minimum count that keeps every span disjoint per
  // column), preserving the layer/depth grouping while spreading overlaps apart.
  // Each endpoint can later be dragged off its default column.
  const cxGates = [];
  let colBase = 0;
  layers.forEach((pairs) => {
    const subColumnSpans = []; // subColumnSpans[i] = array of [lo, hi] CNOT spans
    pairs.forEach(([control, target]) => {
      const lo = Math.min(control, target);
      const hi = Math.max(control, target);
      let sub = 0;
      for (;;) {
        const spans = subColumnSpans[sub] || (subColumnSpans[sub] = []);
        const overlaps = spans.some(([a, b]) => lo <= b && a <= hi);
        if (!overlaps) {
          spans.push([lo, hi]);
          break;
        }
        sub += 1;
      }
      cxGates.push({ control, target, col: colBase + sub });
    });
    colBase += Math.max(subColumnSpans.length, 1);
  });
  const numCols = Math.max(colBase, 1);

  const rowGap = numQubits <= 24 ? 24 : numQubits <= 60 ? 15 : numQubits <= 140 ? 10 : 7;
  const colGap = 30;
  const leftPad = 96;
  const rightPad = 36;
  const topPad = 28;
  const bottomPad = 28;

  const colX = (col) => leftPad + col * colGap + colGap / 2;
  const rowY = (qubit) => topPad + qubit * rowGap;

  const dotR = Math.min(4, rowGap * 0.26);
  const targetR = Math.min(5.6, rowGap * 0.34);

  const wireEndX = colX(numCols - 1) + colGap / 2 + 8;
  const width = wireEndX + rightPad;
  const height = topPad + numQubits * rowGap + bottomPad;

  // Drag bounds: keep CNOT endpoints inside the gate area.
  const dragMin = leftPad + 8;
  const dragMax = wireEndX - 10;

  const circuitId = `mqtcircuit-${state.t}-${state.n}`;
  if (!state.circuitDragOverrides[circuitId]) {
    state.circuitDragOverrides[circuitId] = {};
  }
  const overrides = state.circuitDragOverrides[circuitId];

  const svg = svgNode("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": `MQT CAT circuit for n = ${state.n}, t = ${state.t}`,
  });

  const wireLayer = svgNode("g");
  const gateLayer = svgNode("g");
  const handleLayer = svgNode("g");

  // Wires + labels. Data qubits are global indices 0..n-1; ancillae are the rest.
  for (let qubit = 0; qubit < numQubits; qubit += 1) {
    const y = rowY(qubit);
    const isData = qubit < dataQubits;
    wireLayer.appendChild(
      svgNode("line", {
        x1: leftPad,
        y1: y,
        x2: wireEndX,
        y2: y,
        stroke: isData ? "var(--data-wire)" : "var(--flag-wire)",
        "stroke-width": isData ? 1.8 : 1.2,
        "stroke-opacity": isData ? 0.6 : 0.4,
      }),
    );

    if (numQubits <= 96 || isData) {
      const labelSize = Math.max(9, Math.min(12, rowGap * 0.56));
      const label = svgNode("text", {
        x: leftPad - 8,
        y: y + labelSize * 0.34,
        "font-size": labelSize,
        "text-anchor": "end",
        fill: "var(--muted)",
      });
      label.textContent = isData ? `q${qubit}` : `a${qubit - dataQubits}`;
      wireLayer.appendChild(label);
    }
  }

  // CNOTs: control dot + target ⊕ joined by a connector. Each endpoint can be
  // dragged horizontally along its own wire so overlapping gates can be spread.
  function eventPointInSvg(event) {
    const ctm = svg.getScreenCTM();
    if (!ctm) {
      return null;
    }
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    return point.matrixTransform(ctm.inverse());
  }

  const gateVisuals = [];
  let activeDrag = null;
  const hitR = Math.max(9, targetR + 4);

  cxGates.forEach((op, index) => {
    const gate = { index, defaultX: colX(op.col) };
    const yc = rowY(op.control);
    const yt = rowY(op.target);

    const connector = svgNode("line", {
      stroke: "var(--ink)",
      "stroke-width": 1.7,
      "stroke-linecap": "round",
    });
    gateLayer.appendChild(connector);

    const controlHandle = svgNode("g", {
      class: "circuit-gate-handle",
      tabindex: "0",
      role: "button",
      "aria-label": `CNOT ${index} control`,
    });
    controlHandle.appendChild(svgNode("circle", { cx: 0, cy: 0, r: dotR, fill: "var(--ink)" }));
    controlHandle.appendChild(svgNode("circle", { cx: 0, cy: 0, r: hitR, fill: "transparent" }));

    const targetHandle = svgNode("g", {
      class: "circuit-gate-handle",
      tabindex: "0",
      role: "button",
      "aria-label": `CNOT ${index} target`,
    });
    targetHandle.appendChild(
      svgNode("circle", { cx: 0, cy: 0, r: targetR, fill: "#ffffff", stroke: "var(--ink)", "stroke-width": 1.7 }),
    );
    targetHandle.appendChild(
      svgNode("line", { x1: -targetR, y1: 0, x2: targetR, y2: 0, stroke: "var(--ink)", "stroke-width": 1.7 }),
    );
    targetHandle.appendChild(
      svgNode("line", { x1: 0, y1: -targetR, x2: 0, y2: targetR, stroke: "var(--ink)", "stroke-width": 1.7 }),
    );
    targetHandle.appendChild(svgNode("circle", { cx: 0, cy: 0, r: hitR, fill: "transparent" }));

    handleLayer.appendChild(controlHandle);
    handleLayer.appendChild(targetHandle);

    function update() {
      const override = overrides[index] || {};
      const cx = override.cx ?? gate.defaultX;
      const tx = override.tx ?? gate.defaultX;
      connector.setAttribute("x1", cx);
      connector.setAttribute("y1", yc);
      connector.setAttribute("x2", tx);
      connector.setAttribute("y2", yt);
      controlHandle.setAttribute("transform", `translate(${cx} ${yc})`);
      targetHandle.setAttribute("transform", `translate(${tx} ${yt})`);
    }

    function startDrag(endpoint, handle, event) {
      event.preventDefault();
      activeDrag = { index, endpoint, pointerId: event.pointerId, update };
      handle.classList.add("dragging");
      svg.classList.add("dragging-graph");
      handle.setPointerCapture(event.pointerId);
    }

    controlHandle.addEventListener("pointerdown", (event) => startDrag("cx", controlHandle, event));
    targetHandle.addEventListener("pointerdown", (event) => startDrag("tx", targetHandle, event));
    // Double-click an endpoint to snap this CNOT back to its scheduled column.
    controlHandle.addEventListener("dblclick", () => {
      delete overrides[index];
      update();
    });
    targetHandle.addEventListener("dblclick", () => {
      delete overrides[index];
      update();
    });

    gateVisuals.push({ controlHandle, targetHandle, update });
    update();
  });

  function moveActiveDrag(event) {
    if (!activeDrag || activeDrag.pointerId !== event.pointerId) {
      return;
    }
    const point = eventPointInSvg(event);
    if (!point) {
      return;
    }
    const x = clampNumber(point.x, dragMin, dragMax);
    overrides[activeDrag.index] = { ...(overrides[activeDrag.index] || {}), [activeDrag.endpoint]: x };
    activeDrag.update();
  }

  function endDrag(event) {
    if (!activeDrag || activeDrag.pointerId !== event.pointerId) {
      return;
    }
    for (const visual of gateVisuals) {
      visual.controlHandle.classList.remove("dragging");
      visual.targetHandle.classList.remove("dragging");
    }
    svg.classList.remove("dragging-graph");
    activeDrag = null;
  }

  svg.addEventListener("pointermove", moveActiveDrag);
  svg.addEventListener("pointerup", endDrag);
  svg.addEventListener("pointercancel", endDrag);

  svg.appendChild(wireLayer);
  svg.appendChild(gateLayer);
  svg.appendChild(handleLayer);

  renderZoomableSvg(svg, circuitId, {
    minScale: 0.2,
    maxScale: 6,
    naturalSize: { width, height },
    hint: "Large circuits render at full size — drag the scrollbars to pan up/down and left/right. Drag a CNOT's control (•) or target (⊕) sideways to spread out overlapping gates; double-click an endpoint to reset. Ctrl-scroll to zoom; zoom out to see the whole circuit.",
  });

  const circuit = getMqtCircuit(state.n, state.t);
  const sourceName = circuit?.fileName || (metric.sourcePath ? metric.sourcePath.split("/").pop() : "Stim");
  refs.visualCaption.textContent =
    `MQT benchmark CAT^${metric.n} circuit for n = ${state.n}, t = ${state.t} (${metric.numCx} CNOTs across ${layers.length} entangling layers), from the bundled ${sourceName}. Data wires q0 … q${state.n - 1} carry the output cat state; ancilla wires a0, a1, … are the flag/verification qubits. Only the entangling CNOT layers are drawn — the exported Stim source also includes the origin Hadamard and the flag measurements. Because several CNOTs share each time slice, drag any control or target sideways to declutter; double-click an endpoint to reset. Use Export above to download the Stim source.`;
}

function appendMqtExport(model) {
  const circuit = getMqtCircuit(state.n, state.t);
  if (!circuit) {
    return;
  }
  const metric = model.actual;
  const stim = circuit.stim;

  const panel = document.createElement("div");
  panel.className = "export-panel";

  const summary = document.createElement("p");
  summary.className = "export-summary";
  summary.textContent =
    `Stim circuit: MQT CAT^${circuit.n}, t = ${circuit.t} — ` +
    `${metric ? `${metric.numCx} CNOTs across ${metric.layers.length} entangling layers, ` : ""}` +
    `exact bundled ${circuit.fileName}.`;
  panel.appendChild(summary);

  const controls = document.createElement("div");
  controls.className = "export-controls";

  const downloadButton = document.createElement("button");
  downloadButton.type = "button";
  downloadButton.className = "export-button";
  downloadButton.textContent = "Download .stim";
  downloadButton.addEventListener("click", () => {
    downloadTextFile(circuit.fileName, stim);
  });
  controls.appendChild(downloadButton);

  const copyButton = document.createElement("button");
  copyButton.type = "button";
  copyButton.className = "export-button ghost";
  copyButton.textContent = "Copy";
  copyButton.addEventListener("click", () => {
    const done = () => {
      copyButton.textContent = "Copied!";
      copyButton.classList.add("copied");
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(stim).then(done).catch(() => {
        copyButton.textContent = "Copy failed";
      });
    } else {
      done();
    }
  });
  controls.appendChild(copyButton);

  panel.appendChild(controls);
  refs.visualHost.appendChild(panel);
}

function renderEmptyVisual(message, caption) {
  clearVisual();
  refs.visualHost.innerHTML = `<div class="empty-state">${message}</div>`;
  refs.visualCaption.textContent = caption;
}

function renderDetailInfo(model) {
  const rows = [];
  if (model.metrics) {
    rows.push(["CNOT count", formatNumber(model.metrics.numCx)]);
    rows.push(["Depth", formatNumber(model.metrics.depth)]);
    rows.push(["Ancillae / flags", formatNumber(model.metrics.ancillas)]);
  } else {
    rows.push(["Availability", "n/a"]);
  }

  if (model.actual?.lowerBoundCnots != null) {
    rows.push(["Repo lower bound", formatNumber(model.actual.lowerBoundCnots)]);
  }

  if (model.noise) {
    rows.push(["Accept @ p=0.05", formatPercent(model.noise.acceptanceRate)]);
    rows.push(["Clean | accepted", formatPercent(model.noise.cleanGivenAccepted)]);
    rows.push(["Overall clean", formatPercent(model.noise.overallCleanRate)]);
  }

  let rangeNote = "";
  if (model.id === "shallow" && !model.available) {
    rangeNote = "Known r_t values from the paper are bundled up to t = 5.";
  } else if (!model.available) {
    rangeNote = "This exact point is not bundled in the repo.";
  } else if (model.id === "spidercat" && model.spiderGraph && !model.spiderGraph.exact) {
    rangeNote = `No exact SpiderCat graph is bundled at n = ${state.n}, t = ${state.t}; the graph view is available only at bundled (n, t) points.`;
  }

  // SpiderCat optimal ships concrete circuits in the repo, so offer a one-click
  // export of the exact bundled circuit for the current (n, t).
  const spiderCircuit = model.id === "spidercat" ? getSpiderCircuit(state.n, state.t) : null;
  const exportHtml = spiderCircuit
    ? `
      <div class="export-block">
        <span class="export-label">Export circuit</span>
        <button type="button" id="exportStimBtn" class="export-button">Download .stim</button>
        <span class="export-hint">Exact bundled SpiderCat circuit (Stim format) for n = ${state.n}, t = ${state.t}.</span>
      </div>
    `
    : model.id === "spidercat"
      ? `
      <div class="export-block">
        <span class="export-label">Export circuit</span>
        <span class="export-hint">No exact bundled circuit at n = ${state.n}, t = ${state.t} to export.</span>
      </div>
    `
      : "";

  refs.detailInfo.innerHTML = `
    <h3>${model.label}</h3>
    <p>${model.description}</p>
    <div class="detail-kpis">
      ${rows
        .map(
          ([label, value]) => `
            <div class="detail-kpi">
              <span>${label}</span>
              <strong>${value}</strong>
            </div>
          `,
        )
        .join("")}
    </div>
    <p><strong>${model.paperHook}</strong></p>
    <p>${model.note}</p>
    ${rangeNote ? `<p>${rangeNote}</p>` : ""}
    ${
      model.estimated
        ? `<p class="mono">${model.formulaLabel}</p>`
        : model.actual?.sourcePath
          ? `<p class="mono">${model.actual.sourcePath}</p>`
          : ""
    }
    ${exportHtml}
  `;

  if (spiderCircuit) {
    const exportButton = refs.detailInfo.querySelector("#exportStimBtn");
    exportButton?.addEventListener("click", () => {
      downloadText(spiderCircuit.fileName, spiderCircuit.stim, "text/plain");
    });
  }
}

function renderDetail(model) {
  refs.detailTitle.textContent = model.label;
  const impliedSuffix = isImpliedT() ? ` ${impliedTNote()}` : "";
  refs.detailSubtitle.textContent = `${model.kindLabel}. ${model.optimize}. ${model.paperHook}.${impliedSuffix}`;
  renderDetailInfo(model);

  if (model.id === "spidercat") {
    if (state.spiderView === "circuit") {
      renderSpiderCircuit(model);
    } else {
      renderSpiderGraph(model);
    }
    prependViewToggle(
      [
        { id: "graph", label: "Graph" },
        { id: "circuit", label: "Circuit" },
      ],
      state.spiderView,
      (view) => {
        state.spiderView = view;
        render();
      },
    );
    return;
  }

  if (model.id === "recursive") {
    if (state.recursiveView === "zx") {
      renderRecursiveZX(model);
    // } else if (state.recursiveView === "simplified") {
    //   renderRecursiveSimplified(model);
    } else {
      renderRecursiveSchematic(model);
    }
    appendRecursiveExport();
    prependViewToggle(
      [
        { id: "schematic", label: "Schematic" },
        { id: "zx", label: "ZX diagram" },
        // { id: "simplified", label: "Simplified" },
      ],
      state.recursiveView,
      (view) => {
        state.recursiveView = view;
        render();
      },
    );
    return;
  }

  if (model.id === "shallow") {
    const hasGraph = Boolean(getNearestSpiderGraph(state.t, state.n));
    if (state.shallowView === "circuit") {
      if (hasGraph) {
        renderShallowCircuit(model);
        appendShallowExport();
      } else {
        renderEmptyVisual(
          `No marked 3-regular graph is bundled for t = ${state.t}, so the Theorem 5.6 circuit cannot be built.`,
          "The explicit shallow circuit is extracted from the bundled SpiderCat graphs (t = 2 through t = 7).",
        );
      }
    } else if (!model.available) {
      renderEmptyVisual(
        "The paper's shallow estimator is only wired up where the demo has a known r_t value.",
        "Try t = 2 through t = 5 for the estimator, or switch to the Circuit view to build the explicit Theorem 5.6 circuit.",
      );
    } else {
      renderShallowSchematic(model);
    }
    prependViewToggle(
      [
        { id: "schematic", label: "Schematic" },
        { id: "circuit", label: "Circuit" },
      ],
      state.shallowView,
      (view) => {
        state.shallowView = view;
        render();
      },
    );
    return;
  }

  if (model.id === "flagAtOrigin") {
    const metric = model.actual;
    const hasCircuit = Boolean(metric && metric.layers);
    if (!hasCircuit) {
      renderEmptyVisual(
        `No bundled flag-at-origin circuit is available for n = ${state.n}, t = ${state.t}.`,
        "These baseline panels use the exact repo circuits when they exist.",
      );
    } else if (state.flagView === "schedule") {
      renderSchedule(
        metric,
        state.n,
        model.accent,
        `Flag-at-origin CNOT layers for n = ${state.n}, t = ${state.t}: ${metric.numCx} CNOTs across ${metric.layers.length} entangling layers from the bundled QASM. Switch to the Circuit view to drag overlapping gates apart and export the Stim source.`,
        model.id,
      );
    } else {
      renderFlagCircuit(model);
      appendFlagExport(metric);
    }
    prependViewToggle(
      [
        { id: "circuit", label: "Circuit" },
        { id: "schedule", label: "Schedule" },
      ],
      state.flagView,
      (view) => {
        state.flagView = view;
        render();
      },
    );
    return;
  }

  if (model.id === "mqt") {
    const metric = model.actual;
    const hasCircuit = Boolean(metric && metric.layers);
    if (!hasCircuit) {
      renderEmptyVisual(
        `No bundled MQT circuit is available for n = ${state.n}, t = ${state.t}.`,
        "These baseline panels use the exact repo circuits when they exist.",
      );
    } else if (state.mqtView === "schedule") {
      renderSchedule(
        metric,
        state.n,
        model.accent,
        `MQT CNOT layers for n = ${state.n}, t = ${state.t}: ${metric.numCx} CNOTs across ${metric.layers.length} entangling layers from the bundled Stim circuit. Switch to the Circuit view to drag overlapping gates apart and export the Stim source.`,
        model.id,
      );
    } else {
      renderMqtCircuit(model);
      appendMqtExport(model);
    }
    prependViewToggle(
      [
        { id: "circuit", label: "Circuit" },
        { id: "schedule", label: "Schedule" },
      ],
      state.mqtView,
      (view) => {
        state.mqtView = view;
        render();
      },
    );
    return;
  }

  if (!model.available || !model.actual) {
    renderEmptyVisual(
      `No bundled ${model.label} circuit is available for n = ${state.n}, t = ${state.t}.`,
      "These baseline panels use the exact repo circuits when they exist.",
    );
    return;
  }

  renderSchedule(
    model.actual,
    state.n,
    model.accent,
    `Exact clean-circuit CNOT layers for ${model.label} at n = ${state.n}, t = ${state.t}. The plotted timeline only shows entangling layers; the reported depth follows the repo's counting convention with initial and final basis layers included.`,
    model.id,
  );
}

function render() {
  refs.nValue.textContent = `n = ${state.n}`;
  // Preserve the user's requested value on the slider; flag the effective t when
  // the request was reduced into the implied region.
  refs.tValue.textContent = isImpliedT()
    ? `t = ${state.requestedT} → ${state.t}`
    : `t = ${state.requestedT}`;

  const models = data.methods.order.map(buildMethodModel);
  buildHighlights(models);
  renderSummary(models);
  renderCards(models);

  const selected = models.find((model) => model.id === state.selectedMethod) || models[0];
  renderDetail(selected);

  renderMethodComparison();
}

syncEffectiveT();
render();
