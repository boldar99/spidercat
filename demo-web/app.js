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
  baseline: "Repo baseline",
};

const state = {
  n: data.controls.defaultN,
  t: data.controls.defaultT,
  selectedMethod: "spidercat",
  graphPositionOverrides: {},
  zoomScales: {},
};

const refs = {
  heroGraphCount: document.getElementById("heroGraphCount"),
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
};

refs.heroGraphCount.textContent = Object.keys(data.spiderGraphs).length;

refs.nRange.value = String(state.n);
refs.tRange.value = String(state.t);

refs.nRange.addEventListener("input", (event) => {
  state.n = Number(event.target.value);
  render();
});

refs.tRange.addEventListener("input", (event) => {
  state.t = Number(event.target.value);
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

function getActualMetric(methodId, n, t) {
  return data.actualMetrics[methodId]?.[keyOf(n, t)] || null;
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
  const available = models.filter((model) => model.available && model.metrics);
  if (!available.length) {
    refs.stateSummary.textContent = `No methods are available at n = ${state.n}, t = ${state.t}.`;
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
    `At n = ${state.n}, t = ${state.t}, ${bestCx.label} is the cheapest in CNOT count while ${bestDepth.label} is the shallowest construction.${extra}`;
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
      </div>
    </article>
  `;
}

function renderCards(models) {
  refs.methodCards.innerHTML = models.map(cardHtml).join("");
  refs.methodCards.querySelectorAll(".method-card").forEach((card) => {
    card.addEventListener("click", () => {
      state.selectedMethod = card.dataset.method;
      render();
    });
  });
}

function clearVisual() {
  refs.visualLegend.innerHTML = "";
  refs.visualHost.innerHTML = "";
  refs.visualCaption.textContent = "";
}

function legendPills(items) {
  refs.visualLegend.innerHTML = items
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
  stage.appendChild(svg);
  viewport.appendChild(stage);

  function applyZoom() {
    const scale = clampNumber(state.zoomScales[zoomKey], minScale, maxScale);
    state.zoomScales[zoomKey] = scale;
    stage.style.width = `${scale * 100}%`;
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
  refs.visualHost.appendChild(figure);
}

function renderSpiderGraph(model) {
  clearVisual();
  const graphBundle = model.spiderGraph;
  if (!graphBundle) {
    refs.visualHost.innerHTML = `<div class="empty-state">No SpiderCat graph is bundled for t = ${state.t}.</div>`;
    refs.visualCaption.textContent = "The graph explorer only covers SpiderCat instances saved in spidercat/circuits_data.";
    return;
  }

  legendPills([
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
        fill: "#fffef8",
        stroke: "var(--ink)",
        "stroke-width": 4,
      }),
    );

    let label = null;
    if (showLabels) {
      label = svgNode("text", {
        "text-anchor": "middle",
        "font-size": labelFontSize,
        "font-weight": 600,
        fill: "var(--muted)",
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

function renderSchedule(metric, dataQubits, accentColor, caption, zoomKey = "schedule") {
  clearVisual();
  legendPills([
    { color: "var(--data-wire)", label: "data qubit" },
    { color: "var(--flag-wire)", label: "flag / ancilla" },
    { color: accentColor, label: "CNOT layer" },
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
      const usesFlag = control >= dataQubits || target >= dataQubits;
      svg.appendChild(
        svgNode("line", {
          x1: x,
          y1,
          x2: x,
          y2,
          stroke: usesFlag ? "var(--flag-wire)" : accentColor,
          "stroke-width": 2.6,
          "stroke-linecap": "round",
        }),
      );
      svg.appendChild(
        svgNode("circle", {
          cx: x,
          cy: y1,
          r: 4.8,
          fill: usesFlag ? "var(--flag-wire)" : accentColor,
          stroke: "#fff",
          "stroke-width": 1.4,
        }),
      );
      svg.appendChild(
        svgNode("circle", {
          cx: x,
          cy: y2,
          r: 4.8,
          fill: "#fffef8",
          stroke: usesFlag ? "var(--flag-wire)" : accentColor,
          "stroke-width": 2,
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

function renderRecursiveSchematic(model) {
  clearVisual();
  legendPills([
    { color: "var(--recursive)", label: "recursive fusion step" },
    { color: "rgba(20, 33, 61, 0.16)", label: "smaller CAT block" },
  ]);

  const levels = Math.max(1, Math.ceil(Math.log2(state.t + 1)));
  const width = 980;
  const height = 220 + levels * 130;
  const laneLeft = 192;
  const laneRight = 64;
  const usableWidth = width - laneLeft - laneRight;
  const topPad = 92;
  const laneGap = 128;
  const boxHeight = 58;
  const svg = svgNode("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": "Recursive CAT state fusion schematic",
  });

  function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }

  function stageTitle(level) {
    if (level === 0) {
      return "Seed blocks";
    }
    if (level === levels) {
      return "Output";
    }
    return `Fusion round ${level}`;
  }

  function stageSubtitle(level, count) {
    if (level === 0) {
      return `${count} smaller CAT blocks`;
    }
    if (level === levels) {
      return "final fault-tolerant CAT state";
    }
    return `${count} merged CAT blocks`;
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
      const span = svgNode("tspan", {
        x,
        dy: index === 0 ? 0 : 16,
      });
      span.textContent = line;
      text.appendChild(span);
    });
    svg.appendChild(text);
  }

  const positionsByLevel = [];
  for (let level = 0; level <= levels; level += 1) {
    const count = 2 ** (levels - level);
    const y = topPad + level * laneGap;
    const xs = [];
    for (let index = 0; index < count; index += 1) {
      const x = laneLeft + (usableWidth * (index + 0.5)) / count;
      xs.push(x);
    }
    positionsByLevel.push({ count, y, xs });
  }

  positionsByLevel.forEach((levelData, level) => {
    const lane = svgNode("rect", {
      x: 18,
      y: levelData.y - 48,
      width: width - 36,
      height: 88,
      rx: 26,
      fill: level === levels ? "rgba(217, 93, 57, 0.08)" : "rgba(255, 255, 255, 0.32)",
      stroke: level === levels ? "rgba(217, 93, 57, 0.18)" : "rgba(20, 33, 61, 0.06)",
      "stroke-width": 1.2,
    });
    svg.appendChild(lane);

    const title = svgNode("text", {
      x: 42,
      y: levelData.y - 8,
      "font-size": 12,
      "font-weight": 700,
      fill: "var(--ink)",
    });
    title.textContent = stageTitle(level);
    svg.appendChild(title);

    const subtitle = svgNode("text", {
      x: 42,
      y: levelData.y + 12,
      "font-size": 11,
      fill: "var(--muted)",
    });
    subtitle.textContent = stageSubtitle(level, levelData.count);
    svg.appendChild(subtitle);
  });

  for (let level = 0; level < levels; level += 1) {
    const current = positionsByLevel[level];
    const next = positionsByLevel[level + 1];
    for (let index = 0; index < current.xs.length; index += 2) {
      const left = current.xs[index];
      const right = current.xs[index + 1];
      const parent = next.xs[index / 2];
      const childBottom = current.y + boxHeight / 2;
      const parentTop = next.y - boxHeight / 2;
      const junctionY = current.y + 52;
      const labelWidth = 94;
      const labelCenterY = junctionY + 18;
      svg.appendChild(
        svgNode("path", {
          d: `M ${left} ${childBottom} Q ${left} ${junctionY - 12}, ${parent} ${junctionY}`,
          fill: "none",
          stroke: "var(--recursive)",
          "stroke-width": 3.5,
          "stroke-linecap": "round",
        }),
      );
      svg.appendChild(
        svgNode("path", {
          d: `M ${right} ${childBottom} Q ${right} ${junctionY - 12}, ${parent} ${junctionY}`,
          fill: "none",
          stroke: "var(--recursive)",
          "stroke-width": 3.5,
          "stroke-linecap": "round",
        }),
      );
      svg.appendChild(
        svgNode("line", {
          x1: parent,
          y1: junctionY,
          x2: parent,
          y2: parentTop,
          stroke: "var(--recursive)",
          "stroke-width": 3.5,
          "stroke-linecap": "round",
        }),
      );
      svg.appendChild(
        svgNode("circle", {
          cx: parent,
          cy: junctionY,
          r: 4.5,
          fill: "var(--recursive)",
          stroke: "#fff",
          "stroke-width": 1.4,
        }),
      );
      svg.appendChild(
        svgNode("rect", {
          x: parent - labelWidth / 2,
          y: labelCenterY - 11,
          width: labelWidth,
          height: 22,
          rx: 11,
          fill: "rgba(255, 250, 241, 0.96)",
          stroke: "rgba(217, 93, 57, 0.2)",
          "stroke-width": 1,
        }),
      );

      const label = svgNode("text", {
        x: parent,
        y: labelCenterY + 4,
        "font-size": 11.5,
        "font-weight": 700,
        "text-anchor": "middle",
        fill: "var(--recursive)",
      });
      label.textContent = `${state.t + 1} ZZ checks`;
      svg.appendChild(label);
    }
  }

  positionsByLevel.forEach((levelData, level) => {
    levelData.xs.forEach((x) => {
      const widthBox =
        level === levels
          ? 340
          : clamp(usableWidth / Math.max(1, levelData.xs.length) - 22, 96, 150);
      const rect = svgNode("rect", {
        x: x - widthBox / 2,
        y: levelData.y - boxHeight / 2,
        width: widthBox,
        height: boxHeight,
        rx: 22,
        fill: level === levels ? "rgba(217, 93, 57, 0.16)" : "rgba(20, 33, 61, 0.06)",
        stroke: level === levels ? "var(--recursive)" : "rgba(20, 33, 61, 0.18)",
        "stroke-width": level === levels ? 2.6 : 1.5,
      });
      svg.appendChild(rect);

      appendBoxLabel(
        x,
        levelData.y + 2,
        level === levels ? ["final", `CAT_${state.n}`] : ["CAT", "sub-block"],
        level === levels,
      );
    });
  });

  refs.visualHost.appendChild(svg);
  refs.visualCaption.textContent =
    `Concept sketch for the recursive paper construction. Pairwise fusions repeat for ${levels} recursive round${levels === 1 ? "" : "s"}, and each fusion uses ${state.t + 1} transversal ZZ checks.`;
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
          fill: "#fffef8",
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
    rangeNote = `Graph view is snapped to the nearest available SpiderCat instance at n = ${model.spiderGraph.targetN}.`;
  }

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
  `;
}

function renderDetail(model) {
  refs.detailTitle.textContent = model.label;
  refs.detailSubtitle.textContent = `${model.kindLabel}. ${model.optimize}. ${model.paperHook}.`;
  renderDetailInfo(model);

  if (model.id === "spidercat") {
    renderSpiderGraph(model);
    return;
  }

  if (model.id === "recursive") {
    renderRecursiveSchematic(model);
    return;
  }

  if (model.id === "shallow") {
    if (!model.available) {
      renderEmptyVisual(
        "The paper's shallow estimator is only wired up where the demo has a known r_t value.",
        "Try t = 2 through t = 5 to activate the shallow estimator.",
      );
      return;
    }
    renderShallowSchematic(model);
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
  refs.tValue.textContent = `t = ${state.t}`;

  const models = data.methods.order.map(buildMethodModel);
  buildHighlights(models);
  renderSummary(models);
  renderCards(models);

  const selected = models.find((model) => model.id === state.selectedMethod) || models[0];
  renderDetail(selected);
}

render();
