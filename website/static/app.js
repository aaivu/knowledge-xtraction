/* ══════════════════════════════════════════════════════════════════════════
   S3KG UI — Frontend Logic
   ══════════════════════════════════════════════════════════════════════════ */

'use strict';

// ── Application State ────────────────────────────────────────────────────────
const state = {
  kgs: { 1: [], 2: [], 3: [] },
  networks: {},
  analysis: null,       // stored after s3kg call for KGAnalytica
  currentTexts: null,   // stored after generation for cache save
};

// ── DOM References ───────────────────────────────────────────────────────────
const settingsToggle  = document.getElementById('settingsToggle');
const settingsPanel   = document.getElementById('settingsPanel');
const alphaInput      = document.getElementById('alpha');
const alphaVal        = document.getElementById('alphaVal');
const generateBtn     = document.getElementById('generateBtn');
const s3kgBtn         = document.getElementById('s3kgBtn');
const kgaBtn          = document.getElementById('kgaBtn');
const kgSection       = document.getElementById('kgSection');
const resultsSection  = document.getElementById('resultsSection');
const kgaSection      = document.getElementById('kgaSection');
const toastContainer  = document.getElementById('toastContainer');
const useCacheCheck   = document.getElementById('useCache');
const cacheActionBtns = document.getElementById('cacheActionBtns');
const saveCacheBtn    = document.getElementById('saveCacheBtn');
const clearCacheBtn   = document.getElementById('clearCacheBtn');

// ── Settings Panel Toggle ─────────────────────────────────────────────────────
settingsToggle.addEventListener('click', () => {
  const isOpen = settingsPanel.classList.toggle('open');
  settingsToggle.classList.toggle('active', isOpen);
});

// ── Alpha Slider ──────────────────────────────────────────────────────────────
function updateSliderGradient(val) {
  const pct = val * 100;
  alphaInput.style.background =
    `linear-gradient(to right, var(--accent) 0%, var(--accent) ${pct}%, rgba(255,255,255,0.1) ${pct}%, rgba(255,255,255,0.1) 100%)`;
}

alphaInput.addEventListener('input', () => {
  const v = parseFloat(alphaInput.value);
  alphaVal.textContent = v.toFixed(2);
  updateSliderGradient(v);
});

// Initialise slider gradient on load
updateSliderGradient(0.5);

// ── Toast Notifications ───────────────────────────────────────────────────────
const TOAST_ICONS = {
  error: `<svg class="toast-icon" viewBox="0 0 20 20" fill="currentColor">
    <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/>
  </svg>`,
  success: `<svg class="toast-icon" viewBox="0 0 20 20" fill="currentColor">
    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l3-3z" clip-rule="evenodd"/>
  </svg>`,
};

function toast(message, type = 'error') {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.innerHTML = `${TOAST_ICONS[type] || TOAST_ICONS.error}
    <span class="toast-msg">${message}</span>`;
  toastContainer.appendChild(el);
  setTimeout(() => el.remove(), 4300);
}

// ── Button Loading State ──────────────────────────────────────────────────────
function setLoading(btn, loading) {
  const label   = btn.querySelector('.btn-label');
  const spinner = btn.querySelector('.spinner');
  btn.disabled  = loading;
  if (label)   label.style.opacity = loading ? '0' : '1';
  if (spinner) spinner.hidden = !loading;
}

// ── Truncate Long Labels ──────────────────────────────────────────────────────
function trunc(str, n = 22) {
  return str.length > n ? str.slice(0, n - 1) + '…' : str;
}

// ── Graph color constants (entities = blue, relations = orange) ───────────────
const NODE_COLOR = { bg: '#dbeafe', border: '#3b82f6', text: '#1e3a8a',
                     hlBg: '#bfdbfe', hlBorder: '#1d4ed8' };
const EDGE_COLOR = { line: '#f97316', label: '#c2410c' };

// ── vis.js Options ────────────────────────────────────────────────────────────
const VIS_OPTIONS = {
  physics: {
    enabled: true,
    solver: 'forceAtlas2Based',
    forceAtlas2Based: {
      gravitationalConstant: -45,
      centralGravity: 0.015,
      springLength: 95,
      springConstant: 0.08,
      damping: 0.4,
    },
    stabilization: { iterations: 250, updateInterval: 25 },
  },
  nodes: {
    shape: 'box',
    borderWidth: 1,
    borderWidthSelected: 2,
    margin: { top: 5, right: 9, bottom: 5, left: 9 },
    font: { color: '#1e3a8a', size: 12, face: 'Inter, system-ui, sans-serif' },
    color: {
      background: '#dbeafe',
      border: '#3b82f6',
      highlight: { background: '#bfdbfe', border: '#1d4ed8' },
      hover:     { background: '#eff6ff', border: '#3b82f6' },
    },
    shadow: { enabled: true, color: 'rgba(59,130,246,0.15)', x: 0, y: 2, size: 6 },
  },
  edges: {
    arrows: { to: { enabled: true, scaleFactor: 0.6 } },
    color: { color: '#f97316', highlight: '#ea580c', hover: '#ea580c', inherit: false },
    font: {
      color: '#c2410c',
      size: 10,
      face: 'JetBrains Mono, Fira Code, monospace',
      align: 'middle',
      strokeWidth: 2,
      strokeColor: '#ffffff',
      background: 'rgba(255,255,255,0.92)',
    },
    smooth: { type: 'curvedCW', roundness: 0.15 },
    width: 1.5,
    selectionWidth: 2.5,
    hoverWidth: 2,
  },
  interaction: {
    hover: true,
    tooltipDelay: 150,
    navigationButtons: false,
    keyboard: true,
    zoomView: true,
    dragView: true,
  },
  layout: { improvedLayout: true },
};

// ── Render vis.js Graph ───────────────────────────────────────────────────────
function renderGraph(containerId, triples) {
  const nodeMap = new Map();
  const edgeArr = [];

  triples.forEach(([s, r, o], i) => {
    [s, o].forEach((label) => {
      if (!nodeMap.has(label)) {
        nodeMap.set(label, { id: label, label: trunc(label, 24), title: label });
      }
    });
    edgeArr.push({ id: i, from: s, to: o, label: trunc(r, 28), title: r });
  });

  const nodes = new vis.DataSet([...nodeMap.values()]);
  const edges = new vis.DataSet(edgeArr);
  const container = document.getElementById(containerId);
  container.innerHTML = '';

  const network = new vis.Network(container, { nodes, edges }, VIS_OPTIONS);

  network.once('stabilizationIterationsDone', () => {
    network.fit({ animation: { duration: 700, easingFunction: 'easeInOutQuad' } });
  });

  return network;
}

// ── Build Triplet Table ───────────────────────────────────────────────────────
function buildTripletTable(tableId, triples) {
  const tbody = document.querySelector(`#${tableId} tbody`);
  tbody.innerHTML = triples.map(([s, r, o]) =>
    `<tr>
      <td title="${escapeHtml(s)}">${escapeHtml(trunc(s, 30))}</td>
      <td title="${escapeHtml(r)}">${escapeHtml(trunc(r, 30))}</td>
      <td title="${escapeHtml(o)}">${escapeHtml(trunc(o, 30))}</td>
    </tr>`
  ).join('');
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Gauge Animation ───────────────────────────────────────────────────────────
// SVG circle r=42 → circumference = 2π×42 ≈ 263.9, rounded to 264
const CIRCUMFERENCE = 264;

function scoreColor(v) {
  if (v >= 0.70) return '#22d3a5';   // green
  if (v >= 0.50) return '#f59e0b';   // amber
  return '#ef4444';                   // red
}

function animateGauge(key, value) {
  const circle = document.getElementById(`gauge-${key}`);
  const valEl  = document.getElementById(`val-${key}`);
  const color  = scoreColor(value);

  circle.style.stroke            = color;
  circle.style.strokeDashoffset  = CIRCUMFERENCE * (1 - value);
  valEl.style.color              = color;

  // Animate the numeric counter
  const duration = 1300;
  let start = null;
  function step(ts) {
    if (!start) start = ts;
    const progress = Math.min((ts - start) / duration, 1);
    const eased    = 1 - Math.pow(1 - progress, 3);   // ease-out cubic
    valEl.textContent = (eased * value).toFixed(4);
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ── Show Section with Fade ────────────────────────────────────────────────────
function showSection(el) {
  el.hidden = false;
  el.classList.remove('fade-in');
  // Force reflow so animation re-triggers
  void el.offsetWidth;
  el.classList.add('fade-in');
}

// ── KG Cache ─────────────────────────────────────────────────────────────────
const KG_CACHE_KEY = 'kgx_kg_cache';

function normalizeInput(s) { return s.trim().replace(/\s+/g, ' '); }

function loadFromCache(texts) {
  try {
    const entry = JSON.parse(localStorage.getItem(KG_CACHE_KEY));
    if (!entry || !Array.isArray(entry.inputs) || entry.inputs.length !== 3) return null;
    return texts.every((t, i) => normalizeInput(t) === entry.inputs[i]) ? entry.kgs : null;
  } catch (_) { return null; }
}

function saveToCache(texts, kgs) {
  localStorage.setItem(KG_CACHE_KEY, JSON.stringify({
    inputs: texts.map(normalizeInput), kgs, savedAt: Date.now(),
  }));
}

function clearKGCache() { localStorage.removeItem(KG_CACHE_KEY); }

// ── Shared KG Render ─────────────────────────────────────────────────────────
function applyKGData(kgsObj) {
  [1, 2, 3].forEach(i => { state.kgs[i] = kgsObj[String(i)] || []; });
  Object.values(state.networks).forEach(n => { try { n.destroy(); } catch (_) {} });
  state.networks = {};
  resultsSection.hidden = true;
  showSection(kgSection);
  [1, 2, 3].forEach(i => {
    const triples = state.kgs[i];
    const count   = triples.length;
    document.getElementById(`tc${i}`).textContent = `${count} triplet${count !== 1 ? 's' : ''}`;
    buildTripletTable(`tt${i}`, triples);
    if (count > 0) {
      state.networks[i] = renderGraph(`kg${i}`, triples);
    } else {
      document.getElementById(`kg${i}`).innerHTML =
        '<div class="kg-empty-msg">No triplets extracted for this paragraph</div>';
    }
  });
  cacheActionBtns.hidden = false;
  setTimeout(() => kgSection.scrollIntoView({ behavior: 'smooth', block: 'start' }), 120);
}

// ── Generate Knowledge Graphs ─────────────────────────────────────────────────
generateBtn.addEventListener('click', async () => {
  const texts  = [1, 2, 3].map(i => document.getElementById(`text${i}`).value.trim());
  const apiKey = document.getElementById('apiKey').value.trim();
  const model  = document.getElementById('modelName').value.trim() || 'llama-3.1-8b-instant';

  if (texts.some(t => !t)) { toast('Please fill in all three text areas before generating.'); return; }
  if (!apiKey) { toast('Open ⚙ Settings and enter your Groq API key.'); return; }

  // Check cache first if toggle is on
  if (useCacheCheck.checked) {
    const cached = loadFromCache(texts);
    if (cached) {
      state.currentTexts = texts;
      kgSection.hidden = true;
      resultsSection.hidden = true;
      setLoading(generateBtn, true);
      await new Promise(r => setTimeout(r, 2000));
      setLoading(generateBtn, false);
      applyKGData(cached);
      return;
    }
  }

  setLoading(generateBtn, true);
  kgSection.hidden      = true;
  resultsSection.hidden = true;

  try {
    const res  = await fetch('/api/generate-kgs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts, model, api_key: apiKey }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Server error');
    state.currentTexts = texts;
    applyKGData(data.kgs);
    toast('Knowledge graphs generated successfully!', 'success');
  } catch (err) {
    toast(err.message);
  } finally {
    setLoading(generateBtn, false);
  }
});

// ── Cache Save / Clear ────────────────────────────────────────────────────────
saveCacheBtn.addEventListener('click', () => {
  if (!state.currentTexts || [1, 2, 3].every(i => !state.kgs[i].length)) {
    toast('No KGs to save.'); return;
  }
  saveToCache(state.currentTexts, { '1': state.kgs[1], '2': state.kgs[2], '3': state.kgs[3] });
  saveCacheBtn.classList.add('saved');
  setTimeout(() => saveCacheBtn.classList.remove('saved'), 1500);
});

clearCacheBtn.addEventListener('click', () => {
  clearKGCache();
  clearCacheBtn.classList.add('cleared');
  setTimeout(() => clearCacheBtn.classList.remove('cleared'), 1500);
});

// ── Calculate S3KG Similarity ─────────────────────────────────────────────────
s3kgBtn.addEventListener('click', async () => {
  const alpha = parseFloat(alphaInput.value);

  if ([1, 2, 3].some(i => state.kgs[i].length === 0)) {
    toast('One or more KGs are empty. Re-generate graphs first.');
    return;
  }

  setLoading(s3kgBtn, true);
  try {
    const res  = await fetch('/api/s3kg', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        kg1: state.kgs[1],
        kg2: state.kgs[2],
        kg3: state.kgs[3],
        alpha,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Server error');

    const { gold_sim, ctx_sim, cus } = data;

    // Save analysis details for KGAnalytica
    state.analysis = data.analysis || null;

    // Show and animate results
    showSection(resultsSection);
    animateGauge('gold', gold_sim);
    animateGauge('ctx',  ctx_sim);
    animateGauge('cus',  cus);

    // Insight text
    const level  = cus >= 0.7 ? 'strong' : cus >= 0.5 ? 'moderate' : 'weak';
    const alphaDesc = alpha <= 0.25 ? 'WL-kernel-dominant'
                    : alpha >= 0.75 ? 'SBERT-dominant'
                    : 'balanced structural–semantic blend';
    const skew   = gold_sim > ctx_sim
      ? 'GoldSim exceeds CtxSim — the response is closer to the reference answer than to the source context.'
      : ctx_sim > gold_sim
      ? 'CtxSim exceeds GoldSim — the response draws more heavily from context than from the reference answer.'
      : 'GoldSim and CtxSim are balanced.';

    document.getElementById('insightBox').innerHTML = `
      <strong>Insight:</strong>
      CUS = <strong>${cus.toFixed(4)}</strong> indicates <strong>${level}</strong> contextual understanding.
      GoldSim (<strong>${gold_sim.toFixed(4)}</strong>) measures factual accuracy against the gold answer;
      CtxSim (<strong>${ctx_sim.toFixed(4)}</strong>) measures faithfulness to the supporting context.
      ${skew}
      Computed with α = <strong>${alpha.toFixed(2)}</strong> (${alphaDesc}).
    `;

    setTimeout(() =>
      resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' }), 120);
    toast('S3KG similarity computed!', 'success');

  } catch (err) {
    toast(err.message);
  } finally {
    setLoading(s3kgBtn, false);
  }
});

// ── KGA Tab Switcher ──────────────────────────────────────────────────────────
document.querySelectorAll('.kga-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.kga-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    document.querySelectorAll('.kga-panel').forEach(p => { p.hidden = true; });
    document.getElementById(tab.dataset.panel).hidden = false;
  });
});

// ── Knowledge Graph Analytica ─────────────────────────────────────────────────
kgaBtn.addEventListener('click', async () => {
  const apiKey = document.getElementById('apiKey').value.trim();
  const model  = document.getElementById('modelName').value.trim() || 'llama-3.1-8b-instant';

  if (!apiKey) {
    toast('Open ⚙ Settings and enter your Groq API key.');
    return;
  }
  if (!state.analysis) {
    toast('Run S3KG Similarity first.');
    return;
  }

  setLoading(kgaBtn, true);
  try {
    const res  = await fetch('/api/kga', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        kg1: state.kgs[1],
        kg2: state.kgs[2],
        kg3: state.kgs[3],
        gold_analysis: state.analysis.gold,
        ctx_analysis:  state.analysis.ctx,
        api_key: apiKey,
        model,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Server error');

    renderKGA('Gold', data.gold, 'Gold Answer');
    renderKGA('Ctx',  data.ctx,  'Supporting Context');

    showSection(kgaSection);
    // Reset to Gold tab
    document.querySelectorAll('.kga-tab').forEach((t, i) => t.classList.toggle('active', i === 0));
    document.getElementById('kgaGoldPanel').hidden = false;
    document.getElementById('kgaCtxPanel').hidden  = true;

    setTimeout(() =>
      kgaSection.scrollIntoView({ behavior: 'smooth', block: 'start' }), 120);
    toast('KGAnalytica analysis complete!', 'success');

  } catch (err) {
    toast(err.message);
  } finally {
    setLoading(kgaBtn, false);
  }
});

// ── Render Knowledge Graph Analytica Results ──────────────────────────────────
function renderKGA(key, data, refLabel) {
  const descs = data.descriptions || {};

  // Overall assessment box
  const overallEl = document.getElementById(`kgaOverall${key}`);
  if (descs.overall) {
    overallEl.innerHTML =
      `<strong>Overall Assessment:</strong> ${escapeHtml(descs.overall)}`;
    overallEl.hidden = false;
  } else {
    overallEl.hidden = true;
  }

  const categories = [
    {
      cls:  'aligned',
      icon: '&#10003;',
      name: 'Aligned Triplets',
      desc: descs.aligned_desc,
      items: data.aligned || [],
      type: 'pair',
      hint: `Facts in KG<sub>LLM</sub> that correctly match ${refLabel}`,
    },
    {
      cls:  'entity-wrong',
      icon: '&#9888;',
      name: 'Entity Wrong',
      desc: descs.entity_wrong_desc,
      items: data.entity_wrong || [],
      type: 'pair',
      hint: 'Triplets where entity names differ from the reference',
    },
    {
      cls:  'rel-wrong',
      icon: '&#9888;',
      name: 'Relation Wrong',
      desc: descs.relation_wrong_desc,
      items: data.relation_wrong || [],
      type: 'pair',
      hint: 'Triplets where the relation predicate differs from the reference',
    },
    {
      cls:  'extra',
      icon: '&#43;',
      name: 'Extra Triplets',
      desc: descs.extra_desc,
      items: data.extra || [],
      type: 'single',
      hint: `In KG<sub>LLM</sub> but absent from ${refLabel} — may be hallucinated`,
    },
    {
      cls:  'missing',
      icon: '&#10007;',
      name: 'Missing Triplets',
      desc: descs.missing_desc,
      items: data.missing || [],
      type: 'missing',
      hint: `In ${refLabel} but not captured by KG<sub>LLM</sub>`,
    },
  ];

  document.getElementById(`kgaCatGrid${key}`).innerHTML = categories.map(cat => {
    const count = cat.items.length;

    // Build table rows based on type
    let rows = '';
    let thead = '';

    if (cat.type === 'pair') {
      thead = '<tr><th>KG<sub>LLM</sub></th><th></th><th>Reference</th><th>Sim</th></tr>';
      rows  = cat.items.map(t => `
        <tr>
          <td class="kga-tri">[${escapeHtml(t.kg1[0])}, <em>${escapeHtml(t.kg1[1])}</em>, ${escapeHtml(t.kg1[2])}]</td>
          <td class="kga-arrow">&harr;</td>
          <td class="kga-tri">[${escapeHtml(t.kg2[0])}, <em>${escapeHtml(t.kg2[1])}</em>, ${escapeHtml(t.kg2[2])}]</td>
          <td><span class="kga-sim">${Math.round(t.sim * 100)}%</span></td>
        </tr>`).join('');
    } else if (cat.type === 'single') {
      thead = '<tr><th>Subject</th><th>Relation</th><th>Object</th></tr>';
      rows  = cat.items.map(t => `
        <tr>
          <td>${escapeHtml(t.triplet[0])}</td>
          <td><em>${escapeHtml(t.triplet[1])}</em></td>
          <td>${escapeHtml(t.triplet[2])}</td>
        </tr>`).join('');
    } else {
      thead = '<tr><th>Subject</th><th>Relation</th><th>Object</th></tr>';
      rows  = cat.items.map(t => `
        <tr>
          <td>${escapeHtml(t[0])}</td>
          <td><em>${escapeHtml(t[1])}</em></td>
          <td>${escapeHtml(t[2])}</td>
        </tr>`).join('');
    }

    return `
      <div class="kga-card kga-card-${cat.cls}">
        <div class="kga-card-hd">
          <span class="kga-cat-icon">${cat.icon}</span>
          <div class="kga-cat-title">
            <span class="kga-cat-name">${cat.name}</span>
            <span class="kga-cat-hint">${cat.hint}</span>
          </div>
          <span class="kga-count">${count}</span>
        </div>
        ${cat.desc
          ? `<p class="kga-desc">${escapeHtml(cat.desc)}</p>`
          : `<p class="kga-desc kga-no-desc">No description available.</p>`}
        ${count > 0 ? `
        <details class="kga-details">
          <summary>View ${count} triplet${count !== 1 ? 's' : ''}</summary>
          <div class="kga-scroll">
            <table class="kga-table"><thead>${thead}</thead><tbody>${rows}</tbody></table>
          </div>
        </details>` : ''}
      </div>`;
  }).join('');
}
