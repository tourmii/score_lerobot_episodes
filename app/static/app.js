/* Episode-quality frontend.
 *
 * No framework and no build step on purpose: the app is one page with two
 * views, and the interesting code is the measurement pipeline behind it, not
 * the shell around it.
 *
 * Layout follows huggingface/lerobot-dataset-visualizer: dataset stats and the
 * episode list live in a fixed rail, the cameras play in a row above the
 * signals, and one floating transport bar drives all of them at once.
 *
 * Routes are hash-based:
 *   #/                       pick or add a dataset
 *   #/d/<id>                 overview: distributions and the episode table
 *   #/d/<id>/e/<episode>     one episode: cameras beside their signals
 */

const state = {
  datasets: [],
  dataset: null,      // active dataset summary
  overview: null,
  rows: [],
  episode: null,      // active episode index
  sort: { key: 'total', asc: true },
  filter: { decision: new Set(), text: '' },
  railFilter: 'all',
  chart: null,
  players: [],        // <video> elements, players[0] is the clock
  poll: null,
  raf: null,
};

const $ = sel => document.querySelector(sel);
const view = $('#view');

/* ------------------------------------------------------------------ api */

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: options.body ? { 'Content-Type': 'application/json' } : {},
    ...options,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch (_) { /* text body */ }
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}

/* -------------------------------------------------------------- helpers */

const esc = t => String(t ?? '').replace(/[&<>"']/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const num = (v, digits = 3) =>
  v === null || v === undefined || Number.isNaN(v) ? '–' : Number(v).toFixed(digits);

const group = n => (n === null || n === undefined) ? '–' : Number(n).toLocaleString('en-US');

function bar(value) {
  const pct = value == null ? 0 : Math.max(0, Math.min(1, value)) * 100;
  return `<span class="cellbar"><i style="width:${pct.toFixed(1)}%"></i></span>`;
}

function barClass(value) {
  if (value == null) return '';
  const floor = state.overview ? state.overview.threshold : 0.35;
  if (value < floor) return 'low';
  if (value < floor + 0.15) return 'mid';
  return '';
}

function el(html) {
  const template = document.createElement('template');
  template.innerHTML = html.trim();
  return template.content.firstElementChild;
}

/* ---------------------------------------------------------------- rail */

async function loadDatasets(selectId) {
  state.datasets = await api('/api/datasets');
  renderPicker();
  if (selectId) location.hash = `#/d/${selectId}`;
}

function renderPicker() {
  const picker = $('#datasetPicker');
  picker.innerHTML = state.datasets.length
    ? state.datasets.map(d =>
        `<option value="${d.id}">${esc(d.name)}${d.analyzed ? '' : ' · not measured'}</option>`).join('')
    : '<option value="">no dataset yet</option>';
  if (state.dataset) picker.value = state.dataset.id;
}

$('#datasetPicker').addEventListener('change', ev => {
  if (ev.target.value) location.hash = `#/d/${ev.target.value}`;
});

function renderStats() {
  const stats = $('#stats');
  const d = state.dataset;
  if (!d) { stats.innerHTML = ''; return; }
  const rows = [
    ['frames', group(d.frames || null)],
    ['episodes', group(d.episodes)],
    ['fps', d.fps ? Math.round(d.fps) : '–'],
  ];
  if (d.cameras && d.cameras.length) rows.push(['cameras', d.cameras.length]);
  stats.innerHTML = rows.map(([k, v]) =>
    `<div><dt>${k}</dt><dd>${v}</dd></div>`).join('');
}

const RAIL_FILTERS = ['all', 'accept', 'review', 'reject'];

function renderEpisodeRail() {
  const rail = $('#episodeRail');
  const list = $('#episodeList');
  if (!state.dataset || !state.dataset.analyzed || !state.rows.length) {
    rail.hidden = true;
    return;
  }
  rail.hidden = false;

  const filters = $('#epFilter');
  filters.innerHTML = '';
  for (const name of RAIL_FILTERS) {
    const count = name === 'all'
      ? state.rows.length
      : state.rows.filter(r => r.decision === name).length;
    if (name !== 'all' && !count) continue;
    const chip = el(`<button class="${state.railFilter === name ? 'on' : ''}">${name} ${count}</button>`);
    chip.addEventListener('click', () => { state.railFilter = name; renderEpisodeRail(); });
    filters.appendChild(chip);
  }

  const rows = state.railFilter === 'all'
    ? state.rows
    : state.rows.filter(r => r.decision === state.railFilter);

  list.innerHTML = rows.map(r => {
    const active = r.episode === state.episode;
    return `<li><a href="#/d/${state.dataset.id}/e/${r.episode}"
      class="${active ? 'active' : ''}" data-episode="${r.episode}">
      <span class="dot ${r.decision}" title="${r.decision}"></span>
      <span>Episode ${r.episode}</span>
      <span class="score mono">${num(r.total, 2)}</span></a></li>`;
  }).join('');

  const active = list.querySelector('a.active');
  if (active) active.scrollIntoView({ block: 'nearest' });
}

$('#addForm').addEventListener('submit', async ev => {
  ev.preventDefault();
  const path = $('#addPath').value.trim();
  $('#addError').textContent = '';
  if (!path) return;
  try {
    const ds = await api('/api/datasets', { method: 'POST', body: { path } });
    $('#addPath').value = '';
    await loadDatasets(ds.id);
  } catch (err) {
    $('#addError').innerHTML = `<span class="err">${esc(err.message)}</span>`;
  }
});

function railForDataset(ds) {
  const analysed = ds && ds.analyzed;
  $('#analyzeBlock').hidden = !ds;
  $('#scoreBlock').hidden = !analysed;
  $('#semanticBlock').hidden = !ds;
  $('#exportBlock').hidden = !analysed;
  $('#settingsHint').textContent = ds
    ? (analysed ? 'measured' : 'needs measuring')
    : 'add a dataset';
  if (!analysed) $('#settings').open = true;
  if (!ds) return;
  $('#exportCsv').href = `/api/datasets/${ds.id}/export.csv`;
  $('#exportJson').href = `/api/datasets/${ds.id}/export.json?decision=accept`;
  $('#analyzeBtn').textContent = analysed ? 're-run measurement' : 'run measurement';
}

/* --------------------------------------------------------------- jobs */

function trackJob(job, els, onDone) {
  const { block, bar: barEl, msg } = els;
  block.hidden = false;
  clearInterval(state.poll);
  const paint = j => {
    const pct = j.total ? (j.done / j.total) * 100 : (j.state === 'done' ? 100 : 8);
    barEl.style.width = pct.toFixed(1) + '%';
    msg.textContent = j.state === 'error'
      ? j.error
      : `${j.done}/${j.total || '?'} · ${j.message || j.state} · ${j.elapsed}s`;
    msg.classList.toggle('err', j.state === 'error');
  };
  paint(job);
  state.poll = setInterval(async () => {
    let current;
    try { current = await api(`/api/jobs/${job.id}`); }
    catch (err) { clearInterval(state.poll); msg.textContent = err.message; return; }
    paint(current);
    if (current.state !== 'running') {
      clearInterval(state.poll);
      if (current.state === 'done') onDone && onDone(current);
      setTimeout(() => { block.hidden = true; }, 2500);
    }
  }, 700);
}

$('#analyzeBtn').addEventListener('click', async () => {
  if (!state.dataset) return;
  const btn = $('#analyzeBtn');
  btn.disabled = true;
  try {
    const job = await api(`/api/datasets/${state.dataset.id}/analyze`, {
      method: 'POST',
      body: {
        useVideo: $('#useVideo').checked,
        workers: Number($('#workers').value) || 4,
        loPct: Number($('#loPct').value) || 5,
        hiPct: Number($('#hiPct').value) || 95,
      },
    });
    trackJob(job, {
      block: $('#analyzeProgress'),
      bar: $('#analyzeProgress .progress-bar > i'),
      msg: $('#analyzeMsg'),
    }, async () => {
      await loadDatasets();
      route();
    });
  } catch (err) {
    $('#analyzeMsg').innerHTML = `<span class="err">${esc(err.message)}</span>`;
    $('#analyzeProgress').hidden = false;
  } finally {
    btn.disabled = false;
  }
});

$('#semanticBtn').addEventListener('click', async () => {
  if (!state.dataset) return;
  $('#semanticError').textContent = '';
  try {
    const job = await api(`/api/datasets/${state.dataset.id}/semantic`, {
      method: 'POST',
      body: {
        baseUrl: $('#semanticUrl').value.trim() || null,
        model: $('#semanticModel').value.trim() || null,
        workers: 4,
      },
    });
    trackJob(job, {
      block: $('#semanticProgress'),
      bar: $('#semanticProgress .progress-bar > i'),
      msg: $('#semanticMsg'),
    }, () => route());
  } catch (err) {
    $('#semanticError').innerHTML = `<span class="err">${esc(err.message)}</span>`;
  }
});

$('#refitBtn').addEventListener('click', async () => {
  if (!state.dataset) return;
  await api(`/api/datasets/${state.dataset.id}/calibrate`, {
    method: 'POST',
    body: { loPct: Number($('#loPct').value) || 5, hiPct: Number($('#hiPct').value) || 95 },
  });
  await refreshScores();
});

/* ------------------------------------------------------- scoring controls */

let policyTimer = null;

function schedulePolicy() {
  clearTimeout(policyTimer);
  policyTimer = setTimeout(applyPolicy, 180);
}

async function applyPolicy() {
  if (!state.dataset) return;
  const weights = {};
  for (const input of document.querySelectorAll('#weightSliders input[type=range]')) {
    weights[input.dataset.family] = Number(input.value);
  }
  state.overview = await api(`/api/datasets/${state.dataset.id}/policy`, {
    method: 'PUT',
    body: {
      weights, threshold: Number($('#threshold').value),
      mode: $('#mode').value, aggregate: $('#aggregate').value,
    },
  });
  state.rows = await api(`/api/datasets/${state.dataset.id}/episodes`);
  renderModeNote();
  renderEpisodeRail();
  if (currentRoute().view === 'overview') renderOverview();
  else if (state.episode !== null) refreshEpisodeVerdict();
}

async function refreshScores() {
  state.overview = await api(`/api/datasets/${state.dataset.id}/overview`);
  state.rows = await api(`/api/datasets/${state.dataset.id}/episodes`);
  renderControls();
  renderEpisodeRail();
  if (currentRoute().view === 'overview') renderOverview();
}

function renderControls() {
  const box = $('#weightSliders');
  box.innerHTML = '';
  const weights = state.overview.weights || {};
  for (const [family, value] of Object.entries(weights)) {
    const row = el(`<div class="weight ${value ? '' : 'zero'}">
      <label for="w-${family}">${esc(family)}</label>
      <input id="w-${family}" type="range" min="0" max="1" step="0.05"
             value="${value}" data-family="${family}">
      <b>${value.toFixed(2)}</b></div>`);
    const input = row.querySelector('input');
    input.addEventListener('input', () => {
      row.querySelector('b').textContent = Number(input.value).toFixed(2);
      row.classList.toggle('zero', Number(input.value) === 0);
      schedulePolicy();
    });
    box.appendChild(row);
  }
  $('#threshold').value = state.overview.threshold;
  $('#thresholdValue').textContent = Number(state.overview.threshold).toFixed(2);
  $('#mode').value = state.overview.mode || 'gate';
  $('#aggregate').value = state.overview.aggregate || 'geometric';
  renderModeNote();
  const calib = state.overview.calibration;
  $('#calibNote').textContent = calib && calib.n_episodes_fitted
    ? `ranges fitted on ${calib.n_episodes_fitted} valid episodes · `
      + `grasp nominal ${calib.grasp_nominal} · median ${num(calib.duration_median, 2)} s`
    : 'using the shipped reference ranges (fitted on another dataset)';
}

const MODE_NOTES = {
  rules: 'A limit per quantity, in its own physical unit — no aggregate is '
       + 'involved. This is the only rule whose verdict does not depend on the '
       + 'rest of the batch: the normalised families are percentile ranks, so a '
       + 'threshold on them removes about the same share of <i>any</i> dataset. '
       + 'Set the limits in <b>limits per quantity</b> on the overview.',
  gate: 'Every family with a weight above zero must clear <b>its own</b> '
      + 'threshold — set them on the overview, one slider each. Relative: the '
      + 'ranges are this dataset\'s p5–p95, so a median episode sits near 0.50 '
      + 'on every family.',
  weighted: 'Compensatory: a strong family can carry a weak one. Only the '
          + 'weighted mean is compared to the threshold.',
};

function renderModeNote() {
  const mode = $('#mode').value;
  const counts = {};
  for (const row of state.rows || []) counts[row.decision] = (counts[row.decision] || 0) + 1;
  const total = (state.rows || []).length;

  $('#modeNote').innerHTML = MODE_NOTES[mode]
    + (total ? `<br><b>${counts.accept || 0}</b> of ${total} accepted`
             + (counts.review ? `, ${counts.review} to review` : '') : '');
  $('#thresholdRow').hidden = mode !== 'weighted';
  $('#weightsNote').textContent = mode === 'weighted' ? '' : '— ranking only';
}

$('#mode').addEventListener('change', () => { renderModeNote(); applyPolicy(); });
$('#aggregate').addEventListener('change', applyPolicy);
$('#threshold').addEventListener('input', () => {
  $('#thresholdValue').textContent = Number($('#threshold').value).toFixed(2);
  schedulePolicy();
});

/* ------------------------------------------------------------- routing */

function currentRoute() {
  const parts = (location.hash || '#/').slice(2).split('/').filter(Boolean);
  if (parts[0] === 'd' && parts[1]) {
    if (parts[2] === 'e' && parts[3] !== undefined) {
      return { view: 'episode', id: parts[1], episode: Number(parts[3]) };
    }
    return { view: 'overview', id: parts[1] };
  }
  return { view: 'home' };
}

function teardown() {
  if (state.chart) { state.chart.destroy(); state.chart = null; }
  if (state.raf) { cancelAnimationFrame(state.raf); state.raf = null; }
  state.players = [];
}

async function route() {
  const r = currentRoute();
  teardown();
  state.episode = r.view === 'episode' ? r.episode : null;

  if (r.view === 'home') {
    state.dataset = null;
    railForDataset(null);
    renderPicker();
    renderStats();
    renderEpisodeRail();
    renderHome();
    return;
  }

  state.dataset = state.datasets.find(d => d.id === r.id) || null;
  if (!state.dataset) {
    await loadDatasets();
    state.dataset = state.datasets.find(d => d.id === r.id) || null;
  }
  renderPicker();
  renderStats();
  railForDataset(state.dataset);
  if (!state.dataset) { view.innerHTML = '<div class="empty">unknown dataset</div>'; return; }

  if (!state.dataset.analyzed) { state.rows = []; renderEpisodeRail(); renderNotAnalysed(); return; }

  view.innerHTML = '<div class="empty">loading…</div>';
  try {
    state.overview = await api(`/api/datasets/${r.id}/overview`);
    state.rows = await api(`/api/datasets/${r.id}/episodes`);
  } catch (err) {
    view.innerHTML = `<div class="empty err">${esc(err.message)}</div>`;
    return;
  }
  renderControls();
  renderEpisodeRail();

  if (r.view === 'episode') renderEpisode(r.episode);
  else renderOverview();
}

window.addEventListener('hashchange', route);

/* --------------------------------------------------------------- views */

function renderHome() {
  view.innerHTML = `
    <div class="topbar"><div class="title"><h1>Episode quality</h1></div></div>
    <div class="page">
      <div class="card stack" style="max-width:760px">
        <p>Point this at a LeRobot dataset folder — the one holding
           <code>meta/</code>, <code>data/</code> and <code>videos/</code> — and it
           measures every episode, fits the normalisation ranges to that dataset,
           and lets you watch each recording next to the signals it was judged on.</p>
        <p class="sub">Measurement produces physical quantities and flags; scoring
           is a separate layer, so moving the weights or the accept threshold
           re-ranks instantly without touching a parquet file.</p>
        <p class="sub">Add a dataset from <b>settings</b> at the bottom of the rail.</p>
      </div>
    </div>`;
}

function renderNotAnalysed() {
  const ds = state.dataset;
  view.innerHTML = `
    <div class="topbar"><div class="title">
      <span class="eyebrow">dataset</span><h1>${esc(ds.name)}</h1></div>
      <span class="sub">${ds.episodes} episodes · ${esc(ds.camera || 'no camera')}</span>
    </div>
    <div class="page"><div class="card" style="max-width:680px">
      <p>Not measured yet. Open <b>settings</b> in the rail and hit
         <b>run measurement</b>.</p>
      <p class="sub">Video is the slowest of the five families by orders of
         magnitude — uncheck it for a first pass, then re-run with it on.</p>
    </div></div>`;
}

/* ------------------------------------------------------------- episode */

async function renderEpisode(episode) {
  view.innerHTML = '<div class="empty">loading episode…</div>';
  let payload;
  try {
    payload = await api(`/api/datasets/${state.dataset.id}/episodes/${episode}`);
  } catch (err) {
    view.innerHTML = `<div class="empty err">${esc(err.message)}</div>`;
    return;
  }

  const score = payload.score || {};
  const nav = payload.neighbours || {};
  const base = `#/d/${state.dataset.id}`;
  const videos = payload.videos || [];
  const frames = payload.frames || Math.round((payload.duration || 0) * (payload.fps || 30));

  view.innerHTML = `
    <div class="topbar">
      <div class="title">
        <span class="eyebrow">${esc(state.dataset.name)}</span>
        <h1>Episode ${String(episode).padStart(6, '0')}</h1>
      </div>
      ${score.decision ? `<span class="badge ${score.decision}">${score.decision}</span>` : ''}
      <span class="sub mono">score ${num(score.total)}</span>
      <span class="sub">${num(payload.duration, 2)} s · ${num(payload.fps, 0)} fps</span>
      <span class="spacer"></span>
      <span class="nav-links">
        ${nav.prev != null ? `<a href="${base}/e/${nav.prev}">← ${nav.prev}</a>` : ''}
        <a href="${base}">overview</a>
        ${nav.next != null ? `<a href="${base}/e/${nav.next}">${nav.next} →</a>` : ''}
      </span>
    </div>

    <div class="page stack">
      <div class="cameras">${
        videos.length
          ? videos.map((v, i) => `
            <div class="camera ${videos.length === 1 ? 'solo' : ''}">
              <header><span class="key">${esc(v.key)}</span>
                ${v.scored ? '<span class="tag">scored</span>' : ''}</header>
              <video data-index="${i}" src="${esc(v.url)}" preload="metadata"
                     playsinline muted${i === 0 ? '' : ' '}></video>
            </div>`).join('')
          : '<div class="card sub">no video for this episode</div>'
      }</div>

      ${payload.task ? `<div class="card">
        <h2>language instruction</h2>
        <div class="task-line">${esc(payload.task)}</div></div>` : ''}

      <div class="verdict-grid">
        <div class="card">
          <h2>score</h2>
          <div class="families">${
            Object.entries(score.families || {}).map(([name, value]) =>
              `<div class="fam"><span>${esc(name)}</span>
                <span class="bar ${barClass(value)}"><i style="width:${value == null ? 0 : value * 100}%"></i></span>
                <span class="num">${num(value, 3)}</span></div>`).join('')
          }</div>
        </div>
        <div class="card stack" id="verdictSide">${verdictSide(payload)}</div>
      </div>

      <details class="card panel" id="signalsPanel" open>
        <summary><h2>signals, synchronised with playback</h2>
          <span class="right">
            <span class="sub readout" id="readout"></span>
            <button class="ghost" id="combineBtn">combine all</button>
          </span></summary>
        <div class="panel-body">
          <div class="legend" id="legend"></div>
          <div id="charts"></div>
          <div class="hint">click any chart to seek · click a legend chip to hide a signal</div>
        </div>
      </details>

      <details class="card panel">
        <summary><h2>raw measurements</h2>
          <span class="right sub">${(payload.measures || []).length} quantities</span></summary>
        <div class="panel-body scrollx">
          <table><thead><tr><th>family</th><th>quantity</th>
            <th class="v">value</th><th>unit</th></tr></thead>
          <tbody>${(payload.measures || []).map(r =>
            `<tr><td>${esc(r.family)}</td><td>${esc(r.name)}</td>
              <td class="v mono">${esc(r.value)}</td><td class="u">${esc(r.unit)}</td></tr>`).join('')}
          </tbody></table>
        </div>
      </details>
    </div>

    <div class="transport" id="transport">
      <button id="tRestart" title="restart">↺</button>
      <button id="tPrev" title="previous frame">⏮</button>
      <button class="play" id="tPlay" title="play / pause (space)">▶</button>
      <button id="tNext" title="next frame">⏭</button>
      <input type="range" id="tScrub" min="0" max="1000" value="0">
      <span class="frames" id="tFrames">0 / ${frames}</span>
      <span class="keys">
        <span><kbd>space</kbd> play</span>
        <span><kbd>↑</kbd><kbd>↓</kbd> episode</span>
        <span><kbd>←</kbd><kbd>→</kbd> frame</span>
      </span>
    </div>`;

  mountCharts(payload);
  mountTransport(payload, frames);
}

function verdictSide(payload) {
  const score = payload.score || {};
  const flags = payload.raisedFlags || [];
  const soft = new Set(['self_collision_suspect', 'video_frozen']);
  const reasons = score.reasons || [];
  const parts = [];

  if (reasons.length) {
    parts.push(`<div><h2>why</h2><div class="reasons ${score.decision === 'reject' ? 'reject' : ''}">
      ${esc(reasons.join(' · '))}</div></div>`);
  } else {
    parts.push('<div><h2>why</h2><div class="sub">clears every gate</div></div>');
  }

  parts.push(`<div><h2>flags</h2><div class="flags">${
    flags.length
      ? flags.map(f => `<span class="flag ${soft.has(f) ? 'soft' : ''}">${esc(f)}</span>`).join('')
      : '<span class="sub">none</span>'
  }</div></div>`);

  const detail = payload.semanticDetail;
  if (detail || score.semantic != null) {
    const value = detail ? detail.score : score.semantic;
    const label = value === 1 ? 'goal reached' : value === 0.5 ? 'attempted'
      : value === 0 ? 'goal not reached' : 'not judged';
    parts.push(`<div><h2>semantic verdict</h2>
      <div><b class="mono">${value == null ? '–' : num(value, 1)}</b> — ${label}</div>
      ${detail && detail.summary ? `<div class="sub">${esc(detail.summary)}</div>` : ''}
      ${detail && detail.error ? `<div class="err">${esc(detail.error)}</div>` : ''}</div>`);
  }
  return parts.join('');
}

function refreshEpisodeVerdict() {
  // Re-scoring changed the decision; repaint the parts that show it without
  // reloading the video or the charts.
  const row = state.rows.find(r => r.episode === state.episode);
  if (!row) return;
  const badge = view.querySelector('.topbar .badge');
  if (badge) { badge.className = `badge ${row.decision}`; badge.textContent = row.decision; }
  const total = view.querySelector('.topbar .mono');
  if (total) total.textContent = `score ${num(row.total)}`;
  view.querySelectorAll('.fam').forEach(fam => {
    const value = row.families[fam.children[0].textContent];
    fam.children[1].className = `bar ${barClass(value)}`;
    fam.children[1].firstElementChild.style.width = `${value == null ? 0 : value * 100}%`;
    fam.children[2].textContent = num(value, 3);
  });
  const side = $('#verdictSide');
  if (side) {
    side.innerHTML = verdictSide({
      score: { reasons: row.reasons, decision: row.decision, semantic: row.semantic },
      raisedFlags: row.flags,
      semanticDetail: null,
    });
  }
  renderEpisodeRail();
}

function mountCharts(payload) {
  const readout = $('#readout');
  const charts = new SyncChart($('#charts'), payload, { onSeek: seekAll });
  state.chart = charts;
  charts.mountLegend($('#legend'));
  charts.onCursor = t => {
    readout.innerHTML = `<b class="mono">${t.toFixed(2)}</b> / ${num(payload.duration, 2)} s`
      + (charts.inContact(t) ? ' · <b style="color:var(--bad)">contact</b>' : '');
  };
  // A canvas sized while its <details> is closed measures 0 wide.
  $('#signalsPanel').addEventListener('toggle', ev => {
    if (ev.target.open) charts.relayout();
  });
  $('#combineBtn').addEventListener('click', ev => {
    const on = !charts.combined;
    charts.setCombined(on);
    ev.currentTarget.classList.toggle('on', on);
    ev.currentTarget.textContent = on ? 'split apart' : 'combine all';
  });
  charts.setCursor(0);
}

/* ----------------------------------------------------------- transport */

function seekAll(t) {
  for (const v of state.players) {
    if (Number.isFinite(v.duration)) v.currentTime = Math.min(t, v.duration - 1e-3);
  }
}

function mountTransport(payload, frames) {
  state.players = [...view.querySelectorAll('.camera video')];
  if (!state.players.length) { $('#transport').hidden = true; return; }

  const clock = state.players[0];
  const play = $('#tPlay');
  const scrub = $('#tScrub');
  const counter = $('#tFrames');
  const duration = payload.duration || 1;
  const fps = payload.fps || 30;
  const step = 1 / fps;

  const paint = () => {
    const t = clock.currentTime;
    scrub.value = String(Math.round((t / duration) * 1000));
    counter.textContent = `${Math.min(Math.round(t * fps), frames)} / ${frames}`;
    if (state.chart) state.chart.setCursor(t);
  };

  const toggle = () => {
    if (clock.paused) state.players.forEach(v => v.play().catch(() => {}));
    else state.players.forEach(v => v.pause());
  };

  play.addEventListener('click', toggle);
  $('#tRestart').addEventListener('click', () => { seekAll(0); paint(); });
  $('#tPrev').addEventListener('click', () => {
    seekAll(Math.max(clock.currentTime - step, 0)); paint();
  });
  $('#tNext').addEventListener('click', () => {
    seekAll(Math.min(clock.currentTime + step, duration)); paint();
  });
  scrub.addEventListener('input', () => {
    seekAll((Number(scrub.value) / 1000) * duration);
    paint();
  });

  clock.addEventListener('play', () => { play.textContent = '⏸'; });
  clock.addEventListener('pause', () => { play.textContent = '▶'; });
  clock.addEventListener('ended', () => { play.textContent = '▶'; });
  clock.addEventListener('timeupdate', paint);
  clock.addEventListener('seeked', paint);

  // The clock drives the other cameras: they are the same recording from
  // different angles, so one transport moves all of them.
  clock.addEventListener('play', () => {
    state.players.slice(1).forEach(v => { v.currentTime = clock.currentTime; v.play().catch(() => {}); });
  });
  clock.addEventListener('pause', () => state.players.slice(1).forEach(v => v.pause()));

  const tick = () => {
    if (!document.body.contains(clock)) return;
    if (!clock.paused && !clock.ended) {
      paint();
      // Nudge any camera that has drifted more than two frames from the clock.
      for (const v of state.players.slice(1)) {
        if (Math.abs(v.currentTime - clock.currentTime) > 2 * step) v.currentTime = clock.currentTime;
      }
    }
    state.raf = requestAnimationFrame(tick);
  };
  state.raf = requestAnimationFrame(tick);

  state.transportToggle = toggle;
  state.transportStep = delta => { seekAll(clock.currentTime + delta * step); paint(); };
  paint();
}

document.addEventListener('keydown', ev => {
  if (/^(INPUT|SELECT|TEXTAREA)$/.test(ev.target.tagName)) return;
  const r = currentRoute();
  if (r.view !== 'episode') return;

  if (ev.code === 'Space') { ev.preventDefault(); state.transportToggle && state.transportToggle(); }
  else if (ev.key === 'ArrowLeft') { ev.preventDefault(); state.transportStep && state.transportStep(-1); }
  else if (ev.key === 'ArrowRight') { ev.preventDefault(); state.transportStep && state.transportStep(1); }
  else if (ev.key === 'ArrowUp' || ev.key === 'ArrowDown') {
    ev.preventDefault();
    const order = state.rows.map(r2 => r2.episode).sort((a, b) => a - b);
    const i = order.indexOf(state.episode);
    const next = order[i + (ev.key === 'ArrowDown' ? 1 : -1)];
    if (next != null) location.hash = `#/d/${state.dataset.id}/e/${next}`;
  }
});

/* ------------------------------------------------------------ overview */

function renderOverview() {
  const o = state.overview;
  const d = o.decisions || {};
  const families = Object.keys(o.weights || {});

  view.innerHTML = `
    <div class="topbar">
      <div class="title">
        <span class="eyebrow">dataset</span>
        <h1>${esc(state.dataset.name)}</h1>
      </div>
      <span class="sub">${o.count} episodes · ${esc(state.dataset.camera || 'no camera')}</span>
      <span class="spacer"></span>
      <span class="sub">${state.dataset.analyzedWithVideo ? 'measured with video' : 'measured without video'}</span>
    </div>
    <div class="page stack">
      ${state.dataset.task ? `<div class="card"><h2>language instruction</h2>
        <div class="task-line">${esc(state.dataset.task)}</div></div>` : ''}
      <div class="tiles">
        <div class="tile"><div class="k">episodes</div><div class="v">${o.count}</div></div>
        <div class="tile accept"><div class="k">accept</div><div class="v">${d.accept || 0}</div></div>
        <div class="tile review"><div class="k">review</div><div class="v">${d.review || 0}</div></div>
        <div class="tile reject"><div class="k">reject</div><div class="v">${d.reject || 0}</div></div>
        <div class="tile"><div class="k">mean score</div><div class="v">${num(o.mean)}</div></div>
        <div class="tile"><div class="k">median</div><div class="v">${num(o.median)}</div></div>
      </div>
      <div class="grid2">
        <div class="card"><h2>score distribution</h2>
          <canvas class="hist" id="hist"></canvas>
          <div class="hint">the line marks the accept threshold (${num(o.threshold, 2)})</div>
        </div>
        <div class="card" id="thresholdCard"></div>
      </div>
      ${o.mode === 'rules' ? '<div id="rulesPanel"></div>' : ''}
      ${agreementPanel(o.agreement)}
      <div class="card">
        <h2>episodes</h2>
        <div class="table-tools" id="tools"></div>
        <div class="tbl-wrap"><table class="rows" id="rows"></table></div>
      </div>
    </div>`;

  drawHistogram($('#hist'), o.histogram, o.threshold);
  renderThresholds();
  if (o.mode === 'rules') renderRules();
  renderTools();
  renderTable();
}

/* ------------------------------------------------------- family thresholds */

/** Threshold per family, dragged against that family's own distribution.
 *
 *  The slider sits on the p5–p95 span with the median marked, so you set a cut
 *  by seeing where it lands in the data and what it costs — not by picking a
 *  number that sounds strict.  Counts update while dragging (locally, no round
 *  trip); the server is told on release. */
function renderThresholds() {
  const card = $('#thresholdCard');
  if (!card) return;
  const o = state.overview;
  const families = Object.keys(o.weights || {});
  const gating = o.mode === 'gate';

  card.innerHTML = `
    <h2>${gating ? 'threshold per family' : 'per-family spread'}</h2>
    <div class="hint" style="margin:-3px 0 12px">
      ${gating
        ? 'Each family must clear its own line. The bar is that family\'s p5–p95 across '
          + 'the dataset, the tick is its median — drag the handle to place the cut.'
        : 'The bar is each family\'s p5–p95, the tick its median. Switch the rule to '
          + '<b>every family must pass</b> to make these thresholds decide.'}
    </div>
    <div class="thresholds">${families.map(f => thresholdRow(f, o, gating)).join('')}</div>
    ${flagsPanel(o.flags)}`;

  if (!gating) return;
  card.querySelectorAll('input[data-family]').forEach(input => {
    input.addEventListener('input', () => {
      const family = input.dataset.family;
      const value = Number(input.value);
      card.querySelector(`[data-value="${family}"]`).textContent = value.toFixed(2);
      card.querySelector(`[data-cut="${family}"]`).style.left = `${value * 100}%`;
      const below = state.rows.filter(r => {
        const v = r.families[family];
        return v != null && v < value;
      }).length;
      const el2 = card.querySelector(`[data-below="${family}"]`);
      el2.textContent = below ? `${below} below` : 'none below';
      el2.classList.toggle('hits', below > 0);
    });
    input.addEventListener('change', applyThresholds);
  });
}

function thresholdRow(family, o, gating) {
  const dist = o.distributions[family];
  const weight = o.weights[family];
  const value = (o.minimums && o.minimums[family] != null) ? o.minimums[family] : o.threshold;
  if (!dist) {
    return `<div class="threshold"><span>${esc(family)}</span>
      <span class="sub">not measured</span><span></span><span></span></div>`;
  }
  const lo = Math.max(0, Math.min(dist.p5 * 100, 100));
  const hi = Math.max(lo, Math.min(dist.p95 * 100, 100));
  const med = Math.max(0, Math.min(dist.p50 * 100, 100));
  const below = state.rows.filter(r => {
    const v = r.families[family];
    return v != null && v < value;
  }).length;

  return `<div class="threshold ${weight ? '' : 'muted'}">
    <span class="tname">${esc(family)}${weight ? '' : ' <span class="sub">(weight 0)</span>'}</span>
    <span class="ttrack">
      <i class="qspan" style="left:${lo}%;width:${Math.max(hi - lo, 1)}%"></i>
      <i class="qmed" style="left:calc(${med}% - 1px)"></i>
      ${gating ? `<i class="tcut" data-cut="${family}" style="left:${value * 100}%"></i>
        <input type="range" min="0" max="1" step="0.01" value="${value}"
               data-family="${family}" aria-label="${esc(family)} threshold">` : ''}
    </span>
    <span class="num mono" data-value="${family}">${gating ? value.toFixed(2) : num(dist.p50, 2)}</span>
    <span class="tbelow sub ${gating && below ? 'hits' : ''}" data-below="${family}">${
      gating ? (below ? `${below} below` : 'none below') : ''}</span>
  </div>`;
}

async function applyThresholds() {
  const minimums = {};
  for (const input of document.querySelectorAll('#thresholdCard input[data-family]')) {
    minimums[input.dataset.family] = Number(input.value);
  }
  state.overview = await api(`/api/datasets/${state.dataset.id}/policy`, {
    method: 'PUT', body: { minimums },
  });
  state.rows = await api(`/api/datasets/${state.dataset.id}/episodes`);
  renderModeNote();
  renderEpisodeRail();
  renderOverview();
}

/* --------------------------------------------------------------- limits */

const FAMILY_ORDER = ['smoothness', 'acceleration', 'contact', 'timing', 'video'];

/** The rules editor: one limit per quantity, in physical units. */
function renderRules() {
  const host = $('#rulesPanel');
  if (!host) return;
  const rules = state.overview.rules || [];
  const enabled = rules.filter(r => r.enabled);
  const caught = new Set();
  for (const row of state.rows) {
    if (row.decision !== 'accept') caught.add(row.episode);
  }

  const byFamily = new Map();
  for (const rule of rules) {
    const key = rule.family || 'other';
    if (!byFamily.has(key)) byFamily.set(key, []);
    byFamily.get(key).push(rule);
  }
  const families = [...byFamily.keys()].sort(
    (a, b) => FAMILY_ORDER.indexOf(a) - FAMILY_ORDER.indexOf(b));

  host.innerHTML = `<div class="card">
    <div class="rules-head">
      <h2 style="margin:0">limits per quantity</h2>
      <span class="sub">${enabled.length} of ${rules.length} active</span>
      <span class="spacer"></span>
      <button class="ghost" id="seedRules">re-seed from this dataset</button>
    </div>
    <div class="hint" style="margin:6px 0 12px">
      Each limit is in that quantity's own unit and is compared to the episode
      directly — nothing here is a percentile, so a limit you set here means the
      same thing on the next dataset. <b>observed</b> shows where this dataset
      actually sits, so you can see what a limit would cost before enabling it.
    </div>
    <div class="scrollx"><table class="rules">
      <thead><tr>
        <th>on</th><th>quantity</th><th class="v">limit</th><th>unit</th>
        <th class="v">observed p5 · p50 · p95</th><th>then</th><th class="v">catches</th>
      </tr></thead>
      <tbody>${families.map(family => `
        <tr class="rules-group"><td colspan="7">${esc(family)}</td></tr>
        ${byFamily.get(family).map(r => ruleRow(r)).join('')}`).join('')}
      </tbody>
    </table></div>
  </div>`;

  host.querySelectorAll('[data-quantity]').forEach(node => {
    node.addEventListener('change', () => applyRules());
  });
  $('#seedRules').addEventListener('click', async () => {
    state.overview = await api(`/api/datasets/${state.dataset.id}/rules/seed`, { method: 'POST' });
    state.rows = await api(`/api/datasets/${state.dataset.id}/episodes`);
    renderModeNote(); renderEpisodeRail(); renderOverview();
  });
}

function ruleRow(r) {
  const q = r.quantity;
  const span = [r.p5, r.p50, r.p95].map(v => num(v, 3)).join(' · ');
  return `<tr class="${r.enabled ? 'on' : ''}">
    <td><input type="checkbox" data-quantity="${q}" data-field="enabled"
         ${r.enabled ? 'checked' : ''}></td>
    <td>${esc(r.label)}
      <select data-quantity="${q}" data-field="op" class="mini">
        <option value=">"${r.op === '>' ? ' selected' : ''}>above</option>
        <option value="<"${r.op === '<' ? ' selected' : ''}>below</option>
      </select></td>
    <td class="v"><input type="number" step="any" class="limit mono"
         data-quantity="${q}" data-field="limit" value="${r.limit}"></td>
    <td class="u">${esc(r.unit)}</td>
    <td class="v sub mono">${span}</td>
    <td><select data-quantity="${q}" data-field="action" class="mini">
        <option value="reject"${r.action === 'reject' ? ' selected' : ''}>reject</option>
        <option value="review"${r.action === 'review' ? ' selected' : ''}>review</option>
      </select></td>
    <td class="v ${r.enabled && r.hits ? 'hits' : 'sub'}">${r.hits}</td>
  </tr>`;
}

async function applyRules() {
  const byQuantity = new Map((state.overview.rules || []).map(r => [r.quantity, { ...r }]));
  for (const node of document.querySelectorAll('[data-quantity]')) {
    const rule = byQuantity.get(node.dataset.quantity);
    if (!rule) continue;
    if (node.dataset.field === 'enabled') rule.enabled = node.checked;
    else if (node.dataset.field === 'limit') rule.limit = Number(node.value);
    else rule[node.dataset.field] = node.value;
  }
  const rules = [...byQuantity.values()].map(r => ({
    quantity: r.quantity, limit: r.limit, op: r.op, action: r.action, enabled: r.enabled,
  }));
  state.overview = await api(`/api/datasets/${state.dataset.id}/policy`, {
    method: 'PUT', body: { rules },
  });
  state.rows = await api(`/api/datasets/${state.dataset.id}/episodes`);
  renderModeNote();
  renderEpisodeRail();
  renderOverview();
}

function quantileRow(family, dist) {
  if (!dist) return `<div class="quantile"><span>${esc(family)}</span>
    <span class="sub">not measured</span><span></span></div>`;
  // Clamp inside the track: a p95 of exactly 1.0 would otherwise push the span
  // and the median marker a couple of pixels past their container.
  const lo = Math.max(0, Math.min(dist.p5 * 100, 100));
  const hi = Math.max(lo, Math.min(dist.p95 * 100, 100));
  const med = Math.max(0, Math.min(dist.p50 * 100, 100));
  return `<div class="quantile">
    <span>${esc(family)}</span>
    <span class="qtrack">
      <i class="qspan" style="left:${lo}%;width:${Math.max(hi - lo, 1)}%"></i>
      <i class="qmed" style="left:calc(${med}% - 1px)"></i>
    </span>
    <span class="num">${num(dist.p50, 2)}</span></div>`;
}

function flagsPanel(flags) {
  const entries = Object.entries(flags || {});
  if (!entries.length) return '<div class="hint" style="margin-top:12px">no flags raised</div>';
  const soft = new Set(['self_collision_suspect', 'video_frozen']);
  return `<div style="margin-top:14px"><h2>flags raised</h2><div class="flags">`
    + entries.map(([name, count]) =>
        `<span class="flag ${soft.has(name) ? 'soft' : ''}">${esc(name)} · ${count}</span>`).join('')
    + `</div></div>`;
}

function agreementPanel(a) {
  if (!a) return '';
  return `<div class="card"><h2>against the dataset's own discard list</h2>
    <div class="sub">${a.discarded} episodes were discarded by hand, ${a.kept} kept.
      The hard flags catch <b>${a.discardedFlagged}</b> of the discarded ones outright;
      the current rule rejects <b>${a.discardedRejected}</b> of them and
      <b>${a.keptRejected}</b> of the kept ones.
      Where the two disagree, the usual reason is that the episode moved well but
      did not achieve the task — which is what the semantic filter is for.</div></div>`;
}

function drawHistogram(canvas, hist, threshold) {
  if (!canvas || !hist || !hist.counts.length) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth, height = 148;
  canvas.width = width * dpr; canvas.height = height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

  const max = Math.max(...hist.counts, 1);
  const pad = { l: 26, r: 6, t: 8, b: 18 };
  const w = width - pad.l - pad.r, h = height - pad.t - pad.b;
  ctx.clearRect(0, 0, width, height);

  ctx.fillStyle = css('--accent');
  hist.counts.forEach((count, i) => {
    const x = pad.l + (i / hist.counts.length) * w;
    const bw = w / hist.counts.length - 1.5;
    const bh = (count / max) * h;
    ctx.globalAlpha = 0.82;
    ctx.fillRect(x, pad.t + h - bh, Math.max(bw, 1), bh);
  });
  ctx.globalAlpha = 1;

  ctx.strokeStyle = css('--bad');
  ctx.lineWidth = 1.5;
  const tx = pad.l + threshold * w;
  ctx.beginPath(); ctx.moveTo(tx, pad.t); ctx.lineTo(tx, pad.t + h); ctx.stroke();

  ctx.strokeStyle = css('--line');
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t + h); ctx.lineTo(pad.l + w, pad.t + h); ctx.stroke();
  ctx.fillStyle = css('--muted');
  ctx.font = '11px ui-monospace, monospace';
  ctx.textAlign = 'right';
  ctx.fillText(String(max), pad.l - 4, pad.t + 8);
  ctx.fillText('0', pad.l - 4, pad.t + h);
  ctx.textAlign = 'center';
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    ctx.fillText(v.toFixed(2), pad.l + v * w, height - 4);
  }
}

/* --------------------------------------------------------------- table */

function renderTools() {
  const tools = $('#tools');
  tools.innerHTML = '';
  for (const decision of ['accept', 'review', 'reject']) {
    const count = state.rows.filter(r => r.decision === decision).length;
    const chip = el(`<button class="chip ${state.filter.decision.has(decision) ? 'on' : ''}">
      ${decision} <span class="sub">${count}</span></button>`);
    chip.addEventListener('click', () => {
      if (state.filter.decision.has(decision)) state.filter.decision.delete(decision);
      else state.filter.decision.add(decision);
      renderTools(); renderTable();
    });
    tools.appendChild(chip);
  }
  const search = el(`<input type="text" placeholder="filter by flag or reason"
                      value="${esc(state.filter.text)}">`);
  search.addEventListener('input', () => {
    state.filter.text = search.value.toLowerCase();
    renderTable();
  });
  tools.appendChild(search);
  tools.appendChild(el(`<span class="hint" id="rowCount"></span>`));
}

function visibleRows() {
  const { decision, text } = state.filter;
  return state.rows.filter(row => {
    if (decision.size && !decision.has(row.decision)) return false;
    if (!text) return true;
    const haystack = [row.episode, row.decision, ...(row.flags || []),
                      ...(row.reasons || [])].join(' ').toLowerCase();
    return haystack.includes(text);
  });
}

function renderTable() {
  const families = Object.keys(state.overview.weights || {});
  const columns = [
    { key: 'episode', label: 'ep', get: r => r.episode, cell: r => r.episode },
    { key: 'decision', label: 'decision', get: r => r.decision,
      cell: r => `<span class="badge ${r.decision}">${r.decision}</span>` },
    { key: 'total', label: 'total', v: true, get: r => r.total,
      cell: r => `${bar(r.total)}<span class="mono">${num(r.total)}</span>` },
    ...families.map(f => ({
      key: f, label: f.slice(0, 9), v: true,
      get: r => r.families[f], cell: r => num(r.families[f], 2),
    })),
    { key: 'semantic', label: 'sem', v: true, get: r => r.semantic,
      cell: r => r.semantic == null ? '–' : num(r.semantic, 1) },
    { key: 'duration', label: 'dur (s)', v: true, get: r => r.duration,
      cell: r => num(r.duration, 1) },
    { key: 'contactEvents', label: 'contact', v: true, get: r => r.contactEvents,
      cell: r => r.contactEvents },
    { key: 'flags', label: 'flags', get: r => (r.flags || []).length,
      cell: r => (r.flags || []).map(f => `<span class="flag">${esc(f)}</span>`).join(' ') },
  ];

  const rows = visibleRows();
  const { key, asc } = state.sort;
  const column = columns.find(c => c.key === key) || columns[2];
  rows.sort((a, b) => {
    const va = column.get(a), vb = column.get(b);
    if (va === null || va === undefined) return 1;
    if (vb === null || vb === undefined) return -1;
    return (va > vb ? 1 : va < vb ? -1 : 0) * (asc ? 1 : -1);
  });

  const table = $('#rows');
  table.innerHTML =
    `<thead><tr>${columns.map(c =>
      `<th data-key="${c.key}" class="${c.v ? 'v' : ''} ${c.key === key ? 'sorted ' + (asc ? 'asc' : '') : ''}">
        ${esc(c.label)}</th>`).join('')}</tr></thead>
     <tbody>${rows.map(r =>
      `<tr data-episode="${r.episode}" class="${r.humanDiscarded ? 'discarded' : ''}">
        ${columns.map(c => `<td class="${c.v ? 'v' : ''}">${c.cell(r)}</td>`).join('')}
      </tr>`).join('')}</tbody>`;

  table.querySelectorAll('th').forEach(th => th.addEventListener('click', () => {
    const next = th.dataset.key;
    state.sort = { key: next, asc: state.sort.key === next ? !state.sort.asc : true };
    renderTable();
  }));
  table.querySelectorAll('tbody tr').forEach(tr => tr.addEventListener('click', () => {
    location.hash = `#/d/${state.dataset.id}/e/${tr.dataset.episode}`;
  }));
  const count = $('#rowCount');
  if (count) count.textContent = `${rows.length} of ${state.rows.length} shown`;
}

/* ----------------------------------------------------------------- theme */

function applyTheme(name) {
  document.documentElement.dataset.theme = name;
  try { localStorage.setItem('eq-theme', name); } catch (_) { /* private mode */ }
  if (state.chart) state.chart.relayout();       // canvases bake in the colours
  if (currentRoute().view === 'overview' && state.overview) {
    drawHistogram($('#hist'), state.overview.histogram, state.overview.threshold);
  }
}

$('#themeBtn').addEventListener('click', () => {
  applyTheme(document.documentElement.dataset.theme === 'light' ? 'dark' : 'light');
});

/* ----------------------------------------------------------------- boot */

(async function start() {
  let stored = null;
  try { stored = localStorage.getItem('eq-theme'); } catch (_) { /* private mode */ }
  document.documentElement.dataset.theme = stored || 'dark';

  try {
    const health = await api('/api/health');
    $('#stateDir').textContent = `cache: ${health.stateDir}`;
  } catch (_) { /* the page still works if health is unavailable */ }
  await loadDatasets();
  route();
})();
