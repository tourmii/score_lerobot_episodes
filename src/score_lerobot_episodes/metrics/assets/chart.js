/* Synchronised signal charts.
 *
 * One small chart per signal in a responsive grid, rather than one tall canvas
 * with the traces stacked in lanes: nine signals crammed into a single plot
 * gives every one of them the same 60 pixels of a shared axis, and none of them
 * enough. Separate cards let each trace keep its own y-scale, its own label and
 * its own live value.
 *
 * They stay one instrument: a common time axis, a playhead that follows the
 * <video> across all of them, shared contact and idle shading, and click-to-seek
 * on any chart.
 *
 * Classic script on purpose (no ES module): the standalone pages
 * `visualize.render_episode_html` writes inline this file and must still work
 * when opened straight off the filesystem.
 *
 *   const charts = new SyncChart(container, payload, {
 *     onSeek: t => video.currentTime = t,
 *   });
 *   charts.mountLegend(document.getElementById('legend'));
 *   video.addEventListener('timeupdate', () => charts.setCursor(video.currentTime));
 */
window.SyncChart = class SyncChart {
  constructor(container, payload, opts = {}) {
    this.container = container;
    this.payload = payload;
    this.onSeek = opts.onSeek || null;
    this.height = opts.height || 88;
    this.pad = { l: 42, r: 8, t: 7, b: 15 };
    this.duration = Math.max(payload.duration || 0, 1e-6);
    this.cursor = 0;
    this.hover = null;
    this.hidden = new Set(opts.hidden || []);
    this.combined = false;
    this.entries = [];
    this.combinedEntry = null;
    this.legendEl = null;

    this.draw = this.draw.bind(this);
    this.relayout = this.relayout.bind(this);

    this._build();
    this.relayout();

    this._observer = window.ResizeObserver ? new ResizeObserver(this.relayout) : null;
    if (this._observer) this._observer.observe(container);
    window.addEventListener('resize', this.relayout);
    this._scheme = window.matchMedia('(prefers-color-scheme: dark)');
    if (this._scheme.addEventListener) this._scheme.addEventListener('change', this.relayout);
  }

  destroy() {
    if (this._observer) this._observer.disconnect();
    window.removeEventListener('resize', this.relayout);
    if (this._scheme && this._scheme.removeEventListener) {
      this._scheme.removeEventListener('change', this.relayout);
    }
  }

  /* ---------------------------------------------------------------- build */

  _build() {
    this.container.classList.add('chart-grid');
    this.container.innerHTML = '';

    for (const track of this.payload.tracks) {
      const card = document.createElement('div');
      card.className = 'chart-card';
      card.dataset.label = track.label;
      card.innerHTML =
        `<header>
           <i class="swatch" style="background:${track.color}"></i>
           <span class="name">${escapeHtml(track.label)}</span>
           <span class="unit">${escapeHtml(track.unit || '')}</span>
           <span class="value" style="color:${track.color}">–</span>
         </header>`;

      const canvas = document.createElement('canvas');
      card.appendChild(canvas);
      this.container.appendChild(card);

      const entry = {
        track,
        card,
        canvas,
        ctx: canvas.getContext('2d'),
        bg: document.createElement('canvas'),   // static layer, drawn once per layout
        valueEl: card.querySelector('.value'),
        width: 0,
        height: 0,
      };
      entry.bgCtx = entry.bg.getContext('2d');

      canvas.addEventListener('pointerdown', ev => {
        this.setCursor(this._timeAt(ev, canvas), { emit: true });
      });
      canvas.addEventListener('pointermove', ev => {
        this.hover = this._timeAt(ev, canvas);
        if (this.onHover) this.onHover(this.hover);
        this._paintAll();
      });
      canvas.addEventListener('pointerleave', () => {
        this.hover = null;
        if (this.onHover) this.onHover(null);
        this._paintAll();
      });

      this.entries.push(entry);
    }

    // "Combine all": every trace on one axis, each min-max normalised to its own
    // range.  The units are incomparable — m/s next to rad/s² next to grey
    // levels — so the only readable overlay is a shape overlay, and it is for
    // spotting *coincidence* (does the base tilt move when the wrist does?),
    // never for reading a value off.
    const card = document.createElement('div');
    card.className = 'chart-card combined is-hidden';
    card.innerHTML =
      `<header><span class="name">all signals</span>
        <span class="unit">each scaled to its own range</span></header>`;
    const canvas = document.createElement('canvas');
    card.appendChild(canvas);
    this.container.appendChild(card);
    canvas.addEventListener('pointerdown', ev => {
      this.setCursor(this._timeAt(ev, canvas), { emit: true });
    });
    canvas.addEventListener('pointermove', ev => {
      this.hover = this._timeAt(ev, canvas);
      if (this.onHover) this.onHover(this.hover);
      this._paintAll();
    });
    canvas.addEventListener('pointerleave', () => {
      this.hover = null;
      if (this.onHover) this.onHover(null);
      this._paintAll();
    });
    this.combinedEntry = {
      track: null, card, canvas, ctx: canvas.getContext('2d'),
      bg: document.createElement('canvas'), valueEl: null, width: 0, height: 0,
    };
    this.combinedEntry.bgCtx = this.combinedEntry.bg.getContext('2d');

    this._syncVisibility();
  }

  setCombined(on) {
    this.combined = !!on;
    this._syncVisibility();
    this.relayout();
  }

  /* ---------------------------------------------------------------- state */

  get visible() {
    return this.entries.filter(e => !this.hidden.has(e.track.label));
  }

  toggle(label) {
    if (this.hidden.has(label)) this.hidden.delete(label);
    else this.hidden.add(label);
    this._syncVisibility();
    this._syncLegend();
    this.relayout();
  }

  _syncVisibility() {
    for (const entry of this.entries) {
      const off = this.combined || this.hidden.has(entry.track.label);
      entry.card.classList.toggle('is-hidden', off);
    }
    if (this.combinedEntry) {
      this.combinedEntry.card.classList.toggle('is-hidden', !this.combined);
    }
  }

  setCursor(t, opts = {}) {
    this.cursor = Math.max(0, Math.min(this.duration, t || 0));
    this._paintAll();
    if (opts.emit && this.onSeek) this.onSeek(this.cursor);
    if (this.onCursor) this.onCursor(this.cursor);
  }

  /** Value of every visible track at time `t`. */
  valuesAt(t) {
    return this.visible.map(({ track }) => ({
      label: track.label,
      unit: track.unit,
      color: track.color,
      value: sampleAt(track, t),
    }));
  }

  inContact(t) {
    return (this.payload.contact || []).some(([a, b]) => t >= a && t <= b);
  }

  /* -------------------------------------------------------------- legend */

  mountLegend(el) {
    this.legendEl = el;
    el.innerHTML = '';
    for (const { track } of this.entries) {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'legend-chip';
      chip.dataset.label = track.label;
      chip.innerHTML =
        `<i class="swatch" style="background:${track.color}"></i>` +
        `<span>${escapeHtml(track.label)}</span>`;
      chip.addEventListener('click', () => this.toggle(track.label));
      el.appendChild(chip);
    }
    for (const [cls, text] of [['contact', 'contact'], ['idle', 'idle']]) {
      const tag = document.createElement('span');
      tag.className = 'legend-static';
      tag.innerHTML = `<i class="swatch swatch-${cls}"></i><span>${text}</span>`;
      el.appendChild(tag);
    }
    this._syncLegend();
  }

  _syncLegend() {
    if (!this.legendEl) return;
    for (const chip of this.legendEl.querySelectorAll('.legend-chip')) {
      chip.classList.toggle('off', this.hidden.has(chip.dataset.label));
    }
  }

  /* --------------------------------------------------------------- input */

  _timeAt(ev, canvas) {
    const rect = canvas.getBoundingClientRect();
    const span = rect.width - this.pad.l - this.pad.r;
    if (span <= 0) return 0;
    const t = ((ev.clientX - rect.left - this.pad.l) / span) * this.duration;
    return Math.max(0, Math.min(this.duration, t));
  }

  /* ---------------------------------------------------------------- draw */

  /** Re-measure every card and re-render its static layer. */
  relayout() {
    const dpr = window.devicePixelRatio || 1;
    const size = (entry, height) => {
      const width = entry.card.clientWidth - 20;   // card padding
      if (width <= 0) return false;
      entry.width = width;
      entry.height = height;
      for (const canvas of [entry.canvas, entry.bg]) {
        canvas.width = Math.round(width * dpr);
        canvas.height = Math.round(height * dpr);
      }
      entry.canvas.style.height = height + 'px';
      entry.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      entry.bgCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return true;
    };

    if (this.combined) {
      if (size(this.combinedEntry, this.height * 2.6)) this._renderCombined();
    } else {
      for (const entry of this.visible) {
        if (size(entry, this.height)) this._renderStatic(entry);
      }
    }
    this._paintAll();
  }

  /** Full redraw — the public name; layout is what actually changes. */
  draw() {
    this.relayout();
  }

  _x(t, width) {
    return this.pad.l + (t / this.duration) * (width - this.pad.l - this.pad.r);
  }

  _scale(entry) {
    const { track } = entry;
    let lo = Math.min(0, track.min);
    let hi = track.max;
    if (track.reference != null) hi = Math.max(hi, track.reference);
    if (hi - lo < 1e-9) hi = lo + 1;
    const top = this.pad.t;
    const bottom = entry.height - this.pad.b;
    return { lo, hi, top, bottom, y: v => bottom - ((v - lo) / (hi - lo)) * (bottom - top) };
  }

  /** Everything that does not move: bands, grid, the trace itself, labels. */
  _renderStatic(entry) {
    const ctx = entry.bgCtx;
    const { track, width, height } = entry;
    const { lo, hi, top, bottom, y } = this._scale(entry);
    const muted = css('--muted'), line = css('--line');

    ctx.clearRect(0, 0, width, height);

    // Idle first, contact over it: contact is the rarer, more urgent mark.
    ctx.fillStyle = 'rgba(128,128,128,0.14)';
    for (const [a, b] of this.payload.idle || []) {
      const xa = this._x(a, width);
      ctx.fillRect(xa, top, Math.max(this._x(b, width) - xa, 1), bottom - top);
    }
    ctx.fillStyle = 'rgba(235,87,87,0.32)';
    for (const [a, b] of this.payload.contact || []) {
      const xa = this._x(a, width);
      ctx.fillRect(xa, top, Math.max(this._x(b, width) - xa, 2), bottom - top);
    }

    // Baseline, and a zero line when the trace goes negative.
    ctx.strokeStyle = line;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(this.pad.l, bottom + 0.5);
    ctx.lineTo(width - this.pad.r, bottom + 0.5);
    ctx.stroke();
    if (lo < 0 && hi > 0) {
      ctx.save();
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      ctx.moveTo(this.pad.l, y(0));
      ctx.lineTo(width - this.pad.r, y(0));
      ctx.stroke();
      ctx.restore();
    }

    if (track.reference != null) {
      ctx.save();
      ctx.strokeStyle = muted;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(this.pad.l, y(track.reference));
      ctx.lineTo(width - this.pad.r, y(track.reference));
      ctx.stroke();
      ctx.restore();
    }

    // The trace: a soft fill under it, then the line on top.
    const points = [];
    for (let k = 0; k < track.t.length; k++) {
      const v = track.y[k];
      points.push(v == null ? null : [this._x(track.t[k], width), y(v)]);
    }

    if (track.kind !== 'step') {
      ctx.fillStyle = withAlpha(track.color, 0.15);
      ctx.beginPath();
      let open = false;
      for (const point of points) {
        if (point === null) {
          if (open) { ctx.lineTo(lastX, bottom); ctx.closePath(); ctx.fill(); open = false; }
          continue;
        }
        if (!open) { ctx.moveTo(point[0], bottom); ctx.lineTo(point[0], point[1]); open = true; }
        else ctx.lineTo(point[0], point[1]);
        var lastX = point[0];
      }
      if (open) { ctx.lineTo(lastX, bottom); ctx.closePath(); ctx.fill(); }
    }

    ctx.strokeStyle = track.color;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    let pen = false, prevY = null;
    for (const point of points) {
      if (point === null) { pen = false; prevY = null; continue; }
      const [x, py] = point;
      if (!pen) { ctx.moveTo(x, py); pen = true; }
      else if (track.kind === 'step') { ctx.lineTo(x, prevY); ctx.lineTo(x, py); }
      else ctx.lineTo(x, py);
      prevY = py;
    }
    ctx.stroke();

    // y range, left; time ticks, bottom.
    ctx.fillStyle = muted;
    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'right';
    ctx.fillText(fmtAxis(hi), this.pad.l - 5, top + 8);
    ctx.fillText(fmtAxis(lo), this.pad.l - 5, bottom);

    const ticks = width < 320 ? 2 : 4;
    for (let k = 0; k <= ticks; k++) {
      const t = (this.duration * k) / ticks;
      const x = this._x(t, width);
      ctx.strokeStyle = line;
      ctx.beginPath();
      ctx.moveTo(x, bottom); ctx.lineTo(x, bottom + 3);
      ctx.stroke();
      // The unit rides on the last label; a separate one collides with it.
      const last = k === ticks;
      ctx.textAlign = last ? 'right' : 'center';
      ctx.fillText(t.toFixed(1) + (last ? ' s' : ''), last ? x + 2 : x, height - 3);
    }
    ctx.textAlign = 'left';
  }

  /** Every visible trace on one axis, each normalised to its own range. */
  _renderCombined() {
    const entry = this.combinedEntry;
    const ctx = entry.bgCtx;
    const { width, height } = entry;
    const top = this.pad.t, bottom = height - this.pad.b;
    const muted = css('--muted'), line = css('--line');

    ctx.clearRect(0, 0, width, height);

    ctx.fillStyle = 'rgba(139,147,167,0.13)';
    for (const [a, b] of this.payload.idle || []) {
      const xa = this._x(a, width);
      ctx.fillRect(xa, top, Math.max(this._x(b, width) - xa, 1), bottom - top);
    }
    ctx.fillStyle = 'rgba(248,113,113,0.30)';
    for (const [a, b] of this.payload.contact || []) {
      const xa = this._x(a, width);
      ctx.fillRect(xa, top, Math.max(this._x(b, width) - xa, 2), bottom - top);
    }

    ctx.strokeStyle = line;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(this.pad.l, bottom + 0.5);
    ctx.lineTo(width - this.pad.r, bottom + 0.5);
    ctx.stroke();

    for (const { track } of this.visible) {
      const lo = Math.min(0, track.min);
      const span = Math.max(track.max - lo, 1e-9);
      ctx.strokeStyle = track.color;
      ctx.lineWidth = 1.3;
      ctx.globalAlpha = 0.9;
      ctx.beginPath();
      let pen = false, prevY = null;
      for (let k = 0; k < track.t.length; k++) {
        const v = track.y[k];
        if (v == null) { pen = false; prevY = null; continue; }
        const x = this._x(track.t[k], width);
        const y = bottom - ((v - lo) / span) * (bottom - top);
        if (!pen) { ctx.moveTo(x, y); pen = true; }
        else if (track.kind === 'step') { ctx.lineTo(x, prevY); ctx.lineTo(x, y); }
        else ctx.lineTo(x, y);
        prevY = y;
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    ctx.fillStyle = muted;
    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'right';
    ctx.fillText('max', this.pad.l - 5, top + 8);
    ctx.fillText('min', this.pad.l - 5, bottom);
    const ticks = width < 320 ? 2 : 6;
    for (let k = 0; k <= ticks; k++) {
      const t = (this.duration * k) / ticks;
      const x = this._x(t, width);
      ctx.strokeStyle = line;
      ctx.beginPath(); ctx.moveTo(x, bottom); ctx.lineTo(x, bottom + 3); ctx.stroke();
      const last = k === ticks;
      ctx.textAlign = last ? 'right' : 'center';
      ctx.fillText(t.toFixed(1) + (last ? ' s' : ''), last ? x + 2 : x, height - 3);
    }
    ctx.textAlign = 'left';
  }

  /** Per-frame layer: blit the static image, then the moving marks. */
  _paintAll() {
    if (this.combined) this._paint(this.combinedEntry);
    else for (const entry of this.visible) this._paint(entry);
  }

  _paint(entry) {
    const ctx = entry.ctx;
    const { width, height } = entry;
    if (!width) return;
    const top = this.pad.t, bottom = height - this.pad.b;
    const y = entry.track ? this._scale(entry).y : null;

    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(entry.bg, 0, 0, width, height);

    if (this.hover != null) {
      ctx.strokeStyle = css('--muted');
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.55;
      ctx.beginPath();
      const hx = this._x(this.hover, width);
      ctx.moveTo(hx, top); ctx.lineTo(hx, bottom);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    const x = this._x(this.cursor, width);
    const accent = css('--accent');
    ctx.strokeStyle = accent;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, top - 3); ctx.lineTo(x, bottom);
    ctx.stroke();

    if (!entry.track) return;                      // the combined chart has no dot
    const value = sampleAt(entry.track, this.cursor);
    if (value != null) {
      ctx.fillStyle = entry.track.color;
      ctx.beginPath();
      ctx.arc(x, y(value), 2.6, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = css('--bg');
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    entry.valueEl.textContent = value == null ? '–' : fmtValue(value);
  }
};

/** Nearest sample of `track` at time `t`, or null in a gap. */
function sampleAt(track, t) {
  if (!track.t.length) return null;
  let lo = 0, hi = track.t.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (track.t[mid] < t) lo = mid + 1; else hi = mid;
  }
  return track.y[lo];
}

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888';
}

function fmtAxis(v) {
  const a = Math.abs(v);
  if (a >= 1000) return v.toFixed(0);
  if (a >= 100) return v.toFixed(1);
  return v.toFixed(2);
}

function fmtValue(v) {
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toFixed(3);
}

/** #rrggbb -> rgba(), for the fill under a trace. */
function withAlpha(hex, alpha) {
  const value = String(hex).replace('#', '');
  if (value.length !== 6) return hex;
  const r = parseInt(value.slice(0, 2), 16);
  const g = parseInt(value.slice(2, 4), 16);
  const b = parseInt(value.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

window.SyncChart.sampleAt = sampleAt;
