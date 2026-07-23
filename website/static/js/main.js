/* ===== STICH demo — front-end logic (vanilla JS) ===== */

(() => {
  "use strict";

  // Shared AudioContext, created lazily (only for drawing waveforms).
  let audioCtx = null;
  const getCtx = () => (audioCtx ||= new (window.AudioContext || window.webkitAudioContext)());

  // Registry so only one player sounds at a time.
  const players = [];

  const NBARS = 96;
  const fmtTime = (s) => {
    if (!isFinite(s)) return "0:00";
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60).toString().padStart(2, "0");
    return `${m}:${sec}`;
  };

  // Decode an audio URL into a compact array of peak magnitudes.
  async function computePeaks(src, nbars = NBARS) {
    const resp = await fetch(src);
    const buf = await resp.arrayBuffer();
    const decoded = await getCtx().decodeAudioData(buf);
    const data = decoded.getChannelData(0);
    const block = Math.floor(data.length / nbars) || 1;
    const peaks = new Array(nbars).fill(0);
    let max = 0.0001;
    for (let i = 0; i < nbars; i++) {
      let peak = 0;
      const start = i * block;
      for (let j = 0; j < block && start + j < data.length; j++) {
        const v = Math.abs(data[start + j]);
        if (v > peak) peak = v;
      }
      peaks[i] = peak;
      if (peak > max) max = peak;
    }
    return peaks.map((p) => p / max);
  }

  class AudioPlayer {
    constructor(el) {
      this.el = el;
      this.label = el.dataset.label || "";
      this.peaks = null;
      this.audio = new Audio();
      this.audio.preload = "metadata";
      this._build();
      this._bind();
      players.push(this);
      if (el.dataset.src) this.setSrc(el.dataset.src);
    }

    _build() {
      this.el.innerHTML = `
        <button class="player__btn" type="button" aria-label="Play">▶</button>
        <div class="player__main">
          <div class="player__label">${this.label}</div>
          <canvas class="player__wave"></canvas>
        </div>
        <span class="player__time">0:00</span>`;
      this.btn = this.el.querySelector(".player__btn");
      this.canvas = this.el.querySelector(".player__wave");
      this.timeEl = this.el.querySelector(".player__time");
      this.accent = getComputedStyle(this.el).getPropertyValue("--accent").trim() || "#8b7bf0";
    }

    _bind() {
      this.btn.addEventListener("click", () => this.toggle());
      this.canvas.addEventListener("click", (e) => this._seek(e));
      this.audio.addEventListener("ended", () => this._onStop());
      this.audio.addEventListener("pause", () => this._onStop());
      this.audio.addEventListener("play", () => this._onPlay());
      this.audio.addEventListener("loadedmetadata", () => this._draw());
      this.audio.addEventListener("timeupdate", () => this._draw());
      window.addEventListener("resize", () => this._draw());
    }

    async setSrc(src) {
      this.audio.src = src;
      this.peaks = null;
      this._draw();
      try {
        this.peaks = await computePeaks(src);
      } catch {
        this.peaks = null; // fall back to a flat bar
      }
      this._draw();
    }

    toggle() {
      if (this.audio.paused) {
        players.forEach((p) => p !== this && !p.audio.paused && p.audio.pause());
        this.audio.play().catch(() => {});
      } else {
        this.audio.pause();
      }
    }

    _onPlay() { this.btn.textContent = "❚❚"; this.btn.setAttribute("aria-label", "Pause"); this._raf(); }
    _onStop() {
      this.btn.textContent = "▶";
      this.btn.setAttribute("aria-label", "Play");
      cancelAnimationFrame(this._rafId);
      this._draw();
    }
    _raf() {
      cancelAnimationFrame(this._rafId);
      const step = () => { this._draw(); this._rafId = requestAnimationFrame(step); };
      this._rafId = requestAnimationFrame(step);
    }

    _seek(e) {
      const dur = this.audio.duration;
      if (!isFinite(dur)) return;
      const rect = this.canvas.getBoundingClientRect();
      const frac = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
      this.audio.currentTime = frac * dur;
      this._draw();
    }

    _draw() {
      const c = this.canvas;
      const w = c.clientWidth;
      if (!w) return;
      const dpr = window.devicePixelRatio || 1;
      const h = c.clientHeight || 34;
      c.width = w * dpr;
      c.height = h * dpr;
      const ctx = c.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      const dur = this.audio.duration;
      const progress = isFinite(dur) && dur > 0 ? this.audio.currentTime / dur : 0;
      const peaks = this.peaks || new Array(NBARS).fill(0.25);
      const n = peaks.length;
      const gap = 2;
      const bw = Math.max(1, (w - gap * (n - 1)) / n);
      const mid = h / 2;

      for (let i = 0; i < n; i++) {
        const bh = Math.max(2, peaks[i] * (h - 2));
        const x = i * (bw + gap);
        const played = (i + 0.5) / n <= progress;
        ctx.fillStyle = played ? this.accent : "rgba(164,157,196,0.35)";
        ctx.fillRect(x, mid - bh / 2, bw, bh);
      }
      this.timeEl.textContent = fmtTime(this.audio.currentTime || 0);
    }
  }

  /* ===== Hero waveform animation ===== */
  function heroWave() {
    const canvas = document.getElementById("heroWave");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let t = 0;

    function frame() {
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth, h = canvas.clientHeight;
      canvas.width = w * dpr; canvas.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      const bars = Math.floor(w / 8);
      const bw = 4;
      for (let i = 0; i < bars; i++) {
        const x = i * (w / bars) + (w / bars - bw) / 2;
        const phase = i * 0.28;
        const amp = reduce
          ? 0.4 + 0.3 * Math.sin(phase)
          : 0.25 + 0.35 * (Math.sin(t + phase) * 0.5 + 0.5) +
            0.2 * Math.sin(t * 1.7 + phase * 0.6);
        const bh = Math.max(3, amp * h);
        const frac = i / bars;
        // blend violet -> gold across the width
        const r = Math.round(139 + (230 - 139) * frac);
        const g = Math.round(123 + (178 - 123) * frac);
        const b = Math.round(240 + (78 - 240) * frac);
        ctx.fillStyle = `rgba(${r},${g},${b},0.75)`;
        ctx.fillRect(x, (h - bh) / 2, bw, bh);
      }
      t += 0.05;
      if (!reduce) requestAnimationFrame(frame);
    }
    frame();
  }

  /* ===== Illustrative loss chart ===== */
  function lossChart() {
    const canvas = document.getElementById("lossChart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    const pad = 24;
    // Four decaying-with-noise series, matching the plotter's four curves.
    const seeds = [
      { color: "#e6b24e", start: 0.9, end: 0.32 },
      { color: "#8b7bf0", start: 0.85, end: 0.30 },
      { color: "#57c7a3", start: 1.0, end: 0.20 },
      { color: "#ef7d92", start: 0.95, end: 0.24 },
    ];
    const N = 40;
    const yOf = (v) => H - pad - v * (H - 2 * pad);
    const xOf = (i) => pad + (i / (N - 1)) * (W - 2 * pad);

    // grid
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 1;
    for (let g = 0; g <= 4; g++) {
      const y = pad + (g / 4) * (H - 2 * pad);
      ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(W - pad, y); ctx.stroke();
    }

    seeds.forEach((s, si) => {
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < N; i++) {
        const p = i / (N - 1);
        const base = s.start + (s.end - s.start) * Math.pow(p, 0.6);
        const noise = Math.sin(i * (1.3 + si * 0.4) + si) * 0.04 * (1 - p);
        const v = Math.max(0.02, base + noise);
        const x = xOf(i), y = yOf(v);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
    });
  }

  /* ===== Try-it upload flow ===== */
  function tryIt() {
    const form = document.getElementById("convertForm");
    if (!form) return;
    const fileInput = document.getElementById("fileInput");
    const dropzone = document.getElementById("dropzone");
    const dzText = document.getElementById("dropzoneText");
    const convertBtn = document.getElementById("convertBtn");
    const result = document.getElementById("result");
    const status = document.getElementById("resultStatus");
    const outWrap = document.getElementById("outputPlayer");
    const downloadLink = document.getElementById("downloadLink");

    const inputPlayer = new AudioPlayer(document.getElementById("inputPlayer"));
    const outputPlayer = new AudioPlayer(outWrap);
    let chosen = null;
    let lastOutUrl = null;

    function setFile(file) {
      if (!file) return;
      chosen = file;
      dzText.innerHTML = `<strong>${file.name}</strong> selected`;
      dropzone.classList.add("has-file");
      convertBtn.disabled = false;
    }

    fileInput.addEventListener("change", () => setFile(fileInput.files[0]));

    ["dragenter", "dragover"].forEach((ev) =>
      dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("is-drag"); }));
    ["dragleave", "drop"].forEach((ev) =>
      dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("is-drag"); }));
    dropzone.addEventListener("drop", (e) => {
      const f = e.dataTransfer.files[0];
      if (f) { fileInput.files = e.dataTransfer.files; setFile(f); }
    });

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      if (!chosen) return;

      const direction = form.querySelector('input[name="direction"]:checked').value;
      result.hidden = false;
      outWrap.hidden = true;
      downloadLink.hidden = true;
      status.className = "result-status is-loading";
      status.textContent = "Converting…";
      convertBtn.disabled = true;
      convertBtn.textContent = "Converting…";

      // Show the user's own upload immediately.
      inputPlayer.setSrc(URL.createObjectURL(chosen));

      try {
        const fd = new FormData();
        fd.append("audio", chosen);
        fd.append("direction", direction);
        const resp = await fetch("/api/convert", { method: "POST", body: fd });
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({}));
          throw new Error(err.error || `Server error (${resp.status})`);
        }
        const blob = await resp.blob();
        if (lastOutUrl) URL.revokeObjectURL(lastOutUrl);
        lastOutUrl = URL.createObjectURL(blob);

        status.className = "result-status";
        status.textContent = "Done (placeholder audio).";
        outWrap.hidden = false;
        outputPlayer.setSrc(lastOutUrl);
        downloadLink.href = lastOutUrl;
        downloadLink.hidden = false;
      } catch (err) {
        status.className = "result-status is-error";
        status.textContent = `⚠ ${err.message}`;
      } finally {
        convertBtn.disabled = false;
        convertBtn.textContent = "Convert";
      }
    });
  }

  /* ===== Boot ===== */
  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll(".gallery .player[data-src]").forEach((el) => new AudioPlayer(el));
    heroWave();
    lossChart();
    tryIt();
  });
})();
