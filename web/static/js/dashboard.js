/* Energy Terminal dashboard */
(() => {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const charts = {};
  const PALETTE = ["#00c8ff", "#f59e0b", "#00d4aa", "#ff4976", "#a78bfa", "#94a3b8"];

  // ── utils ────────────────────────────────────────────────────────
  const api = async (url, opts) => {
    const r = await fetch(url, opts);
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.error || r.statusText);
    return body;
  };
  const post = (url, data) =>
    api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data || {}) });
  const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d }));
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const localTime = (iso) => new Date(iso).toLocaleString("en-GB", { timeZone: "Europe/London", day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
  const ago = (iso) => {
    if (!iso) return "—";
    const sec = Math.round((Date.now() - new Date(iso)) / 1000);
    const m = Math.round(sec / 60);
    if (m < -1) return "future dates";  // e.g. bank holidays and weather forecasts published ahead
    if (sec < 60) return `${Math.max(sec, 0)}s ago`;
    return m < 60 ? `${m}m ago` : m < 2880 ? `${Math.round(m / 60)}h ago` : `${Math.round(m / 1440)}d ago`;
  };
  const dur = (sec) => sec == null ? "" : sec < 90 ? `${Math.round(sec)}s` : sec < 5400 ? `${Math.round(sec / 60)}m` : `${(sec / 3600).toFixed(1)}h`;
  // Relative times that keep ticking: <span data-ago=iso> and <span data-elapsed=iso> are
  // re-rendered every second, so "3m ago" / run durations stay current without a reload.
  const agoSpan = (iso) => iso ? `<span data-ago="${esc(iso)}">${ago(iso)}</span>` : "—";
  const elapsedSpan = (iso) => `<span data-elapsed="${esc(iso)}">${dur((Date.now() - new Date(iso)) / 1000)}</span>`;
  setInterval(() => {
    document.querySelectorAll("[data-ago]").forEach((el) => { el.textContent = ago(el.dataset.ago); });
    document.querySelectorAll("[data-elapsed]").forEach((el) => { el.textContent = dur((Date.now() - new Date(el.dataset.elapsed)) / 1000); });
  }, 1000);
  const toast = (msg) => {
    const t = $("#toast"); t.textContent = msg; t.classList.add("show");
    clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove("show"), 3000);
  };
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  function lineChart(id, labels, datasets) {
    const ctx = $(`#${id}`);
    if (charts[id]) charts[id].destroy();
    const grid = css("--border-subtle"), text = css("--text-secondary");
    charts[id] = new Chart(ctx, {
      type: "line",
      data: { labels, datasets: datasets.map((d, i) => ({
        borderColor: d.color || PALETTE[i % PALETTE.length], backgroundColor: "transparent",
        borderWidth: d.width || 1.6, pointRadius: 0, tension: 0.25, spanGaps: false, ...d })) },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        interaction: { mode: "index", intersect: false },
        plugins: { legend: { labels: { color: text, font: { family: "JetBrains Mono", size: 11 }, boxWidth: 12 } } },
        scales: {
          x: { ticks: { color: text, maxTicksLimit: 10, font: { family: "JetBrains Mono", size: 10 } }, grid: { color: grid } },
          y: { ticks: { color: text, font: { family: "JetBrains Mono", size: 10 } }, grid: { color: grid } },
        },
      },
    });
  }

  // merge several [{timestamp, key}] series onto one sorted label axis
  // Merge several [{timestamp, key}] series onto one continuous half-hourly axis. Periods with
  // no data stay on the axis as gaps, so a missing day shows up as a gap instead of vanishing.
  const STEP_MS = 30 * 60 * 1000;
  function align(seriesList) {
    const all = seriesList.flatMap((s) => s.points.map((p) => Date.parse(p.timestamp))).filter((t) => !Number.isNaN(t));
    if (!all.length) return { labels: [], data: seriesList.map(() => []) };
    const lo = Math.min(...all), hi = Math.max(...all);
    const n = Math.min(Math.round((hi - lo) / STEP_MS) + 1, 20000);
    const labels = Array.from({ length: n }, (_, i) => localTime(new Date(lo + i * STEP_MS).toISOString()));
    const data = seriesList.map((s) => {
      const arr = new Array(n).fill(null);
      s.points.forEach((p) => {
        const i = Math.round((Date.parse(p.timestamp) - lo) / STEP_MS);
        if (i >= 0 && i < n) arr[i] = p[s.key];
      });
      return arr;
    });
    return { labels, data };
  }

  // ── tabs ─────────────────────────────────────────────────────────
  const loaders = {};
  $$(".sidebar-nav-item").forEach((b) => b.addEventListener("click", () => {
    $$(".sidebar-nav-item").forEach((x) => x.classList.toggle("active", x === b));
    $$(".tab-pane").forEach((p) => p.classList.toggle("active", p.id === `tab-${b.dataset.tab}`));
    loaders[b.dataset.tab]?.();
  }));

  // ── header/status ────────────────────────────────────────────────
  async function loadStatus() {
    const s = await api("/api/status");
    $("#header-region").textContent = s.region;
    $("#header-last-run").innerHTML = s.last_run ? `${esc(s.last_run.status?.toUpperCase() || "?")} · ${agoSpan(s.last_run.finished || s.last_run.started)}` : "never";
    const champs = Object.entries(s.champions || {});
    $("#champions").innerHTML = champs.length ? champs.map(([t, c]) => c ? `
      <div class="kpi-block">
        <div class="kpi-block-label">${esc(t.toUpperCase())}</div>
        <div class="kpi-block-value">${esc(c.model)}</div>
        <div class="kpi-block-sub">MAE ${fmt(c.metrics?.mae)} · sMAPE ${fmt(c.metrics?.smape, 1)}% · R² ${fmt(c.metrics?.r2)}</div>
        ${c.metrics?.weights ? `<div class="kpi-block-sub muted">${Object.entries(c.metrics.weights).map(([m, w]) => `${esc(m)} ${fmt(w, 2)}`).join(" · ")}</div>` : ""}
      </div>` : `<div class="kpi-block"><div class="kpi-block-label">${esc(t.toUpperCase())}</div><div class="kpi-block-sub muted">no backtest yet</div></div>`).join("")
      : `<div class="bento-empty">Run a backtest to pick champions</div>`;
    renderJob(s.jobs);
  }

  let lastJobFull = null;
  function renderJob(jobs) {
    const cur = jobs?.current;
    lastJobFull = cur || lastJobFull;
    if (cur?.job === "nightly" && activeTab() === "schedule") renderScheduleRun(cur);
    const pill = $("#job-status");
    pill.className = "status-pill" + (cur?.status === "running" ? " running" : cur?.status === "error" ? " error" : "");
    pill.textContent = cur?.status === "running" ? `● ${cur.job.toUpperCase()}…` : cur?.status === "error" ? `● ${cur.job.toUpperCase()} FAILED`
      : cur?.status === "cancelled" ? `● ${cur.job.toUpperCase()} CANCELLED` : "● IDLE";
    $$(".btn-job[data-job]").forEach((b) => (b.disabled = cur?.status === "running"));
    $("#bt-go").disabled = $("#pr-run").disabled = $("#sc-run-now").disabled = cur?.status === "running";
    $$("[data-fetch-now]").forEach((b) => (b.disabled = cur?.status === "running"));
    $("#job-cancel").hidden = cur?.status !== "running";
    if (cur) $("#job-output").textContent = JSON.stringify(cur, null, 2);
    if (cur?.job === "ingest" && cur.status === "running") {
      setFetchStatus(cur.progress ? `Fetching… ${cur.progress.done}/${cur.progress.total} sources` : "Fetching…", "running");
    }
    if (cur?.job === "predict" && cur.status === "running" && cur.progress) {
      setStatus("#pr-status", `${cur.progress.label[0].toUpperCase()}${cur.progress.label.slice(1)}…`, "running");
    }
    if (cur?.job === "backtest" && cur.status === "running") {
      const pg = cur.progress;
      $("#bt-progress").hidden = false;
      $("#bt-progress-bar").style.width = pg ? `${Math.round((pg.done / Math.max(1, pg.total)) * 100)}%` : "3%";
      setStatus("#bt-status", pg ? `Running ${pg.done}/${pg.total}: ${pg.label}` : "Building features…", "running");
    }
  }

  // ── overview: forecast ───────────────────────────────────────────
  async function loadForecast(keepDate) {
    const target = $("#fc-target").value;
    const date = keepDate ? $("#fc-date").value : "";
    const f = await api(`/api/forecast?target=${encodeURIComponent(target)}${date ? `&date=${date}` : ""}`);
    if (!keepDate) $("#fc-date").innerHTML = (f.dates || []).map((d) => `<option ${d === f.date ? "selected" : ""}>${d}</option>`).join("");
    // Previous predictions: the forecasts made for the days before this one (today included), so the
    // line runs on from them into this forecast instead of leaving a gap after the last actuals.
    let prev = null;
    if ($("#fc-show-prev").checked && f.date) {
      prev = await api(`/api/forecast/history?target=${encodeURIComponent(target)}&date=${f.date}&days=${$("#fc-prev-days").value}&combine=1`);
    }
    const prevPts = prev?.series?.previous || [];
    const actualPts = [...new Map([...(prev?.actual || []), ...(f.actual || [])].map((a) => [a.timestamp, a])).values()];
    const { labels, data } = align([{ points: actualPts, key: "y" }, { points: prevPts, key: "y_pred" }, { points: f.pred || [], key: "y_pred" }]);
    const sets = [{ label: "actual", data: data[0], color: css("--text-secondary") }];
    if (prev) {
      const mae = prev.mae?.previous;
      sets.push({ label: `previous forecasts${mae != null ? ` (MAE ${fmt(mae)})` : ""}`, data: data[1], color: css("--amber"), width: 1.6, borderDash: [5, 4] });
    }
    sets.push({ label: `forecast (${f.model || "-"})`, data: data[2], color: css("--cyan"), width: 2.2 });
    lineChart("fc-chart", labels, sets);
    const used = Object.entries(prev?.models_by_day || {});
    $("#fc-model").textContent = (f.model || "no predictions") + (prev ? (used.length
      ? ` · earlier: ${used.map(([d, m]) => `${d.slice(5)} ${m}`).join(", ")}${prev.hindsight_days?.previous ? ` (${prev.hindsight_days.previous} in hindsight)` : ""}`
      : " · no earlier predictions") : "");
  }

  async function loadRecommendations() {
    const r = await api("/api/recommendations");
    $("#rec-date").textContent = r.delivery_date || "--";
    const prods = r.products || [];
    $("#products-table").innerHTML = prods.length ? `
      <tr><th>PRODUCT</th><th>PRED MEAN</th><th>7D MEAN</th><th>Δ%</th><th>SPREAD</th><th>S/N</th><th>VIEW</th></tr>
      ${prods.map((p) => `<tr>
        <td>${esc(p.target)}</td><td>${fmt(p.pred_mean)}</td><td>${fmt(p.recent_7d_mean)}</td>
        <td class="${p.expected_change_pct < 0 ? "up" : "down"}">${fmt(p.expected_change_pct, 1)}</td>
        <td>${fmt(p.pred_spread)}</td><td>${fmt(p.signal_to_noise)}</td>
        <td><span class="tag ${p.view.includes("buy") ? "buy" : p.view.includes("sell") ? "sell" : ""}">${esc(p.view)}</span></td></tr>`).join("")}`
      : `<tr><td class="bento-empty">No price predictions yet</td></tr>`;
    const byTarget = {};
    (r.windows || []).forEach((w) => ((byTarget[w.target] ||= { buy: [], sell: [] })[w.action].push(w)));
    $("#windows").innerHTML = Object.keys(byTarget).length ? Object.entries(byTarget).map(([t, g]) => `
      <div class="window-col"><h4>${esc(t.toUpperCase())} · CHEAPEST</h4>
        ${g.buy.map((w) => `<div class="window-chip"><span class="up">${esc(w.local_time)}</span><span>${fmt(w.pred)}</span></div>`).join("")}</div>
      <div class="window-col"><h4>${esc(t.toUpperCase())} · DEAREST</h4>
        ${g.sell.map((w) => `<div class="window-chip"><span class="down">${esc(w.local_time)}</span><span>${fmt(w.pred)}</span></div>`).join("")}</div>`).join("")
      : `<div class="bento-empty">—</div>`;
  }

  loaders.overview = () => Promise.all([loadStatus(), loadForecast(), loadRecommendations()]).catch((e) => toast(e.message));
  $("#fc-target").addEventListener("change", () => loadForecast());
  $("#fc-date").addEventListener("change", () => loadForecast(true));
  $("#fc-show-prev").addEventListener("change", () => {
    $("#fc-prev-days").hidden = !$("#fc-show-prev").checked;
    try { localStorage.setItem("energy-fc-show-prev", $("#fc-show-prev").checked ? "1" : "0"); } catch { /* ignore */ }
    loadForecast(true).catch((e) => toast(e.message));
  });
  $("#fc-prev-days").addEventListener("change", () => loadForecast(true).catch((e) => toast(e.message)));
  try {
    if (localStorage.getItem("energy-fc-show-prev") !== "0") { $("#fc-show-prev").checked = true; $("#fc-prev-days").hidden = false; }
  } catch { /* ignore */ }

  // ── jobs ─────────────────────────────────────────────────────────
  $$(".btn-job[data-job]").forEach((b) => b.addEventListener("click", async () => {
    try {
      await post(`/api/jobs/${b.dataset.job}`, {});
      toast(`${b.dataset.job} started`);
      kick();
    } catch (e) { toast(e.message); }
  }));
  // ── live updates ─────────────────────────────────────────────────
  // One heartbeat drives everything: job progress, job completion, and data changes made
  // anywhere (dashboard jobs, CLI, nightly timer). Polls every 2s while a job runs, else 5s.
  let lastJob = null, lastVersion = null, hbTimer = null, hbBusy = false;
  const activeTab = () => $(".sidebar-nav-item.active")?.dataset.tab;
  const AUTO_REFRESH = new Set(["overview", "predict", "backtest", "schedule", "trading", "data", "runs"]);  // not config: don't wipe edits

  function refreshActive() {
    loadStatus().catch(() => {});
    const tab = activeTab();
    if (AUTO_REFRESH.has(tab) && tab !== "overview") loaders[tab]?.({ auto: true });
    else if (tab === "overview") Promise.all([loadForecast(), loadRecommendations()]).catch(() => {});
  }

  async function heartbeat() {
    if (hbBusy) return;
    hbBusy = true;
    let running = false;
    try {
      const hb = await api("/api/heartbeat");
      const job = hb.job;
      running = job?.status === "running";
      const finished = job && !running && lastJob !== null &&
        ((lastJob?.id === job.id && lastJob.status === "running") || (lastJob?.id !== job.id));
      if (running) renderJob({ current: job });
      if (finished) {
        const full = await api("/api/jobs");   // includes result / error
        renderJob(full);
        if (job.trigger !== "refresh" || full.current?.status !== "ok") toast(`${full.current?.job} ${full.current?.status}`);
        onJobDone(full.current);
        refreshActive();
      } else if (lastJob === null && job) {
        renderJob(await api("/api/jobs"));
      }
      if (lastVersion !== null && hb.version !== lastVersion && !running && !finished) refreshActive();
      // a running pipeline saves its record after every step: keep the Schedule timeline in sync
      else if (lastVersion !== null && hb.version !== lastVersion && running) {
        loadStatus().catch(() => {});  // header "last run"
        if (activeTab() === "schedule") loaders.schedule({ auto: true });
        if (activeTab() === "trading") loaders.trading({ auto: true });  // trading runs alongside jobs
      }
      onTradingBeat(hb.trading);
      lastJob = job || {};
      lastVersion = hb.version;
    } catch { /* server restarting; try again next tick */ }
    finally {
      hbBusy = false;
      clearTimeout(hbTimer);
      hbTimer = setTimeout(heartbeat, running ? 2000 : 5000);
    }
  }
  const kick = () => { clearTimeout(hbTimer); setTimeout(heartbeat, 300); };

  // ── predict ──────────────────────────────────────────────────────
  let predictOpts = null;
  const predictView = { date: null, target: null };

  function updateDateHint() {
    const d = $("#pr-date").value;
    const chosen = $$("#pr-targets input:checked").map((el) => el.value);
    const tooFar = (predictOpts?.targets || []).filter((t) => chosen.includes(t.name) && (!t.max_date || d > t.max_date));
    const hint = $("#pr-date-hint");
    if (!predictOpts) return;
    if (tooFar.length && $("#pr-refresh")?.checked) {
      hint.className = "field-hint";
      hint.textContent = `Stored data only reaches ${tooFar.map((t) => t.max_date || "nothing").sort()[0]} — the latest data will be fetched before predicting.`;
    } else if (tooFar.length) {
      hint.className = "field-hint warn";
      hint.textContent = tooFar.map((t) => t.max_date ? `${t.name}: latest predictable day is ${t.max_date}` : `${t.name}: no data yet`).join("\n");
    } else {
      const maxes = predictOpts.targets.filter((t) => chosen.includes(t.name)).map((t) => t.max_date).sort();
      hint.className = "field-hint";
      const past = d && d < predictOpts.default_date;
      hint.textContent = maxes.length ? `Data supports predictions up to ${maxes[0]}.` +
        (past ? " Past date: the model has already seen this day, so the error vs actuals is optimistic — use BACKTEST for honest accuracy." : "") : "";
    }
  }

  async function loadPredictOptions() {
    predictOpts = await api("/api/predict/options");
    if (!$("#pr-date").value) $("#pr-date").value = predictOpts.default_date;
    const prev = new Set($$("#pr-targets input:checked").map((el) => el.value));
    $("#pr-targets").innerHTML = predictOpts.targets.map((t) => `
      <label><input type="checkbox" value="${esc(t.name)}" ${!prev.size || prev.has(t.name) ? "checked" : ""}>
        ${esc(t.name)}<span class="muted">${t.champion ? "champ: " + esc(t.champion) : "no backtest yet"}</span></label>`).join("");
    const cur = $("#pr-model").value || "champion";
    $("#pr-model").innerHTML = `<option value="champion">champion (per target)</option>` +
      predictOpts.models.map((m) => `<option value="${esc(m)}">${esc(m)}</option>`).join("");
    $("#pr-model").value = cur;
    $("#pr-view-target").innerHTML = predictOpts.targets.map((t) => `<option>${esc(t.name)}</option>`).join("");
    if (predictView.target) $("#pr-view-target").value = predictView.target;
    $$("#pr-targets input").forEach((el) => el.addEventListener("change", updateDateHint));
    updateDateHint();
  }

  async function loadPredictResult(date, target) {
    if (date) predictView.date = date;
    if (target) { predictView.target = target; $("#pr-view-target").value = target; }
    predictView.target = $("#pr-view-target").value || predictView.target;
    if (!predictView.target) return;
    const q = `target=${encodeURIComponent(predictView.target)}${predictView.date ? `&date=${predictView.date}` : ""}`;
    const f = await api(`/api/forecast?${q}`);
    predictView.date = f.date || predictView.date;
    $("#pr-view-date").textContent = predictView.date || "--";
    $("#pr-view-models").textContent = (f.models || []).length ? f.models.join(" · ") : "no predictions";
    const models = Object.entries(f.series || {});
    const showPrev = $("#pr-show-prev").checked && predictView.date;
    let prev = null;
    if (showPrev) {
      prev = await api(`/api/forecast/history?target=${encodeURIComponent(predictView.target)}&date=${predictView.date}&days=${$("#pr-prev-days").value}`);
    }
    const prevModels = Object.entries(prev?.series || {});
    // actuals: the history window when comparing, else the 2 days shown with the forecast
    const actualPts = [...new Map([...(prev?.actual || []), ...(f.actual || [])].map((a) => [a.timestamp, a])).values()];
    const { labels, data } = align([
      { points: actualPts, key: "y" },
      ...prevModels.map(([, pts]) => ({ points: pts, key: "y_pred" })),
      ...models.map(([, pts]) => ({ points: pts, key: "y_pred" })),
    ]);
    // same colour per model; previous predictions are drawn dashed
    const allModels = [...new Set([...models.map(([m]) => m), ...prevModels.map(([m]) => m)])].sort();
    const colorOf = (m) => PALETTE[allModels.indexOf(m) % (PALETTE.length - 1)];
    lineChart("pr-chart", labels, [
      { label: "actual", data: data[0], color: css("--text-secondary"), width: 1.6 },
      ...prevModels.map(([m], i) => ({
        label: `previous: ${m}` + (prev.mae?.[m] != null ? ` (MAE ${fmt(prev.mae[m])})` : ""),
        data: data[i + 1], color: colorOf(m), width: 1.4, borderDash: [5, 4] })),
      ...models.map(([m], i) => ({ label: `${predictView.date}: ${m}`, data: data[i + 1 + prevModels.length], color: colorOf(m), width: 2.2 })),
    ]);
    const badge = $("#pr-prev-badge");
    if (prev && prevModels.length) {
      badge.hidden = false;
      badge.textContent = prevModels.map(([m]) => `${m}: MAE ${fmt(prev.mae?.[m])} over ${prev.genuine_days[m] + prev.hindsight_days[m]}d` +
        (prev.hindsight_days[m] ? ` (${prev.hindsight_days[m]} in hindsight)` : "")).join(" · ");
    } else if (prev) {
      badge.hidden = false;
      badge.textContent = `no earlier predictions in the last ${$("#pr-prev-days").value} days`;
    } else {
      badge.hidden = true;
    }
    // summary for every target on this date
    const rows = [];
    for (const t of predictOpts?.targets || []) {
      const ft = t.name === predictView.target ? f : await api(`/api/forecast?target=${encodeURIComponent(t.name)}&date=${predictView.date}`);
      const actual = (ft.actual || []).filter((a) => (ft.pred || []).some((p) => p.timestamp === a.timestamp));
      for (const [m, pts] of Object.entries(ft.series || {})) {
        const vals = pts.map((p) => p.y_pred);
        const byTs = new Map(pts.map((p) => [p.timestamp, p.y_pred]));
        const errs = actual.map((a) => Math.abs((byTs.get(a.timestamp) ?? NaN) - a.y)).filter((e) => !Number.isNaN(e));
        const minP = pts.reduce((a, b) => (b.y_pred < a.y_pred ? b : a), pts[0]);
        const maxP = pts.reduce((a, b) => (b.y_pred > a.y_pred ? b : a), pts[0]);
        rows.push({ target: t.name, model: m, mean: vals.reduce((a, b) => a + b, 0) / vals.length,
          min: minP?.y_pred, minAt: minP?.timestamp, max: maxP?.y_pred, maxAt: maxP?.timestamp,
          mae: errs.length ? errs.reduce((a, b) => a + b, 0) / errs.length : null, issued: ft.issued?.[m] });
      }
    }
    const hhmm = (iso) => iso ? new Date(iso).toLocaleTimeString("en-GB", { timeZone: "Europe/London", hour: "2-digit", minute: "2-digit" }) : "—";
    $("#pr-summary").innerHTML = rows.length ? `
      <tr><th>TARGET</th><th>MODEL</th><th>MEAN</th><th>LOW</th><th>AT</th><th>HIGH</th><th>AT</th><th>MAE</th></tr>
      ${rows.map((r) => `<tr><td>${esc(r.target)}</td><td>${esc(r.model)}</td><td>${fmt(r.mean)}</td>
        <td class="up">${fmt(r.min)}</td><td>${hhmm(r.minAt)}</td><td class="down">${fmt(r.max)}</td><td>${hhmm(r.maxAt)}</td>
        <td>${r.mae == null ? '<span class="muted">pending</span>' : fmt(r.mae)}</td></tr>`).join("")}`
      : `<tr><td class="bento-empty">No predictions for ${esc(predictView.date || "this date")} yet</td></tr>`;
    loadPredictHistory();
  }

  async function loadPredictHistory() {
    const idx = await api("/api/predictions/index");
    $("#pr-history").innerHTML = idx.length ? `
      <tr><th>DELIVERY DATE</th><th>TARGET</th><th>MODEL</th><th>MEAN</th><th>ISSUED</th></tr>
      ${idx.map((r) => `<tr data-date="${esc(r.delivery_date)}" data-target="${esc(r.target)}"
          class="${r.delivery_date === predictView.date && r.target === predictView.target ? "selected" : ""}">
        <td>${esc(r.delivery_date)}</td><td>${esc(r.target)}</td><td>${esc(r.model)}</td><td>${fmt(r.mean)}</td>
        <td>${r.issued_at ? localTime(r.issued_at) : "—"}</td></tr>`).join("")}`
      : `<tr><td class="bento-empty">No predictions yet</td></tr>`;
    $$("#pr-history tr[data-date]").forEach((tr) => tr.addEventListener("click", () => loadPredictResult(tr.dataset.date, tr.dataset.target)));
  }

  function setStatus(sel, text, cls) {
    const el = $(sel); el.textContent = text; el.className = `pr-status ${cls || ""}`;
  }
  const setPredictStatus = (text, cls) => setStatus("#pr-status", text, cls);

  $("#pr-run").addEventListener("click", async () => {
    const targets = $$("#pr-targets input:checked").map((el) => el.value);
    if (!targets.length) { setPredictStatus("Pick at least one target.", "error"); return; }
    const body = { date: $("#pr-date").value, targets, model: $("#pr-model").value,
      retrain: $("#pr-retrain").checked, refresh: $("#pr-refresh").checked };
    try {
      await post("/api/jobs/predict", body);
      setPredictStatus(`${body.refresh ? "Fetching latest data, then predicting" : "Predicting"} ${body.date}${body.model !== "champion" ? ` with ${body.model}` : ""}${body.retrain ? " (retraining first)" : ""}…`, "running");
      predictView.date = body.date;
      kick();
    } catch (e) { setPredictStatus(e.message, "error"); }
  });
  $("#pr-date").addEventListener("change", updateDateHint);
  $("#pr-refresh").addEventListener("change", updateDateHint);
  $("#pr-view-target").addEventListener("change", () => loadPredictResult());
  $("#pr-show-prev").addEventListener("change", () => {
    $("#pr-prev-days").hidden = !$("#pr-show-prev").checked;
    try { localStorage.setItem("energy-show-prev", $("#pr-show-prev").checked ? "1" : "0"); } catch { /* ignore */ }
    loadPredictResult();
  });
  $("#pr-prev-days").addEventListener("change", () => loadPredictResult());
  try {
    if (localStorage.getItem("energy-show-prev") === "1") { $("#pr-show-prev").checked = true; $("#pr-prev-days").hidden = false; }
  } catch { /* ignore */ }

  function onJobDone(job) {
    if (job?.job === "backtest") return onBacktestDone(job);
    if (job?.job === "ingest") return onFetchDone(job);
    if (job?.job === "strategy") return onStrategyDone(job);
    if (job?.job === "nightly") {
      setStatus("#sc-status", job.status === "ok" ? `Pipeline ${job.result?.status === "ok" ? "finished" : "finished with errors"}.` : (job.error || "failed").split("\n")[0],
        job.status === "ok" && job.result?.status === "ok" ? "ok" : "error");
      return;
    }
    if (job?.job !== "predict") return;
    if (job.status === "ok") {
      const w = job.result?.warnings || [];
      setPredictStatus(`Done: ${(job.result?.summary || []).length} target(s) predicted for ${job.result?.date}.` + (w.length ? `\n⚠ ${w.join("\n⚠ ")}` : ""), w.length ? "running" : "ok");
      if (job.result?.summary?.length && !job.result.summary.some((s) => s.target === predictView.target)) predictView.target = job.result.summary[0].target;
      predictView.date = job.result?.date || predictView.date;
    } else {
      setPredictStatus((job.error || "failed").split("\n")[0], "error");
    }
  }

  loaders.predict = (o) => loadPredictOptions().then(() => loadPredictResult()).catch((e) => { if (!o?.auto) toast(e.message); });

  // ── backtest form ────────────────────────────────────────────────
  let btOpts = null;

  function btEstimate() {
    const t = $$("#bt-targets input:checked").length, m = $$("#bt-models input:checked").length;
    const days = Number($("#bt-days").value) || 0, every = Math.max(1, Number($("#bt-retrain").value) || 1);
    const fits = t * m * Math.ceil(days / every);
    $("#bt-estimate").textContent = fits ? `≈ ${fits} model fits (${t} target${t === 1 ? "" : "s"} × ${m} model${m === 1 ? "" : "s"} × ${Math.ceil(days / every)} retrains)` : "";
    const end = $("#bt-end").value;
    if (end && days && btOpts?.data_start) {
      const start = new Date(new Date(end).getTime() - (days - 1) * 864e5).toISOString().slice(0, 10);
      const trainFrom = new Date(new Date(start).getTime() - (Number($("#bt-train-days").value) || 0) * 864e5).toISOString().slice(0, 10);
      const short = trainFrom < btOpts.data_start;
      const hint = $("#bt-range-hint");
      hint.className = `field-hint${short ? " warn" : ""}`;
      hint.textContent = `Tests ${start} → ${end}. Data available ${btOpts.data_start} → ${btOpts.last_full_day}.` +
        (short ? ` Early test days will train on less than the full window (data starts ${btOpts.data_start}).` : "");
    }
  }

  async function loadBacktestForm() {
    btOpts = await api("/api/backtest/options");
    const d = btOpts.defaults;
    if (!$("#bt-days").value) $("#bt-days").value = d.days;
    if (!$("#bt-retrain").value) $("#bt-retrain").value = d.retrain_every_days;
    if (!$("#bt-train-days").value) $("#bt-train-days").value = d.train_days;
    if (btOpts.last_full_day) { $("#bt-end").max = btOpts.last_full_day; if (!$("#bt-end").value) $("#bt-end").value = btOpts.last_full_day; }
    if (btOpts.data_start) $("#bt-end").min = btOpts.data_start;
    if (!$("#bt-targets").children.length) {
      $("#bt-targets").innerHTML = btOpts.targets.map((t) => `<label><input type="checkbox" value="${esc(t)}" checked> ${esc(t)}</label>`).join("");
      $("#bt-models").innerHTML = btOpts.models.map((m) => `<label><input type="checkbox" value="${esc(m)}" ${btOpts.candidates.includes(m) ? "checked" : ""}> ${esc(m)}</label>`).join("");
      $$("#bt-form input").forEach((el) => el.addEventListener("input", btEstimate));
    }
    btEstimate();
  }

  $("#bt-go").addEventListener("click", async () => {
    const targets = $$("#bt-targets input:checked").map((el) => el.value);
    const models = $$("#bt-models input:checked").map((el) => el.value);
    if (!targets.length || !models.length) { setStatus("#bt-status", "Pick at least one target and one model.", "error"); return; }
    const body = { end: $("#bt-end").value, days: Number($("#bt-days").value), retrain_every_days: Number($("#bt-retrain").value),
      train_days: Number($("#bt-train-days").value), targets, models, set_champions: $("#bt-set-champions").checked };
    try {
      await post("/api/jobs/backtest", body);
      $("#bt-progress").hidden = false; $("#bt-progress-bar").style.width = "3%";
      setStatus("#bt-status", "Starting…", "running");
      kick();
    } catch (e) { setStatus("#bt-status", e.message, "error"); }
  });

  function onBacktestDone(job) {
    $("#bt-progress-bar").style.width = "100%";
    setTimeout(() => { $("#bt-progress").hidden = true; }, 800);
    if (job.status !== "ok") { setStatus("#bt-status", (job.error || "failed").split("\n")[0], "error"); return; }
    const r = job.result || {};
    const champs = Object.entries(r.champions || {}).map(([t, m]) => `${t}: ${m}`).join(", ");
    setStatus("#bt-status", `Done: ${r.run_id}` + (champs ? `\nChampions → ${champs}` : "\nChampions unchanged") +
      ((r.errors || []).length ? `\n⚠ ${r.errors.join("\n⚠ ")}` : ""), (r.errors || []).length ? "running" : "ok");
    btPendingRun = r.run_id;   // show this run when the backtest tab next refreshes
  }

  // ── backtest ─────────────────────────────────────────────────────
  let btPendingRun = null, btNewest = null;
  async function loadBacktests(keepRun) {
    let run = keepRun === true ? $("#bt-run").value : "";
    if (keepRun !== true) {
      // auto/tab refresh: jump to a run that just finished, else keep the selected run
      const cur = $("#bt-run").value;
      run = btPendingRun || (cur && cur !== btNewest ? cur : "");
      btPendingRun = null;
    }
    const b = await api(`/api/backtests${run ? `?run_id=${encodeURIComponent(run)}` : ""}`);
    btNewest = (b.runs || [])[0] || null;
    if (keepRun !== true) $("#bt-run").innerHTML = (b.runs || []).map((r) => `<option ${r === b.run_id ? "selected" : ""}>${esc(r)}</option>`).join("");
    const rows = b.leaderboard || [];
    const r0 = rows[0];
    const lastDay = r0?.end ? new Date(new Date(r0.end).getTime() - 3600e3).toISOString().slice(0, 10) : null;
    $("#bt-range").textContent = r0 ? `${r0.start?.slice(0, 10)} → ${lastDay}` +
      (r0.days ? ` · ${r0.days}d · retrain ${r0.retrain_every_days}d · window ${r0.train_days}d` : "") : "--";
    const best = {};
    rows.forEach((r) => { if (r.mae != null && r.model !== "ensemble" && (best[r.target] == null || r.mae < best[r.target])) best[r.target] = r.mae; });
    const hasChamp = rows.some((r) => r.champion === true);
    const isBest = (r) => (hasChamp ? r.champion === true : r.mae === best[r.target]);
    const hasHold = rows.some((r) => r.mae_holdout != null);
    $("#bt-table").innerHTML = rows.length ? `
      <tr><th>TARGET</th><th>MODEL</th><th>MAE</th>${hasHold ? '<th title="MAE on the second half of the window only: the days the ensemble is judged on">MAE 2ND HALF</th>' : ""}<th>RMSE</th><th>sMAPE%</th><th>BIAS</th><th>R²</th><th>INTRADAY ρ</th><th>CHEAP HIT</th><th>DEAR HIT</th><th>DIR ACC</th><th>NOTE</th></tr>
      ${rows.map((r) => `<tr class="${isBest(r) ? "best" : ""}" title="${isBest(r) ? "champion" : ""}">
        <td>${esc(r.target)}</td><td>${esc(r.model)}${r.champion === true ? " ★" : ""}</td><td>${fmt(r.mae)}</td>${hasHold ? `<td>${fmt(r.mae_holdout)}</td>` : ""}<td>${fmt(r.rmse)}</td><td>${fmt(r.smape, 1)}</td>
        <td>${fmt(r.bias)}</td><td>${fmt(r.r2)}</td><td>${fmt(r.intraday_spearman)}</td><td>${fmt(r.cheapest_hit_rate)}</td>
        <td>${fmt(r.dearest_hit_rate)}</td><td>${fmt(r.daily_direction_acc)}</td>
        <td class="${r.error ? "down" : "muted"}" title="${esc(r.error || r.note || "")}">${esc((r.error || r.note || "").slice(0, 90))}</td></tr>`).join("")}`
      : `<tr><td class="bento-empty">No backtests yet — press BACKTEST on the overview</td></tr>`;
    if (b.run_id) loadBacktestSeries(b.run_id);
  }
  async function loadBacktestSeries(run) {
    const target = $("#bt-target").value;
    const s = await api(`/api/backtest_series?run_id=${encodeURIComponent(run)}&target=${encodeURIComponent(target)}`);
    const models = Object.entries(s.models || {});
    const { labels, data } = align([{ points: s.actual || [], key: "y_true" }, ...models.map(([, pts]) => ({ points: pts, key: "y_pred" }))]);
    lineChart("bt-chart", labels, [
      { label: "actual", data: data[0], color: css("--text-secondary"), width: 2 },
      ...models.map(([m], i) => ({ label: m, data: data[i + 1], color: PALETTE[i % PALETTE.length], width: 1.3 })),
    ]);
  }
  loaders.backtest = (o) => Promise.all([loadBacktestForm(), loadBacktests()]).catch((e) => { if (!o?.auto) toast(e.message); });
  $("#bt-run").addEventListener("change", () => loadBacktests(true));
  $("#bt-target").addEventListener("change", () => loadBacktestSeries($("#bt-run").value));

  // ── fetch now (Data tab + Schedule tab) ─────────────────────────
  function setFetchStatus(text, cls, detail) {
    $$("[data-fetch-status]").forEach((el) => { el.textContent = text; el.className = `fetch-status ${cls || ""}`; el.title = detail || ""; });
  }
  // "`Import openpyxl` failed" / "No module named 'lxml'" -> the packages to pip install
  const missingPackages = (msgs) => [...new Set(msgs.flatMap((m) =>
    [...String(m || "").matchAll(/`Import (\w+)` failed|No module named '(\w+)'/g)].map((x) => x[1] || x[2])))];
  function onFetchDone(job) {
    if (job.status !== "ok") { setFetchStatus(`Fetch failed: ${(job.error || "").split("\n")[0]}`, "error"); return; }
    const res = job.result || [];
    const rows = res.reduce((a, r) => a + (r.rows || 0), 0);
    const bad = res.filter((r) => r.status === "error");
    const failed = bad.map((r) => r.source);
    const pkgs = missingPackages(bad.map((r) => r.message));
    const when = new Date().toLocaleTimeString("en-GB", { timeZone: "Europe/London", hour: "2-digit", minute: "2-digit" });
    setFetchStatus(`Fetched ${when} · ${rows.toLocaleString()} rows updated` + (failed.length ? ` · failed: ${failed.join(", ")}` : "") +
      (pkgs.length ? ` · missing package${pkgs.length > 1 ? "s" : ""}: pip install ${pkgs.join(" ")}` : ""),
      failed.length ? "error" : "ok", bad.map((r) => `${r.source}: ${r.message}`).join("\n"));
  }
  $$("[data-fetch-now]").forEach((b) => b.addEventListener("click", async () => {
    try {
      await post("/api/fetch-now");
      setFetchStatus("Fetching…", "running");
      kick();
    } catch (e) { setFetchStatus(e.message, "error"); }
  }));

  // ── schedule ─────────────────────────────────────────────────────
  let scData = null, scDirty = false;
  const STEP_LABEL = { ingest: "Ingest", tune: "Tune", backtest: "Backtest", train: "Train", predict: "Predict", strategy: "Strategy" };
  const until = (iso) => {
    const m = Math.round((new Date(iso) - Date.now()) / 60000);
    if (m <= 0) return "now";
    return m < 60 ? `in ${m}m` : `in ${Math.floor(m / 60)}h ${m % 60}m`;
  };

  function renderScheduleState() {
    if (!scData) return;
    const c = scData.config;
    const st = $("#sc-state");
    st.className = `sc-state${c.enabled ? " on" : ""}`;
    st.textContent = c.enabled ? `● ON · DAILY AT ${c.time}` : "○ STOPPED";
    $("#sc-next").textContent = c.enabled && scData.next_run
      ? `Next run ${new Date(scData.next_run).toLocaleString("en-GB", { timeZone: "Europe/London", weekday: "short", hour: "2-digit", minute: "2-digit" })} (${until(scData.next_run)})`
      : "Press START to run the pipeline every day at the time below.";
    $("#sc-refresh-on").checked = !!c.refresh_enabled;
    if (document.activeElement !== $("#sc-refresh-min")) $("#sc-refresh-min").value = c.refresh_minutes || 30;
    $("#sc-refresh-next").textContent = c.refresh_enabled
      ? `Last fetch ${c.last_refresh ? ago(c.last_refresh) : "never"}` + (scData.next_refresh ? ` · next ${until(scData.next_refresh)}` : "") + ". Runs whether or not the daily schedule is on."
      : "Off: data is only downloaded by the daily pipeline, so today's actuals appear after tomorrow's 02:30 run.";
    const btn = $("#sc-toggle");
    btn.textContent = c.enabled ? "STOP SCHEDULE" : "START SCHEDULE";
    btn.classList.toggle("stop", c.enabled);
  }

  // Which run the timeline shows (null = the running/latest one) and which step's log is open.
  let scView = null, scStep = null, scShown = null;
  const scLog = { key: null, next: 0, busy: false, gen: 0 };

  function renderPipeline(planned, results, live) {
    // results: [{step,status,seconds,error}] saved so far; live: {step,label} while running
    const byStep = Object.fromEntries((results || []).map((r) => [r.step, r]));
    const liveIdx = live ? planned.indexOf(live.step) : -1;
    $("#sc-pipeline").innerHTML = scData.all_steps.map((step, i) => {
      const inPlan = planned.includes(step);
      const r = byStep[step];
      let cls = inPlan ? "" : "skipped", state = inPlan ? "pending" : "skipped", detail = "";
      const pi = planned.indexOf(step);
      if (live && inPlan && pi === liveIdx) { cls = "running"; state = "● running"; detail = (live.label || "").replace(/^\w+:\s*/, ""); }
      else if (r && r.status !== "running") { cls = r.status === "ok" ? "ok" : "error"; state = r.status === "ok" ? "✓ done" : "✕ failed"; detail = r.status === "ok" ? dur(r.seconds) : (r.error || "").split("\n")[0]; }
      else if (r?.status === "running") { cls = "error"; state = "✕ stopped"; detail = "did not finish"; }
      else if (live && inPlan && pi < liveIdx) { cls = "ok"; state = "✓ done"; }
      const clickable = inPlan && (r || (live && pi <= liveIdx));
      if (clickable && step === scShown?.step) cls += " selected";
      return (i ? '<span class="pipe-arrow">→</span>' : "") +
        `<div class="pipe-step ${cls}" ${clickable ? `data-step="${step}"` : ""} title="${esc(detail || (clickable ? "Show this step's log" : ""))}"><div class="ps-name">${STEP_LABEL[step].toUpperCase()}</div>
         <div class="ps-state">${state}</div><div class="ps-detail">${esc(detail)}</div></div>`;
    }).join("");
  }

  // Default step to show: the one running, else the first that failed, else the last one run.
  function pickStep(planned, results, live) {
    const ran = (results || []).map((r) => r.step).filter((s) => STEP_LABEL[s]);
    if (scStep && (ran.includes(scStep) || (live && planned.indexOf(scStep) <= planned.indexOf(live.step)))) return scStep;
    if (live?.step && STEP_LABEL[live.step]) return live.step;
    const bad = (results || []).find((r) => r.status === "error" && STEP_LABEL[r.step]);
    return bad?.step || ran[ran.length - 1] || null;
  }

  function renderScheduleRun(job) {
    if (!scData) return;
    const runs = scData.runs || [];
    const running = job?.job === "nightly" && job.status === "running";
    const viewed = scView && runs.find((r) => r.run_id === scView);
    $("#sc-latest").hidden = !viewed;
    if (!viewed && running) {
      const runId = job.progress?.run_id;
      const rec = runs.find((r) => r.run_id === runId) || (runId ? { run_id: runId, steps: [] }
        : runs.find((r) => r.status === "running" && new Date(r.started) >= new Date(job.started) - 10000));  // not a stale one
      const planned = rec?.planned || scData.config.steps;
      const live = job.progress?.step && STEP_LABEL[job.progress.step] ? job.progress : { step: planned[0], label: "starting" };
      $("#sc-run-title").textContent = "RUNNING NOW";
      $("#sc-run-meta").innerHTML = `started ${agoSpan(job.started)} · running ${elapsedSpan(job.started)} · ${job.progress ? `step ${Math.min(job.progress.done + 1, job.progress.total)}/${job.progress.total}` : "starting"}`;
      scShown = rec ? { run: rec, step: pickStep(planned, rec.steps, live), live: live.step } : null;
      renderPipeline(planned, rec?.steps, live);
      $("#sc-run-detail").textContent = job.progress?.label || "starting…";
      loadStepLog();
      return;
    }
    const last = viewed || runs.find((r) => r.status !== "running") || runs[0];
    $("#sc-run-title").textContent = viewed ? `RUN · ${localTime(viewed.started)}` : "LATEST RUN";
    if (!last) {
      $("#sc-run-meta").textContent = "no runs yet";
      scShown = null;
      renderPipeline(scData.config.steps, null, null);
      $("#sc-run-detail").textContent = "Nothing has run yet. Press RUN NOW to try the pipeline.";
      loadStepLog();
      return;
    }
    const total = (last.steps || []).reduce((a, s) => a + (s.seconds || 0), 0);
    $("#sc-run-meta").textContent = `${last.status?.toUpperCase()} · ${last.trigger || "manual"} · ${localTime(last.started)} · ${dur(total)}`;
    const planned = last.planned || (last.steps || []).map((s) => s.step).filter((s) => STEP_LABEL[s]);
    scShown = { run: last, step: pickStep(planned, last.steps, null), live: null };
    renderPipeline(planned, last.steps, null);
    const errs = (last.steps || []).filter((s) => s.status !== "ok");
    $("#sc-run-detail").textContent = ["running", "interrupted"].includes(last.status) ? "This run did not finish (the server stopped or the job was cancelled)."
      : errs.length ? errs.map((e) => `${e.step}: ${e.error}`).join("\n\n")
      : `All steps finished.` + (Object.keys(last.champions || {}).length ? `\nChampions: ${Object.entries(last.champions).map(([t, m]) => `${t} → ${m}`).join(", ")}` : "");
    loadStepLog();
  }

  const logTime = (iso) => new Date(iso).toLocaleTimeString("en-GB", { timeZone: "Europe/London", hour12: false });
  function logLine(x) {
    return `<div class="log-line ${esc(x.level)}"><span class="lt">${logTime(x.t)}</span><span class="ll">${esc(x.level)}</span>` +
      `<span class="lg">${esc(x.logger)}</span><span class="lm">${esc(x.msg)}</span></div>`;
  }

  // Load (or, while the step is running, keep appending to) the selected step's log.
  async function loadStepLog() {
    const box = $("#sc-log");
    const sel = scShown;
    if (!sel?.step) {
      scLog.key = null;
      $("#sc-log-title").textContent = "LOG";
      $("#sc-log-meta").textContent = sel ? "This step has not started yet." : "Click a step above to see its log.";
      box.innerHTML = "";
      return;
    }
    const level = $("#sc-log-warn").checked ? "WARNING" : "INFO";
    const key = `${sel.run.run_id}|${sel.step}|${level}`;
    const fresh = key !== scLog.key;
    const following = sel.live === sel.step;  // step still running: poll for new lines
    if (!fresh && (!following || scLog.busy)) return;
    if (fresh) {
      scLog.key = key; scLog.next = 0; scLog.gen += 1;
      $("#sc-log-title").textContent = `LOG · ${STEP_LABEL[sel.step].toUpperCase()}`;
      box.innerHTML = `<div class="log-empty">loading…</div>`;
    }
    const gen = scLog.gen;
    scLog.busy = true;
    try {
      const r = await api(`/api/runs/${encodeURIComponent(sel.run.run_id)}/logs?step=${sel.step}&after=${scLog.next}&level=${level}`);
      if (gen !== scLog.gen) return;  // user picked another step meanwhile
      const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
      if (scLog.next === 0) box.innerHTML = "";
      box.insertAdjacentHTML("beforeend", r.lines.map(logLine).join(""));
      scLog.next = r.next;
      if (!r.total) box.innerHTML = `<div class="log-empty">${following ? "No log lines yet…" : level === "WARNING" ? "No warnings or errors." : "No log lines for this step."}</div>`;
      if (fresh || atBottom) box.scrollTop = box.scrollHeight;
      $("#sc-log-meta").textContent = `${r.total} line${r.total === 1 ? "" : "s"}` + (following ? " · live" : "") +
        (r.source === "jobs.log" ? " · from logs/jobs.log (run predates step logs)" : "");
    } catch (e) {
      if (gen === scLog.gen) { box.innerHTML = `<div class="log-empty">${esc(e.message)}</div>`; scLog.key = null; }
    } finally { scLog.busy = false; }
  }

  $("#sc-pipeline").addEventListener("click", (e) => {
    const el = e.target.closest(".pipe-step[data-step]");
    if (!el) return;
    scStep = el.dataset.step;
    renderScheduleRun(lastJobFull);
  });
  $("#sc-log-warn").addEventListener("change", () => loadStepLog());
  $("#sc-latest").addEventListener("click", () => { scView = null; scStep = null; renderScheduleRun(lastJobFull); markHistory(); });
  $("#sc-history").addEventListener("click", (e) => {
    const tr = e.target.closest("tr[data-run]");
    if (!tr) return;
    scView = tr.dataset.run; scStep = null;
    renderScheduleRun(lastJobFull); markHistory();
    $("#sc-pipeline").scrollIntoView({ behavior: "smooth", block: "nearest" });
  });
  function markHistory() {
    $$("#sc-history tr[data-run]").forEach((tr) => tr.classList.toggle("selected", tr.dataset.run === scShown?.run?.run_id));
  }

  async function loadSchedule() {
    scData = await api("/api/schedule");
    const c = scData.config;
    if (!scDirty) {
      $("#sc-time").value = c.time;
      $("#sc-bt-days").value = c.backtest_days ?? scData.default_backtest_days;
      $("#sc-tune-days").value = c.tune_days ?? scData.default_tune_days;
      $("#sc-steps").innerHTML = scData.all_steps.map((st) => `<label><input type="checkbox" value="${st}" ${c.steps.includes(st) ? "checked" : ""}> ${STEP_LABEL[st]}</label>`).join("");
      $$("#sc-steps input").forEach((el) => el.addEventListener("change", () => { scDirty = true; }));
    }
    renderScheduleState();
    renderScheduleRun(lastJobFull);
    $("#sc-history").innerHTML = scData.runs.length ? `
      <tr><th>STARTED</th><th>TRIGGER</th><th>STATUS</th><th>TOTAL</th><th>STEPS</th><th>CHAMPIONS</th></tr>
      ${scData.runs.map((r) => `<tr data-run="${esc(r.run_id)}" title="Show this run's steps and logs">
        <td>${localTime(r.started)} <span class="muted">· ${agoSpan(r.started)}</span></td><td>${esc(r.trigger || "manual")}</td>
        <td class="${r.status === "ok" ? "up" : r.status === "running" ? "" : "down"}">${esc(r.status)}</td>
        <td>${r.status === "running" && lastJobFull?.status === "running" && lastJobFull.job === "nightly" ? elapsedSpan(r.started)
          : dur((r.steps || []).reduce((a, s) => a + (s.seconds || 0), 0))}</td>
        <td>${(r.steps || []).map((s) => `<span class="tag ${s.status === "ok" ? "on" : s.status === "running" ? "" : "sell"}" title="${esc(s.error || "")}">${esc(s.step)} ${dur(s.seconds)}</span>`).join(" ")}</td>
        <td>${Object.entries(r.champions || {}).map(([t, m]) => `${esc(t)}: ${esc(m)}`).join(" · ")}</td></tr>`).join("")}`
      : `<tr><td class="bento-empty">No pipeline runs yet</td></tr>`;
    markHistory();
  }

  function scheduleForm() {
    return {
      time: $("#sc-time").value,
      steps: $$("#sc-steps input:checked").map((el) => el.value),
      backtest_days: $("#sc-bt-days").value ? Number($("#sc-bt-days").value) : null,
      tune_days: $("#sc-tune-days").value ? Number($("#sc-tune-days").value) : null,
    };
  }
  async function saveSchedule(extra) {
    const body = { ...scheduleForm(), ...(extra || {}) };
    if (!body.steps.length) { setStatus("#sc-status", "Pick at least one step.", "error"); return; }
    try {
      await post("/api/schedule", body);
      scDirty = false;
      await loadSchedule();
      setStatus("#sc-status", extra?.enabled === true ? `Schedule started: daily at ${body.time}.` : extra?.enabled === false ? "Schedule stopped." : "Saved.", "ok");
    } catch (e) { setStatus("#sc-status", e.message, "error"); }
  }
  ["#sc-time", "#sc-bt-days", "#sc-tune-days"].forEach((sel) => $(sel).addEventListener("input", () => { scDirty = true; }));
  async function saveRefresh() {
    try {
      await post("/api/schedule", { refresh_enabled: $("#sc-refresh-on").checked, refresh_minutes: Number($("#sc-refresh-min").value) || 30 });
      scData = await api("/api/schedule"); renderScheduleState();
      setStatus("#sc-status", $("#sc-refresh-on").checked ? `Fetching new data every ${$("#sc-refresh-min").value} min.` : "Background data fetch off.", "ok");
    } catch (e) { setStatus("#sc-status", e.message, "error"); }
  }
  $("#sc-refresh-on").addEventListener("change", saveRefresh);
  $("#sc-refresh-min").addEventListener("change", saveRefresh);
  $("#sc-toggle").addEventListener("click", () => saveSchedule({ enabled: !scData?.config.enabled }));
  $("#sc-save").addEventListener("click", () => saveSchedule());
  $("#sc-run-now").addEventListener("click", async () => {
    try {
      if (scDirty) await saveSchedule();
      await post("/api/schedule/run-now");
      setStatus("#sc-status", "Pipeline started.", "running");
      kick();
    } catch (e) { setStatus("#sc-status", e.message, "error"); }
  });
  setInterval(() => { if (activeTab() === "schedule") renderScheduleState(); }, 30000);
  loaders.schedule = (o) => loadSchedule().catch((e) => { if (!o?.auto) toast(e.message); });

  // ── trading (paper) ──────────────────────────────────────────────
  let trData = null, trDirty = false, trWasBusy = false, trShownCycle = null;
  const money = (v) => v == null ? "—" : `${v < 0 ? "−" : ""}£${fmt(Math.abs(v), 2)}`;
  const signed = (v, d = 2) => v == null ? "—" : `${v > 0 ? "+" : ""}${fmt(v, d)}`;
  const posText = (p) => !p ? "FLAT" : `${p > 0 ? "LONG" : "SHORT"} ${fmt(Math.abs(p), 2)}`;

  function onTradingBeat(t) {
    if (!t) return;
    if (t.busy) setStatus("#tr-status", "Trading cycle running…", "running");
    if (trWasBusy && !t.busy && activeTab() === "trading") loaders.trading({ auto: true });
    trWasBusy = t.busy;
  }

  async function loadTrading() {
    trData = await api("/api/trading/status");
    const t = trData, st = t.state, m = t.market;
    const st$ = $("#tr-state");
    st$.className = `sc-state${st.enabled ? " on" : ""}`;
    st$.textContent = st.enabled ? `● AUTO-TRADING · EVERY ${st.interval_minutes} MIN` : "○ AUTO-TRADING OFF";
    $("#tr-ig").innerHTML = t.ig_configured
      ? `IG demo login found${m ? ` · market <b>${esc(m.name || m.epic)}</b> (${esc(m.epic)}) · £${fmt(t.config.value_per_point, 2)} per point` : " · market not looked up yet (runs on first cycle)"}`
      : `No IG demo login: add IG_API_KEY, IG_USERNAME and IG_PASSWORD to <code>.env</code>. Until then prices come from stored candles only.`;
    $("#tr-market").textContent = m ? (m.name || m.epic).toUpperCase() : "MARKET";
    $("#tr-horizon").textContent = t.config.horizon;
    const ret = t.equity - t.starting_balance;
    $("#tr-kpis").innerHTML = [
      ["POSITION", posText(t.position), t.position ? `avg ${fmt(t.avg_price, 2)}` : ""],
      ["EQUITY", money(t.equity), `${signed(100 * ret / t.starting_balance, 2)}% vs ${money(t.starting_balance)}`],
      ["UNREALISED", money(t.unrealised), t.mid ? `mid ${fmt(t.mid, 2)}` : ""],
      ["REALISED", money(t.realised), `spread ${money(t.spread_cost)} · funding ${money(t.funding)}`],
    ].map(([l, v, sub]) => `<div class="kpi-block"><div class="kpi-block-label">${l}</div><div class="kpi-block-value">${v}</div><div class="kpi-block-sub">${esc(sub)}</div></div>`).join("");
    if (!trDirty) {
      $("#tr-enabled").checked = !!st.enabled;
      $("#tr-mirror").checked = !!st.mirror_ig;
      $("#tr-interval").value = st.interval_minutes || 60;
      $("#tr-size").value = st.size ?? t.config.size;
      $("#tr-k").value = st.threshold_k ?? t.config.threshold_k;
      if (!$("#tr-bt-days").value) $("#tr-bt-days").value = 90;
    }
    $("#tr-mirror").disabled = !t.ig_configured;
    const last = st.last_result;
    $("#tr-next").innerHTML = (st.enabled && t.next_cycle ? `Next cycle ${until(t.next_cycle)} (weekdays only). ` : "") +
      (st.last_cycle ? `Last cycle ${agoSpan(st.last_cycle)}: ${esc(last?.action || "?")}${last?.pred != null ? ` (predicted ${signed(last.pred)})` : ""}.` : "No cycles yet.") +
      (t.model ? ` Model ${esc(t.model.model)} trained ${agoSpan(t.model.trained_at)}.` : " No model yet: run BACKTEST + TRAIN.");
    const lastRun = t.runner?.last;
    if (!t.runner?.current && lastRun && lastRun.finished !== trShownCycle) {  // show each cycle's outcome once
      trShownCycle = lastRun.finished;
      setStatus("#tr-status", lastRun.status === "ok" ? `Cycle done: ${lastRun.result?.action}${(lastRun.result?.notes || []).length ? ` — ${lastRun.result.notes.join("; ")}` : ""}` : `Cycle failed: ${lastRun.error}`, lastRun.status === "ok" ? "ok" : "error");
    }
    await Promise.all([loadTradingChart(), loadTradingBacktests()]);
  }

  async function loadTradingChart() {
    const d = await api(`/api/trading/chart?days=${$("#tr-days").value}`);
    const c = d.candles;
    const labels = c.map((r) => localTime(r.ts));
    const at = (iso) => {  // index of the last candle at or before a time
      const t = Date.parse(iso); let idx = -1;
      for (let i = 0; i < c.length; i++) { if (Date.parse(c[i].ts) <= t) idx = i; else break; }
      return idx;
    };
    const marks = (side) => { const arr = new Array(c.length).fill(null); d.fills.filter((f) => f.side === side).forEach((f) => { const i = at(f.ts); if (i >= 0) arr[i] = f.price; }); return arr; };
    if (c.length) {
      const q = c[c.length - 1];
      $("#tr-quote").textContent = `${fmt(q.bid, 2)} / ${fmt(q.ask, 2)} · ${localTime(q.ts)}`;
    }
    lineChart("tr-chart", labels, [
      { label: "mid", data: c.map((r) => r.mid), color: PALETTE[0], width: 1.8 },
      { label: "bid", data: c.map((r) => r.bid), color: PALETTE[5], width: 0.8, borderDash: [3, 3] },
      { label: "ask", data: c.map((r) => r.ask), color: PALETTE[5], width: 0.8, borderDash: [3, 3] },
      { label: "buy", data: marks("BUY"), color: PALETTE[2], showLine: false, pointRadius: 6, pointStyle: "triangle", pointBackgroundColor: PALETTE[2] },
      { label: "sell", data: marks("SELL"), color: PALETTE[3], showLine: false, pointRadius: 6, pointStyle: "triangle", rotation: 180, pointBackgroundColor: PALETTE[3] },
    ]);
    const eq = d.equity;
    $("#tr-eq-badge").textContent = eq.length ? `${eq.length} marks` : "no cycles yet";
    lineChart("tr-equity", eq.map((e) => localTime(e.ts)), [
      { label: "equity", data: eq.map((e) => e.equity), color: PALETTE[2], width: 1.8, stepped: true },
    ]);
    $("#tr-fills").innerHTML = d.all_fills.length ? `<tr><th>TIME</th><th>SIDE</th><th>SIZE</th><th>PRICE</th><th>POSITION</th><th>REALISED</th><th>IG DEMO</th><th>WHY</th></tr>` +
      d.all_fills.slice().reverse().map((f) => `<tr><td>${localTime(f.ts)}</td><td class="${f.side === "BUY" ? "up" : "down"}">${f.side}</td><td>${fmt(f.size, 2)}</td><td>${fmt(f.price, 2)}</td>
        <td>${posText(f.position)}</td><td class="${f.realised > 0 ? "up" : f.realised < 0 ? "down" : ""}">${f.realised ? money(f.realised) : ""}</td>
        <td title="${esc(f.ig_reason || "")}">${f.ig_status ? `${esc(f.ig_status)}${f.ig_level ? ` @ ${fmt(f.ig_level, 2)}` : ""}` : "—"}</td><td>${esc(f.reason || "")}</td></tr>`).join("")
      : `<tr><td class="bento-empty">No paper trades yet</td></tr>`;
    $("#tr-decisions").innerHTML = d.decisions.length ? `<tr><th>TIME</th><th>PREDICTED</th><th>THRESHOLD</th><th>POSITION → TARGET</th><th>BID / ASK</th><th>ACTION</th></tr>` +
      d.decisions.slice().reverse().map((x) => `<tr title="${esc(x.note || "")}"><td>${localTime(x.ts)}</td><td class="${x.pred > 0 ? "up" : x.pred < 0 ? "down" : ""}">${signed(x.pred)}</td>
        <td>${x.threshold != null ? `±${fmt(x.threshold, 2)}` : "—"}</td><td>${posText(x.position)} → ${posText(x.target)}</td>
        <td>${fmt(x.bid, 2)} / ${fmt(x.ask, 2)}</td><td>${esc(x.action)}${x.note ? " ⚠" : ""}</td></tr>`).join("")
      : `<tr><td class="bento-empty">No decisions yet</td></tr>`;
  }

  async function loadTradingBacktests() {
    const runs = await api("/api/trading/backtests");
    const sel = $("#tr-bt-run"), keep = sel.value;
    sel.innerHTML = runs.map((r) => `<option value="${esc(r.run_id)}">${localTime(r.created)} · ${r.config.model} · ${r.config.backtest_days}d</option>`).join("");
    if (keep && runs.some((r) => r.run_id === keep)) sel.value = keep;
    if (!runs.length) {
      $("#tr-bt-summary").innerHTML = `<div class="muted" style="grid-column: 1 / -1">No strategy backtest yet. Press BACKTEST + TRAIN once IG price history has downloaded.</div>`;
      lineChart("tr-bt-chart", [], []);
      return;
    }
    const b = await api(`/api/trading/backtest/${encodeURIComponent(sel.value)}`);
    const sm = b.summary, cls = (v) => (v > 0 ? "up" : v < 0 ? "down" : "");
    const pct = (v) => v == null ? "—" : `${fmt(100 * v, 0)}%`;
    $("#tr-bt-summary").innerHTML = [
      ["P&L", `<span class="${cls(sm.pnl)}">${money(sm.pnl)}</span>`], ["RETURN", `${signed(sm.return_pct)}%`],
      ["BUY & HOLD", `<span class="${cls(sm.buy_hold_pnl)}">${money(sm.buy_hold_pnl)}</span>`], ["SHARPE", fmt(sm.sharpe, 2)],
      ["MAX DRAWDOWN", money(sm.max_drawdown)], ["TRADES", `${sm.trades} (${sm.round_trips} closed)`],
      ["WIN RATE", pct(sm.win_rate)], ["DIRECTION HIT", pct(sm.hit_rate)],
      ["SPREAD PAID", money(sm.spread_cost)], ["FUNDING", money(sm.funding)], ["IN MARKET", pct(sm.exposure)],
      ["MAE vs NO-CHANGE", `${fmt(sm.mae, 2)} / ${fmt(sm.mae_no_change, 2)}`],
    ].map(([l, v]) => `<div>${l}<b>${v}</b></div>`).join("");
    const ser = b.series;
    lineChart("tr-bt-chart", ser.map((r) => localTime(r.ts)), [
      { label: "strategy equity", data: ser.map((r) => r.equity), color: PALETTE[2], width: 1.6 },
    ]);
  }

  function onStrategyDone(job) {
    if (job.status !== "ok") { setStatus("#tr-status", `Strategy backtest failed: ${(job.error || "").split("\n")[0]}`, "error"); return; }
    const r = job.result || {};
    if (r.status === "skipped") { setStatus("#tr-status", `Skipped: ${r.reason}.`, "error"); return; }
    const sm = r.backtest?.summary || {};
    setStatus("#tr-status", `Backtest: ${money(sm.pnl)} over ${sm.trades} trades (buy & hold ${money(sm.buy_hold_pnl)}). Live model retrained.`, "ok");
    $("#tr-bt-run").value = "";
    if (activeTab() === "trading") loaders.trading({ auto: true });
  }

  async function saveTrading(extra) {
    const body = { mirror_ig: $("#tr-mirror").checked, interval_minutes: Number($("#tr-interval").value) || 60,
      size: $("#tr-size").value === "" ? null : Number($("#tr-size").value),
      threshold_k: $("#tr-k").value === "" ? null : Number($("#tr-k").value), ...(extra || {}) };
    try {
      await post("/api/trading/settings", body);
      trDirty = false;
      setStatus("#tr-status", extra?.enabled === true ? "Auto-trading on." : extra?.enabled === false ? "Auto-trading off." : "Saved.", "ok");
      await loadTrading();
    } catch (e) { setStatus("#tr-status", e.message, "error"); }
  }
  ["#tr-interval", "#tr-size", "#tr-k"].forEach((sel) => $(sel).addEventListener("input", () => { trDirty = true; }));
  $("#tr-enabled").addEventListener("change", () => saveTrading({ enabled: $("#tr-enabled").checked }));
  $("#tr-mirror").addEventListener("change", () => saveTrading());
  $("#tr-save").addEventListener("click", () => saveTrading());
  $("#tr-cycle").addEventListener("click", async () => {
    try { await post("/api/trading/cycle", {}); setStatus("#tr-status", "Trading cycle running…", "running"); kick(); }
    catch (e) { setStatus("#tr-status", e.message, "error"); }
  });
  $("#tr-close").addEventListener("click", async () => {
    try { await post("/api/trading/cycle", { close: true }); setStatus("#tr-status", "Closing position…", "running"); kick(); }
    catch (e) { setStatus("#tr-status", e.message, "error"); }
  });
  $("#tr-reset").addEventListener("click", async () => {
    if (!window.confirm("Reset the paper account? This deletes all paper trades, decisions and equity history (IG demo positions are not touched).")) return;
    try { await post("/api/trading/reset", {}); setStatus("#tr-status", "Paper account reset.", "ok"); loaders.trading(); }
    catch (e) { setStatus("#tr-status", e.message, "error"); }
  });
  $("#tr-bt-go").addEventListener("click", async () => {
    try {
      await post("/api/jobs/strategy", { backtest_days: Number($("#tr-bt-days").value) || null });
      setStatus("#tr-status", "Strategy backtest running…", "running"); kick();
    } catch (e) { setStatus("#tr-status", e.message, "error"); }
  });
  $("#tr-days").addEventListener("change", () => loadTradingChart().catch((e) => toast(e.message)));
  $("#tr-bt-run").addEventListener("change", () => loadTradingBacktests().catch((e) => toast(e.message)));
  loaders.trading = (o) => loadTrading().catch((e) => { if (!o?.auto) toast(e.message); });

  // ── data ─────────────────────────────────────────────────────────
  async function loadSources() {
    const src = await api("/api/sources");
    $("#sources-table").innerHTML = `
      <tr><th>SOURCE</th><th>CATEGORY</th><th>MODE</th><th>ENABLED</th><th>FIRST</th><th>LAST</th><th>FRESHNESS</th><th>SIZE</th><th>DESCRIPTION</th></tr>
      ${src.map((s) => `<tr>
        <td>${esc(s.name)}</td><td>${esc(s.category)}</td><td>${esc(s.mode)}</td>
        <td><span class="tag ${s.enabled ? "on" : "off"}">${s.enabled ? "on" : "off"}</span></td>
        <td>${s.first ? esc(s.first.slice(0, 10)) : "—"}</td><td>${s.last ? localTime(s.last) : "—"}</td>
        <td>${agoSpan(s.last)}</td><td>${s.bytes ? fmt(s.bytes / 1e6, 1) + " MB" : "—"}</td>
        <td class="muted">${esc(s.description)}</td></tr>`).join("")}`;
    const withData = src.filter((s) => s.last).map((s) => s.name);
    const cur = $("#ex-dataset").value;
    $("#ex-dataset").innerHTML = withData.map((n) => `<option ${n === cur ? "selected" : ""}>${esc(n)}</option>`).join("");
    if (withData.length) loadExplorer(cur === $("#ex-dataset").value);
  }
  async function loadExplorer(keepCol) {
    const ds = $("#ex-dataset").value, days = $("#ex-days").value;
    const col = keepCol ? $("#ex-column").value : "";
    const s = await api(`/api/series?dataset=${encodeURIComponent(ds)}&days=${days}${col ? `&column=${encodeURIComponent(col)}` : ""}`);
    if (!keepCol) $("#ex-column").innerHTML = `<option value="">(first 6)</option>` + (s.columns || []).map((c) => `<option>${esc(c)}</option>`).join("");
    const pts = s.points || [];
    lineChart("ex-chart", pts.map((p) => localTime(p.timestamp)), (s.shown || []).map((c) => ({ label: c, data: pts.map((p) => p[c]) })));
  }
  loaders.data = (o) => loadSources().catch((e) => { if (!o?.auto) toast(e.message); });
  $("#ex-dataset").addEventListener("change", () => loadExplorer());
  $("#ex-column").addEventListener("change", () => loadExplorer(true));
  $("#ex-days").addEventListener("change", () => loadExplorer(true));

  // ── runs ─────────────────────────────────────────────────────────
  loaders.runs = async () => {
    const runs = await api("/api/runs");
    $("#runs-table").innerHTML = runs.length ? `
      <tr><th>RUN</th><th>STATUS</th><th>STARTED</th><th>STEPS</th><th>CHAMPIONS</th></tr>
      ${runs.map((r) => `<tr>
        <td>${esc(r.run_id)}</td><td class="${r.status === "ok" ? "up" : r.status === "running" ? "" : "down"}">${esc(r.status)}</td>
        <td>${r.started ? localTime(r.started) : "—"}</td>
        <td>${(r.steps || []).map((s) => `<span class="tag ${s.status === "ok" ? "on" : s.status === "running" ? "" : "sell"}" title="${esc(s.error || "")}">${esc(s.step)} ${s.seconds ?? ""}s</span>`).join(" ")}</td>
        <td>${Object.entries(r.champions || {}).map(([t, m]) => `${esc(t)}: ${esc(m)}`).join(" · ")}</td></tr>`).join("")}`
      : `<tr><td class="bento-empty">No pipeline runs yet</td></tr>`;
  };

  // ── config ───────────────────────────────────────────────────────
  let settingsCache = null;
  loaders.config = async () => {
    settingsCache = await api("/api/settings");
    const v = settingsCache.values;
    const help = settingsCache.help || {}, mhelp = settingsCache.model_help || {}, shelp = settingsCache.source_help || {};
    const tip = (t) => (t ? ` title="${esc(t)}"` : "");
    $("#settings-form").innerHTML = Object.entries(v).map(([k, val]) => {
      const lab = `<label class="${help[k] ? "has-help" : ""}"${tip(help[k])}>${esc(k)}</label>`;
      if (k === "models.candidates") {
        return `${lab}<div class="model-checks">${settingsCache.available_models.map((m) =>
          `<label${tip(mhelp[m])}><input type="checkbox" data-model="${esc(m)}" ${val?.includes(m) ? "checked" : ""}> ${esc(m)}</label>`).join("")}</div>`;
      }
      if (typeof val === "boolean") return `${lab}<input type="checkbox" data-key="${esc(k)}" ${val ? "checked" : ""}${tip(help[k])}>`;
      return `${lab}<input type="number" step="any" data-key="${esc(k)}" value="${esc(val)}"${tip(help[k])}>`;
    }).join("");
    $("#sources-form").innerHTML = Object.entries(settingsCache.sources).map(([n, on]) =>
      `<label class="${shelp[n] ? "has-help" : ""}"${tip(shelp[n])}>${esc(n)}</label><input type="checkbox" data-source="${esc(n)}" ${on ? "checked" : ""}${tip(shelp[n])}>`).join("");
  };
  $("#save-settings").addEventListener("click", async () => {
    const values = {};
    $$("#settings-form [data-key]").forEach((el) => { values[el.dataset.key] = el.type === "checkbox" ? el.checked : Number(el.value); });
    values["models.candidates"] = $$("#settings-form [data-model]").filter((el) => el.checked).map((el) => el.dataset.model);
    const sources = {};
    $$("#sources-form [data-source]").forEach((el) => { sources[el.dataset.source] = el.checked; });
    try { const r = await post("/api/settings", { values, sources }); toast(`saved ${r.saved}`); }
    catch (e) { toast(e.message); }
  });

  $("#job-cancel").addEventListener("click", async () => {
    try { await post("/api/jobs/cancel"); toast("cancelling…"); } catch (e) { toast(e.message); }
  });

  // ── theme + clock ────────────────────────────────────────────────
  const savedTheme = (() => { try { return localStorage.getItem("energy-theme"); } catch { return null; } })();
  if (savedTheme) document.documentElement.dataset.theme = savedTheme;
  $("#theme-toggle").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("energy-theme", next); } catch { /* ignore */ }
    loaders[$(".sidebar-nav-item.active").dataset.tab]?.();
  });
  setInterval(() => { $("#current-time").textContent = new Date().toLocaleTimeString("en-GB", { timeZone: "Europe/London" }); }, 1000);

  loaders.overview();
  heartbeat();
})();
