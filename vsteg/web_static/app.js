const modes = document.querySelectorAll(".mode");
const panels = {
  hide: document.getElementById("panel-hide"),
  reveal: document.getElementById("panel-reveal"),
  check: document.getElementById("panel-check"),
  compare: document.getElementById("panel-compare"),
};
const statusEl = document.getElementById("status");
const busyEl = document.getElementById("busy");
const busyLabelEl = document.getElementById("busy-label");
const detectResult = document.getElementById("detect-result");
const compareResult = document.getElementById("compare-result");
let compareCharts = [];

function setStatus(message, kind = "") {
  statusEl.textContent = message || "";
  statusEl.className = "status" + (kind ? ` is-${kind}` : "");
}

function setBusy(on, label = "Working…") {
  if (!busyEl) return;
  if (on) {
    if (busyLabelEl) busyLabelEl.textContent = label;
    busyEl.hidden = false;
    busyEl.setAttribute("aria-hidden", "false");
    document.body.style.overflow = "hidden";
  } else {
    busyEl.hidden = true;
    busyEl.setAttribute("aria-hidden", "true");
    document.body.style.overflow = "";
  }
}

function formatBytes(n) {
  if (!n && n !== 0) return "";
  const units = ["B", "KB", "MB", "GB"];
  let size = n;
  let i = 0;
  while (size >= 1024 && i < units.length - 1) {
    size /= 1024;
    i += 1;
  }
  return `${size.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function updateDropzone(zone, file) {
  const title = zone.querySelector(".dropzone-title");
  const hint = zone.querySelector(".dropzone-hint");
  const fileEl = zone.querySelector(".dropzone-file");
  if (!file) {
    zone.classList.remove("has-file");
    if (fileEl) {
      fileEl.hidden = true;
      fileEl.textContent = "";
    }
    return;
  }
  zone.classList.add("has-file");
  if (title) title.textContent = "File selected";
  if (hint) hint.textContent = "Click or drop to replace";
  if (fileEl) {
    fileEl.hidden = false;
    fileEl.textContent = `${file.name} · ${formatBytes(file.size)}`;
  }
}

function bindDropzones() {
  document.querySelectorAll("[data-dropzone]").forEach((zone) => {
    const input = zone.querySelector('input[type="file"]');
    if (!input) return;

    const setFiles = (files) => {
      if (!files || !files.length) return;
      const dt = new DataTransfer();
      dt.items.add(files[0]);
      input.files = dt.files;
      updateDropzone(zone, files[0]);
      input.dispatchEvent(new Event("change", { bubbles: true }));
    };

    input.addEventListener("change", () => {
      updateDropzone(zone, input.files && input.files[0] ? input.files[0] : null);
    });

    ["dragenter", "dragover"].forEach((evt) => {
      zone.addEventListener(evt, (e) => {
        e.preventDefault();
        e.stopPropagation();
        zone.classList.add("is-dragover");
      });
    });

    ["dragleave", "drop"].forEach((evt) => {
      zone.addEventListener(evt, (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (evt === "dragleave" && zone.contains(e.relatedTarget)) return;
        zone.classList.remove("is-dragover");
      });
    });

    zone.addEventListener("drop", (e) => {
      const files = e.dataTransfer && e.dataTransfer.files;
      if (!files || !files.length) return;
      setFiles(files);
    });
  });
}

bindDropzones();

function destroyCompareCharts() {
  compareCharts.forEach((c) => {
    try { c.destroy(); } catch (_) {}
  });
  compareCharts = [];
}

function switchMode(mode) {
  modes.forEach((btn) => {
    const active = btn.dataset.mode === mode;
    btn.classList.toggle("is-active", active);
    btn.setAttribute("aria-selected", active ? "true" : "false");
  });
  Object.entries(panels).forEach(([key, panel]) => {
    if (!panel) return;
    const active = key === mode;
    panel.classList.toggle("is-active", active);
    panel.hidden = !active;
  });
  setStatus("");
  detectResult.hidden = true;
  if (compareResult) compareResult.hidden = true;
  destroyCompareCharts();
}

modes.forEach((btn) => {
  btn.addEventListener("click", () => switchMode(btn.dataset.mode));
});

async function downloadFromResponse(res, fallbackName) {
  const blob = await res.blob();
  const cd = res.headers.get("content-disposition") || "";
  const match = /filename="?([^"]+)"?/i.exec(cd);
  const name = match ? match[1] : fallbackName;
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

async function postForm(url, form, { downloadAs, busyLabel } = {}) {
  const btn = form.querySelector('button[type="submit"]');
  btn.disabled = true;
  setBusy(true, busyLabel || "Working…");
  setStatus(busyLabel || "Working…");
  try {
    const body = new FormData(form);
    const res = await fetch(url, { method: "POST", body });
    if (!res.ok) {
      let detail = "Something went wrong.";
      try {
        const data = await res.json();
        detail = data.detail || detail;
      } catch (_) {
        detail = await res.text();
      }
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    if (downloadAs) {
      await downloadFromResponse(res, downloadAs);
      setStatus("Done — your file should be downloading.", "ok");
      return null;
    }
    return await res.json();
  } catch (err) {
    setStatus(err.message || String(err), "error");
    return null;
  } finally {
    btn.disabled = false;
    setBusy(false);
  }
}

document.getElementById("form-hide").addEventListener("submit", async (e) => {
  e.preventDefault();
  await postForm("/api/encode", e.target, {
    downloadAs: "stego.mp4",
    busyLabel: "Hiding secret in video…",
  });
});

document.getElementById("form-reveal").addEventListener("submit", async (e) => {
  e.preventDefault();
  await postForm("/api/decode", e.target, {
    downloadAs: "recovered.txt",
    busyLabel: "Checking for a vsteg payload…",
  });
});

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderDetectReport(data) {
  const verdict = data.verdict || "clean";
  const media = data.media || {};
  const categories = data.categories || [];
  const signals = data.signals || [];
  const thresholds = data.thresholds || {};

  const mediaRows = [
    ["Probed with", media.probe_source],
    ["File", media.filename],
    ["Size", media.size_human],
    ["Resolution", media.width && media.height ? `${media.width}×${media.height}` : null],
    ["Duration", media.duration_sec ? `${media.duration_sec}s` : null],
    ["Frames", media.frames || null],
    ["FPS", media.fps || null],
    ["Video codec", media.codec],
    ["Profile", media.profile],
    ["Pixel format", media.pix_fmt],
    ["Bitrate", media.bitrate_human || media.bitrate],
    ["Container", media.container || media.format_long_name],
    ["Encoder tag", media.encoder],
    ["Audio streams", media.audio_count != null ? String(media.audio_count) : null],
    ["Subtitle streams", media.subtitle_count != null ? String(media.subtitle_count) : null],
    ["Data streams", media.data_count != null ? String(media.data_count) : null],
  ]
    .filter(([, v]) => v !== null && v !== undefined && v !== "")
    .map(
      ([k, v]) =>
        `<div><dt>${escapeHtml(k)}</dt><dd>${escapeHtml(v)}</dd></div>`
    )
    .join("");

  const audioHtml = (media.audio_streams || [])
    .map(
      (a) =>
        `<li class="finding-meta">#${escapeHtml(a.index)} ${escapeHtml(a.codec)} · ${escapeHtml(a.sample_rate)} Hz · ${escapeHtml(a.channels)} ch${a.language ? ` · ${escapeHtml(a.language)}` : ""}</li>`
    )
    .join("");

  const subHtml = (media.subtitle_streams || [])
    .map(
      (s) =>
        `<li class="finding-meta">#${escapeHtml(s.index)} ${escapeHtml(s.codec)}${s.language ? ` · ${escapeHtml(s.language)}` : ""}${s.title ? ` · ${escapeHtml(s.title)}` : ""}</li>`
    )
    .join("");

  const tagEntries = Object.entries(media.tags || {});
  const tagsHtml = tagEntries.length
    ? tagEntries
        .slice(0, 12)
        .map(([k, v]) => `<li class="finding-meta"><strong>${escapeHtml(k)}</strong>: ${escapeHtml(v)}</li>`)
        .join("")
    : `<li class="finding-meta">No format tags.</li>`;

  const categoryHtml = categories
    .map((cat) => {
      const status = cat.status || "clear";
      return `
        <li>
          <div class="cat-head">
            <strong>${escapeHtml(cat.title)}</strong>
            <span class="badge ${escapeHtml(status)}">${escapeHtml(status)}</span>
          </div>
          <p class="cat-detail">${escapeHtml(cat.summary || "")}</p>
          <p class="cat-detail">${escapeHtml(cat.detail || "")}</p>
        </li>
      `;
    })
    .join("");

  const findingsHtml = signals.length
    ? signals
        .map((s) => {
          const sev = s.severity || "info";
          return `
            <li>
              <p class="finding-title">
                <span class="sev-${escapeHtml(sev)}">+${escapeHtml(s.weight ?? 0)}</span>
                ${escapeHtml(s.label || s.signal)}
              </p>
              <p class="finding-meta">
                ${escapeHtml(s.title || s.signal)} · severity ${escapeHtml(sev)}
                ${s.explanation ? ` — ${escapeHtml(s.explanation)}` : ""}
              </p>
            </li>
          `;
        })
        .join("")
    : `<li><p class="cat-detail">No individual signals fired.</p></li>`;

  const statFeatures = data.stat_features || [];
  const featuresHtml = statFeatures.length
    ? `
      <div class="report-section">
        <h3>Statistical / ML feature checks</h3>
        <p class="finding-meta">
          Handcrafted features used by deep Check (chi · SPA · RS · DCT · keyframe · mdat slack)
          and the optional ML ensemble.
        </p>
        <div class="feature-table" role="table" aria-label="Statistical feature results">
          <div class="feature-row feature-head" role="row">
            <span>Check</span><span>Result</span><span>Status</span><span>Notes</span>
          </div>
          ${statFeatures
            .map(
              (f) => `
            <div class="feature-row status-${escapeHtml(f.status || "info")}" role="row">
              <span class="feature-name">${escapeHtml(f.name || f.key)}</span>
              <span class="feature-value">${escapeHtml(f.display)}</span>
              <span class="feature-status">${escapeHtml(f.status || "info")}</span>
              <span class="feature-note">${escapeHtml(f.note || "")}</span>
            </div>`
            )
            .join("")}
        </div>
      </div>`
    : "";

  detectResult.innerHTML = `
    <div class="report-head">
      <div>
        <p class="verdict ${escapeHtml(verdict)}">${escapeHtml(verdict)}</p>
        <p class="score">
          Score <strong>${escapeHtml(data.score)}/100</strong>
          · ${escapeHtml(data.confidence_label || "")}
        </p>
        <div class="score-bar is-${escapeHtml(verdict)}"><span style="width:${Math.min(100, data.score || 0)}%"></span></div>
      </div>
      <div>
        <p class="summary">${escapeHtml(data.summary || "")}</p>
        <p class="finding-meta">
          Thresholds: clean ≤ ${escapeHtml(thresholds.clean_max ?? 24)},
          suspicious ≥ ${escapeHtml(thresholds.suspicious_min ?? 25)},
          likely-stego ≥ ${escapeHtml(thresholds.likely_stego_min ?? 60)}.
          Informational ffprobe notes (+0) do not raise the score.
        </p>
      </div>
    </div>

    <div class="report-section">
      <h3>Media probed (ffprobe)</h3>
      <dl class="media-grid">${mediaRows || "<div><dd>Unavailable</dd></div>"}</dl>
    </div>

    ${featuresHtml}

    <div class="report-columns">
      <div class="report-section">
        <h3>Audio / subtitles</h3>
        <ul class="findings">
          ${audioHtml || '<li class="finding-meta">No audio streams.</li>'}
          ${subHtml || '<li class="finding-meta">No subtitle streams.</li>'}
        </ul>
      </div>
      <div class="report-section">
        <h3>Metadata tags</h3>
        <ul class="findings">${tagsHtml}</ul>
      </div>
    </div>

    <div class="report-columns">
      <div class="report-section">
        <h3>Checks run</h3>
        <ul class="cat-list">${categoryHtml}</ul>
      </div>
      <div class="report-section">
        <h3>Findings</h3>
        <ul class="findings">${findingsHtml}</ul>
      </div>
    </div>
  `;
}

document.getElementById("form-check").addEventListener("submit", async (e) => {
  e.preventDefault();
  detectResult.hidden = true;
  const data = await postForm("/api/detect", e.target, {
    busyLabel: "Analyzing video…",
  });
  if (!data) return;
  renderDetectReport(data);
  detectResult.hidden = false;
  setStatus("Analysis complete.", "ok");
});

function chartDefaults() {
  return {
    responsive: true,
    plugins: { legend: { labels: { color: "#e8f2ee" } } },
    scales: {
      x: { ticks: { color: "#e8f2ee" }, grid: { color: "rgba(232,242,238,0.08)" } },
      y: { ticks: { color: "#e8f2ee" }, grid: { color: "rgba(232,242,238,0.08)" } },
    },
  };
}

function renderCompareReport(data) {
  destroyCompareCharts();
  const nameA = (data.media_a && data.media_a.filename) || "A";
  const nameB = (data.media_b && data.media_b.filename) || "B";
  const similarity = data.similarity || "different";
  const rows = (data.attributes || [])
    .map((r) => `
      <tr class="${r.changed ? "changed" : ""}">
        <td>${escapeHtml(r.label)}</td>
        <td>${escapeHtml(r.a)}</td>
        <td>${escapeHtml(r.b)}</td>
        <td>${escapeHtml(r.delta ?? "")}</td>
      </tr>`)
    .join("");
  const notes = (data.notes || [])
    .map((n) => `<li class="finding-meta">${escapeHtml(n)}</li>`)
    .join("") || `<li class="finding-meta">No extra notes.</li>`;
  const fa = data.frame_analysis || {};
  const ca = data.container_analysis || {};
  const containerRows = [
    ["Size Δ", ca.size_delta],
    ["mdat payload A", ca.mdat_payload_a],
    ["mdat payload B", ca.mdat_payload_b],
    ["mdat slack A", ca.mdat_slack_a],
    ["mdat slack B", ca.mdat_slack_b],
    ["mdat slack Δ", ca.mdat_slack_delta],
    ["Identical prefix (LCP)", ca.lcp_bytes],
    ["OpenPuff-like", ca.openpuff_like],
  ]
    .filter(([, v]) => v !== null && v !== undefined && v !== "")
    .map(([k, v]) => `<div><dt>${escapeHtml(k)}</dt><dd>${escapeHtml(v)}</dd></div>`)
    .join("");

  compareResult.innerHTML = `
    <div class="report-head">
      <div>
        <span class="similarity ${escapeHtml(similarity)}">${escapeHtml(similarity)}</span>
        <p class="finding-meta">${escapeHtml(data.changed_count || 0)} attribute(s) changed.</p>
        ${fa.mean_abs_diff != null ? `<p class="finding-meta">Mean frame abs diff: <strong>${escapeHtml(fa.mean_abs_diff)}</strong> (max ${escapeHtml(fa.max_abs_diff)})</p>` : ""}
        ${ca.openpuff_like ? `<p class="finding-meta sev-high">OpenPuff-like container pattern detected.</p>` : ""}
      </div>
      <div>
        <p class="summary">${escapeHtml(data.summary || "")}</p>
        ${fa.message ? `<p class="finding-meta">${escapeHtml(fa.message)}</p>` : ""}
        ${ca.message ? `<p class="finding-meta">${escapeHtml(ca.message)}</p>` : ""}
      </div>
    </div>

    <div class="report-section">
      <h3>Attribute comparison</h3>
      <div class="compare-table-wrap">
        <table class="compare-table">
          <thead><tr><th>Attribute</th><th>${escapeHtml(nameA)}</th><th>${escapeHtml(nameB)}</th><th>Δ</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </div>

    <div class="report-section">
      <h3>Container forensics</h3>
      <dl class="media-grid">${containerRows || "<div><dd>Unavailable</dd></div>"}</dl>
    </div>

    <div class="report-section">
      <h3>Graphs</h3>
      <div class="chart-grid">
        <div class="chart-box"><h4>Scalars</h4><canvas id="chart-scalar"></canvas></div>
        <div class="chart-box"><h4>Streams</h4><canvas id="chart-streams"></canvas></div>
        <div class="chart-box"><h4>Container</h4><canvas id="chart-container"></canvas></div>
        <div class="chart-box"><h4>Frame MAD</h4><canvas id="chart-mad"></canvas></div>
        <div class="chart-box"><h4>Luma histogram</h4><canvas id="chart-hist"></canvas></div>
      </div>
    </div>

    <div class="report-section">
      <h3>Notes</h3>
      <ul class="findings">${notes}</ul>
    </div>
  `;

  const charts = data.charts || {};
  if (typeof Chart === "undefined") {
    setStatus("Charts library failed to load; table report is still available.", "error");
    return;
  }
  const opts = chartDefaults();
  compareCharts.push(new Chart(document.getElementById("chart-scalar"), {
    type: "bar",
    data: {
      labels: (charts.scalar && charts.scalar.labels) || [],
      datasets: [
        { label: nameA, data: (charts.scalar && charts.scalar.a) || [], backgroundColor: "#c8f560" },
        { label: nameB, data: (charts.scalar && charts.scalar.b) || [], backgroundColor: "#ff6b4a" },
      ],
    },
    options: opts,
  }));
  compareCharts.push(new Chart(document.getElementById("chart-streams"), {
    type: "bar",
    data: {
      labels: (charts.streams && charts.streams.labels) || [],
      datasets: [
        { label: nameA, data: (charts.streams && charts.streams.a) || [], backgroundColor: "#c8f560" },
        { label: nameB, data: (charts.streams && charts.streams.b) || [], backgroundColor: "#ff6b4a" },
      ],
    },
    options: opts,
  }));
  if (document.getElementById("chart-container") && charts.container) {
    compareCharts.push(new Chart(document.getElementById("chart-container"), {
      type: "bar",
      data: {
        labels: charts.container.labels || [],
        datasets: [
          { label: nameA, data: charts.container.a || [], backgroundColor: "#c8f560" },
          { label: nameB, data: charts.container.b || [], backgroundColor: "#ff6b4a" },
        ],
      },
      options: opts,
    }));
  }
  const madVals = (charts.frame_mad && charts.frame_mad.values) || [];
  compareCharts.push(new Chart(document.getElementById("chart-mad"), {
    type: "line",
    data: {
      labels: (charts.frame_mad && charts.frame_mad.labels) || [],
      datasets: [{
        label: "Mean abs diff",
        data: madVals,
        borderColor: "#c8f560",
        tension: 0.25,
      }],
    },
    options: opts,
  }));
  const histA = (charts.histogram && charts.histogram.a) || [];
  compareCharts.push(new Chart(document.getElementById("chart-hist"), {
    type: "line",
    data: {
      labels: (charts.histogram && charts.histogram.labels) || [],
      datasets: [
        { label: `${nameA} luma`, data: histA, borderColor: "#c8f560", tension: 0.2 },
        { label: `${nameB} luma`, data: (charts.histogram && charts.histogram.b) || [], borderColor: "#ff6b4a", tension: 0.2 },
      ],
    },
    options: opts,
  }));
}

document.getElementById("form-compare").addEventListener("submit", async (e) => {
  e.preventDefault();
  compareResult.hidden = true;
  destroyCompareCharts();
  const data = await postForm("/api/compare", e.target, {
    busyLabel: "Comparing videos…",
  });
  if (!data) return;
  renderCompareReport(data);
  compareResult.hidden = false;
  setStatus("Comparison complete.", "ok");
});
