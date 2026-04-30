const $ = (id) => document.getElementById(id);

function setTab(name){
  document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
  document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
  document.querySelector(`.tab[data-tab="${name}"]`).classList.add("active");
  $(`tab-${name}`).classList.add("active");

  if(name === "about"){
    setTimeout(() => {
      try { refreshHelpObserver(); } catch(_) {}
    }, 50);
  }
}

function bindTabs(){
  document.querySelectorAll(".tab").forEach(btn => {
    btn.addEventListener("click", () => setTab(btn.dataset.tab));
  });
}

function toInt(v){
  if (v === "" || v === null || v === undefined) return null;
  const n = parseInt(v, 10);
  return Number.isNaN(n) ? null : n;
}
function toFloat(v){
  if (v === "" || v === null || v === undefined) return null;
  const n = parseFloat(v);
  return Number.isNaN(n) ? null : n;
}

function setVisible(el, visible){
  if(!el) return;
  el.style.display = visible ? "" : "none";
}

function updateDropoutVisibility(){
  const enabled = $("train-dropout")?.checked === true;
  setVisible($("train-dropout-settings"), enabled);
}
function updateEarlyStoppingVisibility(){
  const enabled = $("train-early")?.checked === true;
  setVisible($("train-early-settings"), enabled);
}
function updateReplayVisibility(){
  const enabled = $("train-replay")?.checked === true;
  setVisible($("train-replay-settings"), enabled);
}

/* ---------- helpers ---------- */
function escapeHtml(s){
  if (s === null || s === undefined) return "";
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
function fmtNum(v, digits=6){
  if(v === null || v === undefined) return "—";
  const n = Number(v);
  if(Number.isNaN(n)) return escapeHtml(v);
  const out = (Math.abs(n) < 1 && digits > 0) ? n.toFixed(digits) : String(n);
  return out;
}
function badge(text, cls=""){
  return `<span class="badge ${cls}">${escapeHtml(text)}</span>`;
}
function onOffBadge(flag){
  return flag ? badge("on", "on") : badge("off", "off");
}
function kvRow(key, valHtml){
  return `<div class="kv-row">
    <div class="kv-key">${escapeHtml(key)}</div>
    <div class="kv-val">${valHtml}</div>
  </div>`;
}
function section(title, innerHtml){
  return `<div class="resume-section">
    <div class="resume-section-title">${escapeHtml(title)}</div>
    ${innerHtml}
  </div>`;
}
