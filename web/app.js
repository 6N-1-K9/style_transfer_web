const $ = (id) => document.getElementById(id);

function setTab(name){
  document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
  document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
  document.querySelector(`.tab[data-tab="${name}"]`).classList.add("active");
  $(`tab-${name}`).classList.add("active");
}

document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => setTab(btn.dataset.tab));
});

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

/* ---------- SVG placeholders ---------- */
function svgPlaceholderSimple(){
  const svg =
`<svg xmlns="http://www.w3.org/2000/svg" width="600" height="600">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#f1f5f9"/>
      <stop offset="1" stop-color="#e2e8f0"/>
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="600" height="600" fill="url(#g)"/>
  <rect x="18" y="18" width="564" height="564" rx="26" ry="26" fill="rgba(0,0,0,0.03)" stroke="rgba(0,0,0,0.10)"/>
  <g transform="translate(170,190)" fill="rgba(0,0,0,0.18)">
    <rect x="0" y="0" width="260" height="210" rx="18" ry="18" fill="rgba(0,0,0,0.06)" stroke="rgba(0,0,0,0.16)" />
    <circle cx="72" cy="70" r="18" />
    <path d="M30 180 L110 115 L160 155 L205 125 L235 180 Z" />
  </g>
</svg>`;
  return "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(svg)));
}

function svgPlaceholderNotFound(text){
  const t = escapeHtml(text || "Датасет не найден");
  const svg =
`<svg xmlns="http://www.w3.org/2000/svg" width="600" height="600">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#f8fafc"/>
      <stop offset="1" stop-color="#eef2ff"/>
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="600" height="600" fill="url(#g)"/>
  <rect x="18" y="18" width="564" height="564" rx="26" ry="26" fill="rgba(0,0,0,0.02)" stroke="rgba(0,0,0,0.10)"/>
  <g transform="translate(0,0)">
    <text x="300" y="290" text-anchor="middle" font-size="30" font-family="ui-sans-serif,system-ui" fill="rgba(0,0,0,0.62)">${t}</text>
    <text x="300" y="335" text-anchor="middle" font-size="18" font-family="ui-sans-serif,system-ui" fill="rgba(0,0,0,0.45)">папка не существует или перемещена</text>
  </g>
</svg>`;
  return "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(svg)));
}

/* ---------- TRAIN: dataset preview ---------- */
function setDatasetPlaceholders(side){
  const ph = svgPlaceholderSimple();
  const setSide = (s) => {
    for(let i=1;i<=3;i++){
      const id = (s === "A" ? `prev-a-${i}` : `prev-b-${i}`);
      const el = $(id);
      if(el) el.src = ph;
    }
  };
  if(!side){
    setSide("A");
    setSide("B");
  }else{
    setSide(side);
  }
}

async function refreshDatasetPreview(side){
  const folder = (side === "A" ? $("train-a").value : $("train-b").value).trim();
  if(!folder){
    setDatasetPlaceholders(side);
    return;
  }

  try{
    const res = await window.pywebview.api.get_dataset_preview(folder, 3);
    if(!res || !res.ok){
      setDatasetPlaceholders(side);
      return;
    }

    const imgs = res.images || [];
    const ph = svgPlaceholderSimple();
    for(let i=1;i<=3;i++){
      const el = $(side === "A" ? `prev-a-${i}` : `prev-b-${i}`);
      const item = imgs[i-1];
      el.src = (item && item.data_url) ? item.data_url : ph;
    }
  }catch(e){
    setDatasetPlaceholders(side);
  }
}

let debounceA = null;
let debounceB = null;
function schedulePreviewRefreshSide(side){
  if(side === "A"){
    if(debounceA) clearTimeout(debounceA);
    debounceA = setTimeout(() => refreshDatasetPreview("A"), 200);
  }else{
    if(debounceB) clearTimeout(debounceB);
    debounceB = setTimeout(() => refreshDatasetPreview("B"), 200);
  }
}

/* ---------- INFER: training dataset preview ---------- */
function setInferTrainPlaceholders(which, kind){
  // kind: "simple" | "not_found"
  const ph = (kind === "not_found") ? svgPlaceholderNotFound("Датасет не найден") : svgPlaceholderSimple();

  const setA = () => {
    for(let i=1;i<=4;i++){
      const el = $(`infer-a-${i}`);
      if(el) el.src = ph;
    }
  };
  const setB = () => {
    for(let i=1;i<=4;i++){
      const el = $(`infer-b-${i}`);
      if(el) el.src = ph;
    }
  };

  if(which === "A") setA();
  else if(which === "B") setB();
  else { setA(); setB(); }
}

function setInferTrainNote(text){
  const el = $("infer-train-note");
  if(el) el.textContent = text || "";
}

async function refreshInferTrainingPreview(){
  const modelPath = ($("infer-model").value || "").trim();
  if(!modelPath){
    setInferTrainPlaceholders("ALL", "simple");
    setInferTrainNote("Выбери модель — появятся 4 случайные картинки из датасетов, на которых она обучалась.");
    return;
  }

  try{
    const res = await window.pywebview.api.get_infer_training_datasets_preview(modelPath, 4);
    if(!res || !res.ok){
      setInferTrainPlaceholders("ALL", "simple");
      setInferTrainNote("Не удалось получить датасеты из модели.");
      return;
    }

    // A
    if(res.a_status === "ok"){
      const ph = svgPlaceholderSimple();
      const arr = res.a_images || [];
      for(let i=1;i<=4;i++){
        const el = $(`infer-a-${i}`);
        const item = arr[i-1];
        el.src = (item && item.data_url) ? item.data_url : ph;
      }
    }else if(res.a_status === "not_found"){
      setInferTrainPlaceholders("A", "not_found");
    }else{
      setInferTrainPlaceholders("A", "simple");
    }

    // B
    if(res.b_status === "ok"){
      const ph = svgPlaceholderSimple();
      const arr = res.b_images || [];
      for(let i=1;i<=4;i++){
        const el = $(`infer-b-${i}`);
        const item = arr[i-1];
        el.src = (item && item.data_url) ? item.data_url : ph;
      }
    }else if(res.b_status === "not_found"){
      setInferTrainPlaceholders("B", "not_found");
    }else{
      setInferTrainPlaceholders("B", "simple");
    }

    // note
    const aPath = res.domain_a_dir || "";
    const bPath = res.domain_b_dir || "";
    if(aPath || bPath){
      const aTxt = aPath ? `A: ${aPath}` : "A: —";
      const bTxt = bPath ? `B: ${bPath}` : "B: —";
      setInferTrainNote(`${aTxt} | ${bTxt}`);
    }else{
      setInferTrainNote("");
    }

  }catch(e){
    setInferTrainPlaceholders("ALL", "simple");
    setInferTrainNote("Ошибка при получении датасетов из модели.");
  }
}

let inferPreviewDebounce = null;
function scheduleInferTrainingPreview(){
  if(inferPreviewDebounce) clearTimeout(inferPreviewDebounce);
  inferPreviewDebounce = setTimeout(refreshInferTrainingPreview, 150);
}

/* ---------- RESUME details ---------- */
function setResumeInlineHtml(html){
  const el = $("resume-inline-project");
  if(el) el.innerHTML = html || "";
}
function setResumeParamsHtml(html){
  const el = $("resume-card-params");
  if(el) el.innerHTML = html || "";
}

function resetResumeCards(){
  setResumeInlineHtml(
    section("—", `<div class="kv">${kvRow("Статус", `<span class="muted">Выбери чекпоинт</span>`)}</div>`)
  );
  setResumeParamsHtml(`
    <div class="resume-info">
      ${section("—", `<div class="kv">${kvRow("Статус", `<span class="muted">Выбери чекпоинт</span>`)}</div>`)}
    </div>
    <div class="resume-info">
      ${section("—", `<div class="kv">${kvRow("Статус", `<span class="muted">Выбери чекпоинт</span>`)}</div>`)}
    </div>
  `);
}

function renderInlineProject(res){
  const epoch = (res.epoch !== undefined && res.epoch !== null) ? String(res.epoch) : "—";

  const summaryBadges = `<div class="badges">
    ${badge(`epoch ${epoch}`)}
    ${badge(String(res.device || "—"))}
    ${badge(`img ${res.image_size ?? "—"}`)}
    ${badge(`batch ${res.batch_size ?? "—"}`)}
  </div>`;

  const pathsKv =
    `<div class="kv">
      ${kvRow("Project", `<span class="kv-val">${escapeHtml(res.project_dir || "—")}</span>`)}
      ${kvRow("Domain A", `<span class="kv-val">${escapeHtml(res.domain_a_dir || "—")}</span>`)}
      ${kvRow("Domain B", `<span class="kv-val">${escapeHtml(res.domain_b_dir || "—")}</span>`)}
    </div>`;

  return `
    <div class="resume-info">
      ${section("Сводка", summaryBadges)}
      ${section("Пути", pathsKv)}
    </div>
  `;
}

function renderParamsTwoColumns(res){
  const device = res.device || "—";
  const imgSize = (res.image_size ?? "—");
  const batch = (res.batch_size ?? "—");
  const resb = (res.residual_blocks ?? "—");

  const lr = res.lr ?? "—";
  const dStart = res.lr_decay_start ?? "—";
  const dEnd = res.lr_decay_end ?? "—";
  const finalRatio = res.final_lr_ratio ?? "—";

  const lc = res.lambda_cycle ?? "—";
  const li = res.lambda_identity ?? "—";

  const replayOn = !!res.use_replay_buffer;
  const replaySize = res.replay_buffer_size ?? "—";
  const gclip = res.gradient_clip_norm ?? "—";

  const dropoutOn = !!res.use_dropout;
  const dropoutP = res.dropout_p ?? "—";

  const earlyOn = !!res.early_stopping;
  const patience = res.early_stopping_patience ?? "—";
  const mindelta = res.early_stopping_min_delta ?? "—";
  const metric = res.early_stopping_metric ?? "—";

  const rightSummary = `<div class="badges">
    ${badge(String(device))}
    ${badge(`img ${imgSize}`)}
    ${badge(`batch ${batch}`)}
    ${badge(`res ${resb}`)}
    ${badge(`LR ${fmtNum(lr, 6)}`)}
  </div>`;

  const rightMain =
    `<div class="kv">
      ${kvRow("Device", badge(device))}
      ${kvRow("Image size", badge(String(imgSize)))}
      ${kvRow("Batch size", badge(String(batch)))}
      ${kvRow("Residual blocks", badge(String(resb)))}
      ${kvRow("Epochs (total)", badge(String(res.epochs_total ?? "—")))}
    </div>`;

  const rightOpt =
    `<div class="kv">
      ${kvRow("LR", badge(`LR ${fmtNum(lr, 6)}`))}
      ${kvRow("LR decay", `<div class="badges">
        ${badge(`start ${dStart}`)}
        ${badge(`end ${dEnd}`)}
        ${badge(`final ${finalRatio}`)}
      </div>`)}
    </div>`;

  const leftLoss =
    `<div class="kv">
      ${kvRow("λ cycle", badge(String(lc)))}
      ${kvRow("λ identity", badge(String(li)))}
    </div>`;

  const leftStab =
    `<div class="kv">
      ${kvRow("Replay buffer", onOffBadge(replayOn))}
      ${replayOn ? kvRow("Replay size", badge(String(replaySize))) : ""}
      ${kvRow("Gradient clip", badge(String(gclip)))}
    </div>`;

  const leftReg =
    `<div class="kv">
      ${kvRow("Dropout", onOffBadge(dropoutOn))}
      ${dropoutOn ? kvRow("Dropout p", badge(String(dropoutP))) : ""}
    </div>`;

  const leftEarly =
    `<div class="kv">
      ${kvRow("Enabled", onOffBadge(earlyOn))}
      ${earlyOn ? kvRow("Patience", badge(String(patience))) : ""}
      ${earlyOn ? kvRow("Min delta", badge(String(mindelta))) : ""}
      ${earlyOn ? kvRow("Metric", badge(String(metric))) : ""}
    </div>`;

  const leftCol = `
    <div class="resume-info">
      ${section("Loss weights", leftLoss)}
      ${section("Стабилизация", leftStab)}
      ${section("Регуляризация", leftReg)}
      ${section("Early stopping", leftEarly)}
    </div>
  `;

  const rightCol = `
    <div class="resume-info">
      ${section("Сводка", rightSummary)}
      ${section("Основное", rightMain)}
      ${section("Оптимизация", rightOpt)}
    </div>
  `;

  return leftCol + rightCol;
}

async function refreshResumeDetails(){
  const path = ($("resume-ckpt").value || "").trim();
  if(!path){
    resetResumeCards();
    return;
  }

  try{
    const res = await window.pywebview.api.get_resume_details(path);
    if(!res || !res.ok){
      resetResumeCards();
      return;
    }
    setResumeInlineHtml(renderInlineProject(res));
    setResumeParamsHtml(renderParamsTwoColumns(res));
  }catch(e){
    resetResumeCards();
  }
}

let resumeDebounce = null;
function scheduleResumeRefresh(){
  if(resumeDebounce) clearTimeout(resumeDebounce);
  resumeDebounce = setTimeout(refreshResumeDetails, 200);
}

/* ---------- CUDA ---------- */
async function getCudaStatus(){
  try{
    const res = await window.pywebview.api.get_cuda_status();
    if(!res.ok) return { available:false, message:"CUDA status error" };
    return res;
  }catch(e){
    return { available:false, message: String(e) };
  }
}

async function enforceCudaSelection(selectEl){
  if(selectEl.value !== "cuda") return true;
  const st = await getCudaStatus();
  if(!st.available){
    alert(st.message);
    selectEl.value = "cpu";
    return false;
  }
  return true;
}

/* ---------- File pickers ---------- */
async function pickFolderInto(inputEl, title){
  const start = (inputEl.value || "").trim();
  const res = await window.pywebview.api.pick_folder(title || "Select folder", start);
  if(res && res.ok && res.path){
    inputEl.value = res.path;
    try{ inputEl.dispatchEvent(new Event("change")); }catch(_){}
    return;
  }
  if(res && res.canceled) return;
  if(res && res.error) alert(res.error);
}

async function pickFileInto(inputEl, title, kind){
  const start = (inputEl.value || "").trim();
  const res = await window.pywebview.api.pick_file(title || "Select file", start, kind || "any");
  if(res && res.ok && res.path){
    inputEl.value = res.path;
    try{ inputEl.dispatchEvent(new Event("change")); }catch(_){}
    return;
  }
  if(res && res.canceled) return;
  if(res && res.error) alert(res.error);
}

function getStatsToSave(){
  const out = [];
  if ($("stat-losses").checked) out.push("losses_csv");
  if ($("stat-lr").checked) out.push("lr_csv");
  if ($("stat-logs").checked) out.push("logs_txt");
  return out;
}

async function loadDefaults(){
  const train = await window.pywebview.api.get_default_train_config();
  if(train.ok){
    const c = train.config;

    $("train-imgsize").value = c.image_size;
    $("train-batch").value = c.batch_size;
    $("train-epochs").value = c.epochs;
    $("train-lr").value = c.lr;

    $("train-resblocks").value = c.residual_blocks;
    $("train-dropout").checked = c.use_dropout;
    $("train-dropoutp").value = c.dropout_p;
    $("train-clip").value = c.gradient_clip_norm;

    $("train-lcycle").value = c.lambda_cycle;
    $("train-lid").value = c.lambda_identity;

    $("train-decaystart").value = c.lr_decay_start;
    $("train-decayend").value = c.lr_decay_end;
    $("train-finalratio").value = c.final_lr_ratio;

    $("train-replay").checked = c.use_replay_buffer;
    $("train-replaysize").value = c.replay_buffer_size;

    $("train-early").checked = c.early_stopping;
    $("train-patience").value = c.early_stopping_patience;
    $("train-mindelta").value = c.early_stopping_min_delta;
    $("train-metric").value = c.early_stopping_metric;

    $("train-recursive").checked = c.recursive_search;
    $("train-device").value = c.device;

    $("train-proj-base").value = c.project_base_dir || "";
    $("train-proj-name").value = c.project_name || "";

    $("train-save-b2a").checked = (c.save_b2a_models !== false);

    const sts = c.stats_to_save || [];
    $("stat-losses").checked = sts.includes("losses_csv");
    $("stat-lr").checked = sts.includes("lr_csv");
    $("stat-logs").checked = sts.includes("logs_txt");

    $("train-save-ckpt").checked = (c.save_checkpoints !== false);
    $("train-ckpt-interval").value = c.checkpoint_interval_epochs ?? 1;
    $("train-ckpt-latest").checked = (c.keep_only_latest_checkpoint === true);

    $("train-model-interval-enabled").checked = (c.models_save_interval_enabled !== false);
    $("train-model-interval").value = c.models_save_interval_epochs ?? 1;

    $("train-model-keep-last-enabled").checked = (c.models_keep_last_enabled === true);
    $("train-model-keep-last-count").value = c.models_keep_last_count ?? 5;
  }

  const infer = await window.pywebview.api.get_default_infer_config();
  if(infer.ok){
    const c = infer.config;
    $("infer-imgsize").value = c.image_size;
    $("infer-device").value = c.device;
    $("infer-output").value = c.output_dir || "";
  }

  updateDropoutVisibility();
  updateEarlyStoppingVisibility();
  updateReplayVisibility();

  setDatasetPlaceholders(null);

  if(($("train-a").value || "").trim()){
    schedulePreviewRefreshSide("A");
  }
  if(($("train-b").value || "").trim()){
    schedulePreviewRefreshSide("B");
  }

  resetResumeCards();

  // infer training datasets preview
  setInferTrainPlaceholders("ALL", "simple");
  setInferTrainNote("Выбери модель — появятся 4 случайные картинки из датасетов, на которых она обучалась.");
}

/* ---------- Training / inference ---------- */
function buildTrainConfig(){
  const saveCkpt = $("train-save-ckpt").checked;
  let ckptInterval = toInt($("train-ckpt-interval").value) ?? 1;
  if(ckptInterval < 1) ckptInterval = 1;

  const modelIntervalEnabled = $("train-model-interval-enabled").checked;
  let modelInterval = toInt($("train-model-interval").value) ?? 1;
  if(modelInterval < 1) modelInterval = 1;

  const keepLastEnabled = $("train-model-keep-last-enabled").checked;
  let keepLastCount = toInt($("train-model-keep-last-count").value) ?? 5;
  if(keepLastCount < 1) keepLastCount = 1;

  return {
    project_base_dir: $("train-proj-base").value.trim(),
    project_name: $("train-proj-name").value.trim(),
    stats_to_save: getStatsToSave(),
    save_b2a_models: $("train-save-b2a").checked,

    domain_a_dir: $("train-a").value.trim(),
    domain_b_dir: $("train-b").value.trim(),

    image_size: toInt($("train-imgsize").value) ?? 256,
    batch_size: toInt($("train-batch").value) ?? 1,
    epochs: toInt($("train-epochs").value) ?? 50,

    max_images_a: toInt($("train-maxa").value),
    max_images_b: toInt($("train-maxb").value),
    recursive_search: $("train-recursive").checked,

    residual_blocks: toInt($("train-resblocks").value) ?? 9,
    use_dropout: $("train-dropout").checked,
    dropout_p: toFloat($("train-dropoutp").value) ?? 0.5,

    lr: toFloat($("train-lr").value) ?? 0.0002,
    lambda_cycle: toFloat($("train-lcycle").value) ?? 10.0,
    lambda_identity: toFloat($("train-lid").value) ?? 5.0,

    lr_decay_start: toInt($("train-decaystart").value) ?? 25,
    lr_decay_end: toInt($("train-decayend").value) ?? 50,
    final_lr_ratio: toFloat($("train-finalratio").value) ?? 0.05,

    use_replay_buffer: $("train-replay").checked,
    replay_buffer_size: toInt($("train-replaysize").value) ?? 50,
    gradient_clip_norm: toFloat($("train-clip").value) ?? 0.0,

    early_stopping: $("train-early").checked,
    early_stopping_patience: toInt($("train-patience").value) ?? 5,
    early_stopping_min_delta: toFloat($("train-mindelta").value) ?? 0.0,
    early_stopping_metric: $("train-metric").value,

    device: $("train-device").value,

    save_checkpoints: saveCkpt,
    checkpoint_interval_epochs: ckptInterval,
    keep_only_latest_checkpoint: $("train-ckpt-latest").checked,

    models_save_interval_enabled: modelIntervalEnabled,
    models_save_interval_epochs: modelInterval,
    models_keep_last_enabled: keepLastEnabled,
    models_keep_last_count: keepLastCount
  };
}

function buildInferConfig(){
  return {
    model_path: $("infer-model").value.trim(),
    input_path: $("infer-input").value.trim(),
    output_dir: $("infer-output").value.trim(),
    image_size: toInt($("infer-imgsize").value) ?? 256,
    device: $("infer-device").value,
    direction: "A2B"
  };
}

function setConsole(lines){
  const el = $("train-console");
  el.textContent = lines.join("\n");
  el.scrollTop = el.scrollHeight;
}
function setResumeConsole(lines){
  const el = $("resume-console");
  if(!el) return;
  el.textContent = lines.join("\n");
  el.scrollTop = el.scrollHeight;
}
function setProgress(pct){
  $("train-progress").style.width = `${pct}%`;
}

let pollTimer = null;

async function pollTraining(){
  const res = await window.pywebview.api.get_training_status();
  if(!res.ok) return;

  const st = res.status;
  const running = res.running;

  if(!st){
    setProgress(0);
    $("train-stats").textContent = "Status: Idle";
    return;
  }

  const logs = st.logs || [];
  setConsole(logs);
  setResumeConsole(logs);

  const step = st.step || 0;
  const total = st.total_steps || 1;
  const pct = Math.max(0, Math.min(100, (step / total) * 100));
  setProgress(pct);

  const lr = st.lr ?? null;
  const losses = st.last_losses || {};

  const statusText = running ? "Training" : (st.early_stopped ? "Idle (early stopped)" : "Idle");
  const lrText = lr !== null ? lr.toFixed(6) : "—";

  const stats =
    `Status: ${statusText} | ` +
    `LR: ${lrText} | ` +
    `Epoch: ${st.epoch} | Step: ${step}/${total} | ` +
    `G_total: ${(losses.G_total ?? 0).toFixed(4)} | D_total: ${(losses.D_total ?? 0).toFixed(4)} | ` +
    `cycle: ${(losses.cycle ?? 0).toFixed(4)} | id: ${(losses.identity ?? 0).toFixed(4)} | adv: ${(losses.adv ?? 0).toFixed(4)}`;

  $("train-stats").textContent = stats;
}

async function startTraining(){
  const ok = await enforceCudaSelection($("train-device"));
  if(!ok) return;

  const cfg = buildTrainConfig();

  if(!cfg.project_base_dir || !cfg.project_name){
    alert("Укажи Project base folder и Project name.");
    return;
  }
  if(!cfg.domain_a_dir || !cfg.domain_b_dir){
    alert("Укажи Domain A и Domain B.");
    return;
  }

  $("train-stats").textContent = "Status: Starting…";
  const res = await window.pywebview.api.start_training(JSON.stringify(cfg));
  if(!res.ok){
    $("train-stats").textContent = "Status: Error";
    alert(res.error);
    return;
  }

  $("train-run-dir").textContent = `Project dir: ${res.project_dir}`;

  if(pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(pollTraining, 700);
}

async function stopTraining(){
  const res = await window.pywebview.api.stop_training();
  if(!res.ok){
    alert(res.error);
    return;
  }
}

async function refreshInferModelInfo(){
  const path = ($("infer-model").value || "").trim();
  const infoEl = $("infer-model-info");
  if(!infoEl) return;

  if(!path){
    infoEl.textContent = "";
    return;
  }

  try{
    const res = await window.pywebview.api.get_model_info(path);
    if(!res.ok){
      infoEl.textContent = "Не удалось прочитать модель: " + (res.error || "");
      return;
    }

    const src = res.has_meta ? "meta" : "state_dict";
    const msg = res.has_meta ? "OK" : "Legacy (inferred)";
    infoEl.textContent = `Модель прочитана (${src}): ${msg}.`;
  }catch(e){
    infoEl.textContent = "Не удалось прочитать модель: " + String(e);
  }
}

async function runInference(){
  const ok = await enforceCudaSelection($("infer-device"));
  if(!ok) return;

  await refreshInferModelInfo();

  const cfg = buildInferConfig();
  if(!cfg.model_path || !cfg.input_path){
    alert("Укажи model_path и input_path.");
    return;
  }
  if(!cfg.output_dir){
    alert("Output folder обязателен.");
    return;
  }

  $("infer-result").textContent = "Running…";
  const res = await window.pywebview.api.run_inference(JSON.stringify(cfg));
  if(!res.ok){
    $("infer-result").textContent = "Error: " + res.error;
    alert(res.error);
    return;
  }

  $("infer-result").textContent = `Saved ${res.count} images to: ${res.output_dir}`;

  const gal = $("infer-gallery");
  gal.innerHTML = "";
  (res.previews || []).forEach(p => {
    const div = document.createElement("div");
    div.className = "item";
    div.innerHTML = `
      <div class="pair">
        <img src="${p.in_img}" />
        <img src="${p.out_img}" />
      </div>
      <div class="cap">in: ${p.in_path}<br/>out: ${p.out_path}</div>
    `;
    gal.appendChild(div);
  });
}

/* -------- RESUME -------- */
async function startResume(){
  const ckpt = ($("resume-ckpt").value || "").trim();
  if(!ckpt){
    alert("Выбери чекпоинт.");
    return;
  }

  $("train-stats").textContent = "Status: Resuming…";
  const res = await window.pywebview.api.resume_training(ckpt);
  if(!res.ok){
    $("train-stats").textContent = "Status: Error";
    alert(res.error);
    return;
  }

  if(pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(pollTraining, 700);
}

window.addEventListener("pywebviewready", async () => {
  await loadDefaults();

  $("btn-train-start").addEventListener("click", startTraining);
  $("btn-train-stop").addEventListener("click", stopTraining);
  $("btn-infer-run").addEventListener("click", runInference);

  $("btn-pick-train-proj-base").addEventListener("click", async () => {
    await pickFolderInto($("train-proj-base"), "Select project base folder");
  });

  $("btn-pick-train-a").addEventListener("click", async () => {
    await pickFolderInto($("train-a"), "Select Domain A folder");
    schedulePreviewRefreshSide("A");
  });

  $("btn-pick-train-b").addEventListener("click", async () => {
    await pickFolderInto($("train-b"), "Select Domain B folder");
    schedulePreviewRefreshSide("B");
  });

  $("train-a").addEventListener("change", () => schedulePreviewRefreshSide("A"));
  $("train-a").addEventListener("blur", () => schedulePreviewRefreshSide("A"));

  $("train-b").addEventListener("change", () => schedulePreviewRefreshSide("B"));
  $("train-b").addEventListener("blur", () => schedulePreviewRefreshSide("B"));

  $("btn-pick-infer-model").addEventListener("click", async () => {
    await pickFileInto($("infer-model"), "Select generator model (.pth)", "pth");
    await refreshInferModelInfo();
    scheduleInferTrainingPreview();
  });

  $("infer-model").addEventListener("change", async () => {
    await refreshInferModelInfo();
    scheduleInferTrainingPreview();
  });
  $("infer-model").addEventListener("blur", async () => {
    await refreshInferModelInfo();
    scheduleInferTrainingPreview();
  });

  $("btn-pick-infer-input-file").addEventListener("click", async () => {
    await pickFileInto($("infer-input"), "Select input image file", "image");
  });
  $("btn-pick-infer-input-folder").addEventListener("click", async () => {
    await pickFolderInto($("infer-input"), "Select input folder");
  });

  $("btn-pick-infer-output").addEventListener("click", async () => {
    await pickFolderInto($("infer-output"), "Select output folder");
  });

  // Resume UI
  $("btn-pick-resume-ckpt").addEventListener("click", async () => {
    await pickFileInto($("resume-ckpt"), "Select checkpoint (.pth)", "pth");
    scheduleResumeRefresh();
  });
  $("btn-resume-start").addEventListener("click", startResume);
  $("btn-resume-stop").addEventListener("click", stopTraining);

  $("resume-ckpt").addEventListener("change", scheduleResumeRefresh);
  $("resume-ckpt").addEventListener("blur", scheduleResumeRefresh);

  // Device change checks
  $("train-device").addEventListener("change", async () => {
    await enforceCudaSelection($("train-device"));
  });
  $("infer-device").addEventListener("change", async () => {
    await enforceCudaSelection($("infer-device"));
  });

  // UI visibility toggles
  $("train-dropout").addEventListener("change", updateDropoutVisibility);
  $("train-early").addEventListener("change", updateEarlyStoppingVisibility);
  $("train-replay").addEventListener("change", updateReplayVisibility);

  updateDropoutVisibility();
  updateEarlyStoppingVisibility();
  updateReplayVisibility();

  if(pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(pollTraining, 700);

  // Initial infer datasets preview state
  scheduleInferTrainingPreview();
});