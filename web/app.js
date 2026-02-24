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

/* ---------- Dataset preview ---------- */
function svgPlaceholder(){
  // square, no text
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
  <!-- simple "image" icon -->
  <g transform="translate(170,190)" fill="rgba(0,0,0,0.18)">
    <rect x="0" y="0" width="260" height="210" rx="18" ry="18" fill="rgba(0,0,0,0.06)" stroke="rgba(0,0,0,0.16)" />
    <circle cx="72" cy="70" r="18" />
    <path d="M30 180 L110 115 L160 155 L205 125 L235 180 Z" />
  </g>
</svg>`;
  return "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(svg)));
}

function setDatasetPlaceholders(side){
  // side: "A" | "B" | null (both)
  const ph = svgPlaceholder();
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
  // side: "A" or "B"
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
    const ph = svgPlaceholder();
    for(let i=1;i<=3;i++){
      const el = $(side === "A" ? `prev-a-${i}` : `prev-b-${i}`);
      const item = imgs[i-1];
      el.src = (item && item.data_url) ? item.data_url : ph;
    }
  }catch(e){
    setDatasetPlaceholders(side);
  }
}

// Debounce per side (so A refresh doesn't trigger B and vice versa)
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
    try{
      inputEl.dispatchEvent(new Event("change"));
    }catch(_){}
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
    try{
      inputEl.dispatchEvent(new Event("change"));
    }catch(_){}
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

    // checkpoint controls
    $("train-save-ckpt").checked = (c.save_checkpoints !== false);
    $("train-ckpt-interval").value = c.checkpoint_interval_epochs ?? 1;
    $("train-ckpt-latest").checked = (c.keep_only_latest_checkpoint === true);

    // model saving controls
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

  // Initial placeholders only. No auto refresh of both sides.
  setDatasetPlaceholders(null);

  // If defaults already contain dataset paths (rare), you may still want initial load:
  // We'll update each side only if the corresponding path is non-empty.
  if(($("train-a").value || "").trim()){
    schedulePreviewRefreshSide("A");
  }
  if(($("train-b").value || "").trim()){
    schedulePreviewRefreshSide("B");
  }
}

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
async function showCheckpointInfo(){
  const path = ($("resume-ckpt").value || "").trim();
  if(!path){
    alert("Выбери чекпоинт.");
    return;
  }
  const res = await window.pywebview.api.get_checkpoint_info(path);
  if(!res.ok){
    $("resume-info").textContent = "Error: " + res.error;
    alert(res.error);
    return;
  }
  $("resume-info").textContent =
    `Epoch in checkpoint: ${res.epoch}. ` +
    `Project: ${res.project_dir || "(unknown)"} | ` +
    `A: ${res.domain_a_dir} | B: ${res.domain_b_dir} | ` +
    `img=${res.image_size}, batch=${res.batch_size}, res=${res.residual_blocks}, device=${res.device}`;
}

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

  $("resume-run-dir").textContent = `Project dir: ${res.project_dir} | resumed from: ${res.resumed_from}`;

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
    // IMPORTANT: refresh only A (no touching B)
    schedulePreviewRefreshSide("A");
  });

  $("btn-pick-train-b").addEventListener("click", async () => {
    await pickFolderInto($("train-b"), "Select Domain B folder");
    // IMPORTANT: refresh only B (no touching A)
    schedulePreviewRefreshSide("B");
  });

  // Manual edits: update only that side on change/blur
  $("train-a").addEventListener("change", () => schedulePreviewRefreshSide("A"));
  $("train-a").addEventListener("blur", () => schedulePreviewRefreshSide("A"));

  $("train-b").addEventListener("change", () => schedulePreviewRefreshSide("B"));
  $("train-b").addEventListener("blur", () => schedulePreviewRefreshSide("B"));

  $("btn-pick-infer-model").addEventListener("click", async () => {
    await pickFileInto($("infer-model"), "Select generator model (.pth)", "pth");
    await refreshInferModelInfo();
  });

  $("infer-model").addEventListener("change", refreshInferModelInfo);
  $("infer-model").addEventListener("blur", refreshInferModelInfo);

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
  });
  $("btn-resume-info").addEventListener("click", showCheckpointInfo);
  $("btn-resume-start").addEventListener("click", startResume);
  $("btn-resume-stop").addEventListener("click", stopTraining);

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

  // Apply once at start
  updateDropoutVisibility();
  updateEarlyStoppingVisibility();
  updateReplayVisibility();

  if(pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(pollTraining, 700);
});