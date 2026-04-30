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

  if(!window.pywebview?.api?.get_model_info){
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

  setInferTrainPlaceholders("ALL", "simple");
  setInferTrainNote("Choose a model to preview training datasets (A → B).");

  setHelpLang("ru");
}
