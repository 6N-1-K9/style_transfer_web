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
    section("—", `<div class="kv">${kvRow("Status", `<span class="muted">Choose a checkpoint</span>`)}</div>`)
  );
  setResumeParamsHtml(`
    <div class="resume-info">
      ${section("—", `<div class="kv">${kvRow("Status", `<span class="muted">Choose a checkpoint</span>`)}</div>`)}
    </div>
    <div class="resume-info">
      ${section("—", `<div class="kv">${kvRow("Status", `<span class="muted">Choose a checkpoint</span>`)}</div>`)}
    </div>
  `);
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
      ${section("Summary", summaryBadges)}
      ${section("Paths", pathsKv)}
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
      ${kvRow("lambda_cycle", badge(String(lc)))}
      ${kvRow("lambda_identity", badge(String(li)))}
    </div>`;

  const leftStab =
    `<div class="kv">
      ${kvRow("use_replay_buffer", onOffBadge(replayOn))}
      ${replayOn ? kvRow("replay_buffer_size", badge(String(replaySize))) : ""}
      ${kvRow("gradient_clip_norm", badge(String(gclip)))}
    </div>`;

  const leftReg =
    `<div class="kv">
      ${kvRow("use_dropout", onOffBadge(dropoutOn))}
      ${dropoutOn ? kvRow("dropout_p", badge(String(dropoutP))) : ""}
    </div>`;

  const leftEarly =
    `<div class="kv">
      ${kvRow("early_stopping", onOffBadge(earlyOn))}
      ${earlyOn ? kvRow("early_stopping_patience", badge(String(patience))) : ""}
      ${earlyOn ? kvRow("early_stopping_min_delta", badge(String(mindelta))) : ""}
      ${earlyOn ? kvRow("early_stopping_metric", badge(String(metric))) : ""}
    </div>`;

  const leftCol = `
    <div class="resume-info">
      ${section("Loss weights", leftLoss)}
      ${section("Stabilization", leftStab)}
      ${section("Regularization", leftReg)}
      ${section("Early stopping", leftEarly)}
    </div>
  `;

  const rightCol = `
    <div class="resume-info">
      ${section("Summary", rightSummary)}
      ${section("Core", rightMain)}
      ${section("Optimization", rightOpt)}
    </div>
  `;

  return leftCol + rightCol;
}
