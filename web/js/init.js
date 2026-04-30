window.addEventListener("pywebviewready", async () => {
  await loadPartials();
  bindTabs();
  await loadDefaults();

  $("btn-help-ru")?.addEventListener("click", () => setHelpLang("ru"));
  $("btn-help-en")?.addEventListener("click", () => setHelpLang("en"));

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

  $("btn-pick-resume-ckpt").addEventListener("click", async () => {
    await pickFileInto($("resume-ckpt"), "Select checkpoint (.pth)", "pth");
    scheduleResumeRefresh();
  });
  $("btn-resume-start").addEventListener("click", startResume);
  $("btn-resume-stop").addEventListener("click", stopTraining);

  $("resume-ckpt").addEventListener("change", scheduleResumeRefresh);
  $("resume-ckpt").addEventListener("blur", scheduleResumeRefresh);

  $("train-device").addEventListener("change", async () => {
    await enforceCudaSelection($("train-device"));
  });
  $("infer-device").addEventListener("change", async () => {
    await enforceCudaSelection($("infer-device"));
  });

  $("train-dropout").addEventListener("change", updateDropoutVisibility);
  $("train-early").addEventListener("change", updateEarlyStoppingVisibility);
  $("train-replay").addEventListener("change", updateReplayVisibility);

  updateDropoutVisibility();
  updateEarlyStoppingVisibility();
  updateReplayVisibility();

  if(pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(pollTraining, 700);

  scheduleInferTrainingPreview();
});
