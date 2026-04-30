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
  const ph = (kind === "not_found") ? svgPlaceholderNotFound("Dataset not found") : svgPlaceholderSimple();

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
    setInferTrainNote("Choose a model to preview training datasets (A → B).");
    return;
  }

  try{
    const res = await window.pywebview.api.get_infer_training_datasets_preview(modelPath, 4);
    if(!res || !res.ok){
      setInferTrainPlaceholders("ALL", "simple");
      setInferTrainNote("Failed to read training dataset info from model.");
      return;
    }

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
    setInferTrainNote("Error while loading training dataset preview.");
  }
}

let inferPreviewDebounce = null;
function scheduleInferTrainingPreview(){
  if(inferPreviewDebounce) clearTimeout(inferPreviewDebounce);
  inferPreviewDebounce = setTimeout(refreshInferTrainingPreview, 150);
}
