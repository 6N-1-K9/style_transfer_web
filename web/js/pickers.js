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
