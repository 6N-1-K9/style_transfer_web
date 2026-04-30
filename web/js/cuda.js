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
