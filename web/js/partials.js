const PARTIALS = [
  { name: "training", path: "./partials/training.html" },
  { name: "continue", path: "./partials/continue.html" },
  { name: "inference", path: "./partials/inference.html" },
  { name: "about", path: "./partials/about.html" },
];

async function fetchPartial(path){
  try{
    const response = await fetch(path, { cache: "no-cache" });
    if(response.ok){
      return await response.text();
    }
  } catch(_) {
    // Fallback below is useful for local file:// launch modes.
  }

  return await new Promise((resolve, reject) => {
    try{
      const xhr = new XMLHttpRequest();
      xhr.open("GET", path, true);
      xhr.onload = () => {
        if(xhr.status === 0 || (xhr.status >= 200 && xhr.status < 300)){
          resolve(xhr.responseText);
        } else {
          reject(new Error(`Cannot load ${path}: ${xhr.status}`));
        }
      };
      xhr.onerror = () => reject(new Error(`Cannot load ${path}`));
      xhr.send();
    } catch(err){
      reject(err);
    }
  });
}

async function loadPartials(){
  const root = $("partials-root");
  if(!root) return;

  const chunks = [];
  for(const partial of PARTIALS){
    const html = await fetchPartial(partial.path);
    chunks.push(html);
  }
  root.innerHTML = chunks.join("\n");
}
