/* ===================== ABOUT HELP (RU/EN) ===================== */
const HELP = {
  ru: {
    title: "Справка по приложению",
    subtitle: "CycleGAN Style Transfer “Комбайн”: обучение, продолжение обучения и инференс. Навигация слева. Язык справки переключается кнопками RU/EN.",
    navTitle: "Разделы",
    sections: [
      {
        id: "purpose",
        title: "Назначение",
        html: `
          <p>Приложение предназначено для обучения и применения <b>CycleGAN</b> для переноса стиля.</p>
          <div class="help-grid-2">
            <div class="help-section">
              <h3>Domain A</h3>
              <p>Контент/предметы — то, что будет стилизоваться (A → B).</p>
            </div>
            <div class="help-section">
              <h3>Domain B</h3>
              <p>Стиль — визуальная “цель”, в которую переводится Domain A.</p>
            </div>
          </div>
          <div class="help-callout warn">Важно: датасеты не копируются в папку проекта — приложение хранит только пути.</div>
        `
      },
      {
        id: "project_structure",
        title: "Папка проекта и файлы",
        html: `
          <p>При обучении создаётся папка проекта: <code>&lt;Project base folder&gt;/&lt;Project name&gt;/</code>.</p>
          <ul>
            <li><code>checkpoints/</code> — checkpoints (для Resume).</li>
            <li><code>models/</code> — models (веса генераторов для inference).</li>
            <li><code>stats/</code> — statistics (CSV/TXT).</li>
            <li><code>samples/</code> — вспомогательные примеры (если используются).</li>
            <li><code>train_config.json</code> — конфигурация обучения проекта.</li>
          </ul>
          <div class="help-callout">
            <b>Checkpoint</b> — полное состояние обучения (генераторы, дискриминаторы, оптимизаторы).<br/>
            <b>Model</b> — веса генератора (обычно <code>G_A2B_...</code>) для inference.
          </div>
        `
      },
      {
        id: "tab_train",
        title: "Вкладка «Обучение»",
        html: `
          <p>Здесь создаётся проект и запускается обучение CycleGAN.</p>

          <div class="help-section">
            <h3>Проект</h3>
            <ul>
              <li><b>Project base folder</b> — где создать папку проекта.</li>
              <li><b>Project name</b> — имя папки проекта.</li>
              <li><b>Сохранять статистику</b> — какие файлы писать в <code>stats/</code> (<code>losses.csv</code>, <code>lr.csv</code>, <code>logs.txt</code>).</li>
            </ul>
          </div>

          <div class="help-section">
            <h3>Датасеты</h3>
            <ul>
              <li><b>Domain A</b> — папка с контентом.</li>
              <li><b>Domain B</b> — папка со стилем.</li>
              <li><b>Image size</b> — к какому квадратному размеру приводятся изображения.</li>
              <li><b>Batch size</b> — сколько изображений обрабатывается за один шаг.</li>
              <li><b>Max images A / Max images B</b> — ограничить число изображений (для быстрых тестов).</li>
              <li><b>Recursive search</b> — искать изображения в подпапках.</li>
            </ul>
          </div>

          <div class="help-section">
            <h3>Модель</h3>
            <ul>
              <li><b>Residual blocks</b> — “ёмкость” генератора (больше = потенциально качественнее, но тяжелее).</li>
              <li><b>Device</b> — <code>cpu</code>/<code>cuda</code> (если CUDA недоступна, приложение предупредит).</li>
              <li><b>Use dropout</b> — регуляризация генератора.</li>
              <li><b>Dropout p</b> — вероятность dropout (параметр виден только когда включено).</li>
              <li><b>Gradient clip (0 = off)</b> — ограничение нормы градиента для стабильности.</li>
            </ul>
          </div>

          <div class="help-section">
            <h3>Обучение</h3>
            <ul>
              <li><b>Epochs</b> — число эпох.</li>
              <li><b>LR</b> — learning rate.</li>
              <li><b>lambda_cycle</b> — вес cycle-loss (A→B→A и B→A→B).</li>
              <li><b>lambda_identity</b> — вес identity-loss (уменьшает лишние изменения).</li>
              <li><b>LR decay start / LR decay end</b> + <b>Final LR ratio</b> — расписание снижения LR.</li>
              <li><b>Replay buffer</b> + <b>Replay buffer size</b> — стабилизация обучения дискриминаторов (size скрывается, если выключено).</li>
              <li><b>Early stopping</b> — ранняя остановка (параметры скрываются, если выключено).</li>
              <li><b>Patience</b>, <b>Min delta</b>, <b>Metric</b> — настройки ранней остановки.</li>
            </ul>
          </div>

          <div class="help-section">
            <h3>Сохранение</h3>
            <ul>
              <li><b>Save checkpoints</b> — сохранять checkpoints в <code>checkpoints/</code>.</li>
              <li><b>Checkpoint interval (epochs)</b> — интервал сохранения.</li>
              <li><b>Keep only latest checkpoint</b> — хранить только последний checkpoint.</li>
              <li><b>Save models by interval</b> — сохранять models в <code>models/</code>.</li>
              <li><b>Model save interval (epochs)</b> — интервал сохранения моделей.</li>
              <li><b>Keep last models</b> / <b>Keep last count</b> — хранить только последние K моделей.</li>
              <li><b>Save B2A models</b> — сохранять генератор B→A (можно выключить для экономии места).</li>
            </ul>
          </div>
        `
      },
      {
        id: "tab_resume",
        title: "Вкладка «Продолжить» (Resume)",
        html: `
          <p>Продолжение обучения из checkpoint: <code>project/checkpoints/epoch_XXXX.pth</code>.</p>
          <ul>
            <li>После выбора checkpoint справа отображаются параметры обучения и пути датасетов.</li>
            <li><b>Resume training</b> — продолжить обучение.</li>
            <li><b>Stop</b> — остановить процесс.</li>
          </ul>
          <div class="help-callout warn">
            В Resume используется сохранённая конфигурация проекта/чекпоинта, чтобы избежать несовместимости.
          </div>
        `
      },
      {
        id: "tab_infer",
        title: "Вкладка «Инференс»",
        html: `
          <p>Применение обученного генератора к файлу или папке изображений.</p>
          <div class="help-section">
            <h3>Модель</h3>
            <ul>
              <li><b>Generator .pth</b> — файл генератора (обычно <code>G_A2B_...</code>).</li>
              <li><b>Image size</b> — размер, к которому приводится вход перед прогоном.</li>
              <li><b>Device</b> — <code>cpu</code>/<code>cuda</code>.</li>
            </ul>
          </div>
          <div class="help-section">
            <h3>Вход / Выход</h3>
            <ul>
              <li><b>Input (file or folder)</b> — файл или папка.</li>
              <li><b>Output folder</b> — обязательная папка сохранения результата.</li>
            </ul>
          </div>
          <div class="help-section">
            <h3>Training datasets preview</h3>
            <p>Справа отображается 2×2 превью датасетов Domain A и Domain B, на которых обучалась модель. Если папка датасета отсутствует — показывается “Dataset not found”.</p>
          </div>
        `
      },
      {
        id: "faq",
        title: "Частые вопросы",
        html: `
          <div class="help-section">
            <h3>CUDA недоступна</h3>
            <p>Если PyTorch установлен без CUDA или нет совместимого GPU/драйверов — используйте <code>cpu</code>. Приложение предупредит при выборе <code>cuda</code>.</p>
          </div>
          <div class="help-section">
            <h3>“Dataset not found” в inference</h3>
            <p>Model хранит пути датасетов из проекта. Если папки перемещены/удалены — показывается заглушка. На inference это не влияет.</p>
          </div>
        `
      }
    ]
  },

  en: {
    title: "Application Help",
    subtitle: "CycleGAN Style Transfer “Combine”: training, resume training, and inference. Use the left navigation; switch language with RU/EN buttons.",
    navTitle: "Sections",
    sections: [
      {
        id: "purpose",
        title: "Purpose",
        html: `
          <p>This application trains and runs a <b>CycleGAN</b> model for style transfer.</p>
          <div class="help-grid-2">
            <div class="help-section">
              <h3>Domain A</h3>
              <p>Content/objects — what you want to stylize (A → B).</p>
            </div>
            <div class="help-section">
              <h3>Domain B</h3>
              <p>Style domain — the target visual style.</p>
            </div>
          </div>
          <div class="help-callout warn">Important: datasets are not copied into the project folder — only paths are stored.</div>
        `
      },
      {
        id: "project_structure",
        title: "Project folder & files",
        html: `
          <p>Training creates a project directory: <code>&lt;base&gt;/&lt;project_name&gt;/</code>.</p>
          <ul>
            <li><code>checkpoints/</code> — training checkpoints (for Resume).</li>
            <li><code>models/</code> — generator weights for inference.</li>
            <li><code>stats/</code> — logs/metrics (CSV/TXT).</li>
            <li><code>samples/</code> — auxiliary samples (if used).</li>
            <li><code>train_config.json</code> — project training configuration.</li>
          </ul>
          <div class="help-callout">
            <b>Checkpoint</b> = full training state (generators, discriminators, optimizers).<br/>
            <b>Model</b> = generator weights (usually <code>G_A2B_...</code>) for inference.
          </div>
        `
      },
      {
        id: "tab_train",
        title: "Training tab",
        html: `
          <p>Create a project and start CycleGAN training.</p>

          <div class="help-section">
            <h3>Project</h3>
            <ul>
              <li><b>Project base folder</b> — where to create the project directory.</li>
              <li><b>Project name</b> — project folder name.</li>
              <li><b>Save statistics</b> — which files to write into <code>stats/</code> (losses.csv, lr.csv, logs.txt).</li>
            </ul>
          </div>

          <div class="help-section">
            <h3>Datasets</h3>
            <ul>
              <li><b>Domain A</b> — content folder.</li>
              <li><b>Domain B</b> — style folder.</li>
              <li><b>Image size</b> — input size (images are resized to a square).</li>
              <li><b>Batch size</b> — number of images per step.</li>
              <li><b>Max images A/B</b> — limit images for quick experiments.</li>
              <li><b>Recursive search</b> — scan subfolders.</li>
            </ul>
          </div>

          <div class="help-section">
            <h3>Model</h3>
            <ul>
              <li><b>Residual blocks</b> — generator capacity (more = potentially better, but heavier).</li>
              <li><b>Device</b> — cpu/cuda (the app warns if CUDA is not available).</li>
              <li><b>Use dropout</b> — generator regularization.</li>
              <li><b>Dropout p</b> — dropout probability (visible only when enabled).</li>
              <li><b>Gradient clip</b> — gradient norm clipping for stability (0 = off).</li>
            </ul>
          </div>

          <div class="help-section">
            <h3>Training</h3>
            <ul>
              <li><b>Epochs</b> — total epochs.</li>
              <li><b>LR</b> — learning rate.</li>
              <li><b>lambda_cycle</b> — cycle-consistency loss weight.</li>
              <li><b>lambda_identity</b> — identity loss weight.</li>
              <li><b>LR decay start/end</b> and <b>Final LR ratio</b> — LR schedule.</li>
              <li><b>Replay buffer</b> and <b>size</b> — stabilizes discriminator training.</li>
              <li><b>Early stopping</b> — stop training when a metric stops improving.</li>
              <li><b>Patience</b>, <b>Min delta</b>, <b>Metric</b> — early stop controls.</li>
            </ul>
          </div>

          <div class="help-section">
            <h3>Saving</h3>
            <ul>
              <li><b>Save checkpoints</b> — store checkpoints in <code>checkpoints/</code>.</li>
              <li><b>Checkpoint interval</b> — interval (epochs).</li>
              <li><b>Keep only latest checkpoint</b> — keep only the newest checkpoint.</li>
              <li><b>Save models by interval</b> — store generator weights in <code>models/</code>.</li>
              <li><b>Model save interval</b> — interval (epochs).</li>
              <li><b>Keep last models</b> / <b>count</b> — keep only last K saved models.</li>
              <li><b>Save B2A models</b> — save B→A generator (can disable to save disk space).</li>
            </ul>
          </div>
        `
      },
      {
        id: "tab_resume",
        title: "Resume tab",
        html: `
          <p>Continue training from a checkpoint: <code>project/checkpoints/epoch_XXXX.pth</code>.</p>
          <ul>
            <li>After selecting a checkpoint, the UI shows training parameters and dataset paths.</li>
            <li><b>Resume training</b> — continue training.</li>
            <li><b>Stop</b> — stop the process.</li>
          </ul>
          <div class="help-callout warn">
            Resume uses the saved configuration from the project/checkpoint to avoid incompatibilities.
          </div>
        `
      },
      {
        id: "tab_infer",
        title: "Inference tab",
        html: `
          <p>Apply a trained generator to a file or a folder of images.</p>
          <div class="help-section">
            <h3>Model</h3>
            <ul>
              <li><b>Generator .pth</b> — generator weights (usually <code>G_A2B_...</code>).</li>
              <li><b>Image size</b> — input resize size for inference.</li>
              <li><b>Device</b> — cpu/cuda.</li>
            </ul>
          </div>
          <div class="help-section">
            <h3>Input / Output</h3>
            <ul>
              <li><b>Input</b> — image file or a folder.</li>
              <li><b>Output folder</b> — required output directory.</li>
            </ul>
          </div>
          <div class="help-section">
            <h3>Training datasets preview</h3>
            <p>The UI shows a 2×2 preview for domains A and B used during training. If folders are missing — “Dataset not found” placeholders are shown.</p>
          </div>
        `
      },
      {
        id: "faq",
        title: "FAQ",
        html: `
          <div class="help-section">
            <h3>CUDA is not available</h3>
            <p>If PyTorch is installed without CUDA support or GPU/drivers are missing, use <code>cpu</code>. The app warns when selecting <code>cuda</code>.</p>
          </div>
          <div class="help-section">
            <h3>“Dataset not found” in inference</h3>
            <p>The model stores dataset paths from the training project. If folders were moved/removed, placeholders appear. Inference still works.</p>
          </div>
        `
      }
    ]
  }
};

let HELP_LANG = "ru";
let helpObserver = null;
let helpFlashTimer = null;

function setHelpLang(lang){
  HELP_LANG = (lang === "en") ? "en" : "ru";

  const bRu = $("btn-help-ru");
  const bEn = $("btn-help-en");
  if(bRu) bRu.classList.toggle("active", HELP_LANG === "ru");
  if(bEn) bEn.classList.toggle("active", HELP_LANG === "en");

  renderHelp();
}

function flashHelpSection(secEl){
  if(!secEl) return;

  // remove old flash
  try{
    document.querySelectorAll(".help-section.flash").forEach(x => x.classList.remove("flash"));
  }catch(_){}

  secEl.classList.add("flash");
  if(helpFlashTimer) clearTimeout(helpFlashTimer);
  helpFlashTimer = setTimeout(() => {
    try{ secEl.classList.remove("flash"); }catch(_){}
  }, 900);
}

function renderHelp(){
  const data = HELP[HELP_LANG];
  if(!data) return;

  const titleEl = $("help-title");
  const subtitleEl = $("help-subtitle");
  const navEl = $("help-nav");
  const contentEl = $("help-content");

  if(titleEl) titleEl.textContent = data.title || "Help";
  if(subtitleEl) subtitleEl.textContent = data.subtitle || "";

  if(navEl){
    navEl.innerHTML = `
      <div class="nav-title">${escapeHtml(data.navTitle || "Sections")}</div>
      ${(data.sections || []).map(s => `
        <a href="#${escapeHtml(s.id)}" data-help-link="${escapeHtml(s.id)}">${escapeHtml(s.title)}</a>
      `).join("")}
    `;

    navEl.querySelectorAll("a[data-help-link]").forEach(a => {
      a.addEventListener("click", (ev) => {
        ev.preventDefault();
        const id = a.getAttribute("data-help-link");
        const target = document.getElementById(`help-sec-${id}`);
        if(target){
          target.scrollIntoView({ behavior:"smooth", block:"start" });
          // flash after scrolling starts
          setTimeout(() => flashHelpSection(target), 220);
        }
      });
    });
  }

  if(contentEl){
    contentEl.innerHTML = (data.sections || []).map(s => `
      <div class="help-section" id="help-sec-${escapeHtml(s.id)}" data-help-sec="${escapeHtml(s.id)}">
        <h3>${escapeHtml(s.title)}</h3>
        ${s.html || ""}
      </div>
    `).join("");
  }

  refreshHelpObserver();
}

function setActiveHelpNav(id){
  const navEl = $("help-nav");
  if(!navEl) return;
  navEl.querySelectorAll("a[data-help-link]").forEach(a => {
    a.classList.toggle("active", a.getAttribute("data-help-link") === id);
  });
}

function refreshHelpObserver(){
  try{
    if(helpObserver){
      helpObserver.disconnect();
      helpObserver = null;
    }
  }catch(_){}

  const contentEl = $("help-content");
  if(!contentEl) return;

  const sections = contentEl.querySelectorAll("[data-help-sec]");
  if(!sections || sections.length === 0) return;

  helpObserver = new IntersectionObserver((entries) => {
    let best = null;
    for(const e of entries){
      if(!e.isIntersecting) continue;
      if(!best || e.intersectionRatio > best.intersectionRatio){
        best = e;
      }
    }
    if(best){
      const id = best.target.getAttribute("data-help-sec");
      if(id) setActiveHelpNav(id);
    }
  }, {
    root: null,
    threshold: [0.15, 0.25, 0.35, 0.45, 0.6],
    rootMargin: "0px 0px -65% 0px"
  });

  sections.forEach(s => helpObserver.observe(s));

  const first = sections[0].getAttribute("data-help-sec");
  if(first) setActiveHelpNav(first);
}
