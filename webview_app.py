import base64
import io
import json
import random
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import webview
from PIL import Image, ImageOps

from styler.config import InferenceConfig, TrainConfig
from styler.device import get_cuda_status
from styler.inference import infer_arch_from_state_dict, load_generator_payload, run_inference_images
from styler.trainer import CycleGANTrainer
from styler.utils import ensure_dir, list_images


APP_DIR = Path(__file__).resolve().parent
WEB_DIR = APP_DIR / "web"


def _pil_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"


def _ensure_dir_arg(start_dir: str) -> str:
    if not start_dir:
        return ""
    try:
        p = Path(start_dir).expanduser()
        if p.suffix:
            p = p.parent
        return str(p.resolve())
    except Exception:
        return ""


def _make_thumb_data_url(img_path: Path, size: int = 420) -> str:
    """
    Make a reasonably small preview thumbnail. We do center-crop to avoid "half image" effects.
    """
    img = Image.open(img_path).convert("RGB")
    # Center crop + resize to square
    thumb = ImageOps.fit(img, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    return _pil_to_data_url(thumb, fmt="PNG")


class Api:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._trainer: Optional[CycleGANTrainer] = None
        self._train_thread: Optional[threading.Thread] = None
        self._stop_flag = False
        self._last_project_dir: Optional[Path] = None
        self._window: Optional[webview.Window] = None

    def set_window(self, window: webview.Window) -> None:
        self._window = window

    def _get_window(self) -> Optional[webview.Window]:
        if self._window is not None:
            return self._window
        try:
            if webview.windows:
                return webview.windows[0]
        except Exception:
            pass
        return None

    # ---------- File dialogs ----------
    def pick_folder(self, title: str = "Select folder", start_dir: str = "") -> Dict[str, Any]:
        window = self._get_window()
        if window is None:
            return {"ok": False, "error": "Window not ready"}

        directory = _ensure_dir_arg(start_dir)
        try:
            result = window.create_file_dialog(webview.FileDialog.FOLDER, directory=directory, allow_multiple=False)
        except Exception as e:
            return {"ok": False, "error": f"Dialog failed: {e}"}
        if not result:
            return {"ok": False, "canceled": True}
        return {"ok": True, "path": result[0]}

    def pick_file(self, title: str = "Select file", start_dir: str = "", kind: str = "any") -> Dict[str, Any]:
        window = self._get_window()
        if window is None:
            return {"ok": False, "error": "Window not ready"}

        directory = _ensure_dir_arg(start_dir)

        if kind == "pth":
            file_types = [("PyTorch checkpoint/model", "*.pth;*.pt"), ("All files", "*.*")]
            allowed_ext = {".pth", ".pt"}
        elif kind == "image":
            file_types = [
                ("Images", "*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.tif;*.tiff"),
                ("All files", "*.*"),
            ]
            allowed_ext = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
        else:
            file_types = [("All files", "*.*")]
            allowed_ext = None

        try:
            result = window.create_file_dialog(
                webview.FileDialog.OPEN,
                directory=directory,
                allow_multiple=False,
                file_types=file_types,
            )
        except Exception:
            try:
                result = window.create_file_dialog(webview.FileDialog.OPEN, directory=directory, allow_multiple=False)
            except Exception as e:
                return {"ok": False, "error": f"Dialog failed: {e}"}

        if not result:
            return {"ok": False, "canceled": True}

        chosen = result[0]
        if allowed_ext is not None:
            ext = Path(chosen).suffix.lower()
            if ext not in allowed_ext:
                return {"ok": False, "error": f"Неверное расширение {ext}. Нужно: {', '.join(sorted(allowed_ext))}"}

        return {"ok": True, "path": chosen}

    # ---------- System ----------
    def ping(self) -> Dict[str, Any]:
        return {"ok": True}

    def get_cuda_status(self) -> Dict[str, Any]:
        st = get_cuda_status()
        return {
            "ok": True,
            "available": st.available,
            "message": st.message,
            "device_count": st.device_count,
            "device_name": st.device_name,
            "torch_cuda_version": st.torch_cuda_version,
        }

    def list_images(self, folder: str) -> Dict[str, Any]:
        p = Path(folder).expanduser().resolve()
        if not p.exists():
            return {"ok": False, "error": "Folder not found"}
        imgs = list_images(p, recursive=True)
        return {"ok": True, "count": len(imgs), "images": [str(x) for x in imgs[:2000]]}

    def get_dataset_preview(self, folder: str, count: int = 3) -> Dict[str, Any]:
        """
        Returns N random image thumbnails from a folder (recursive search).
        Used for UI dataset preview. If folder doesn't exist or has no images, returns ok=False.
        """
        try:
            p = Path(folder).expanduser().resolve()
            if not p.exists():
                return {"ok": False, "error": "Folder not found"}

            imgs = list_images(p, recursive=True)
            if not imgs:
                return {"ok": False, "error": "No images found"}

            k = max(1, min(int(count or 3), 6))
            chosen = random.sample(imgs, k=min(k, len(imgs)))

            out = []
            for ip in chosen:
                try:
                    out.append(
                        {
                            "path": str(ip),
                            "data_url": _make_thumb_data_url(ip, size=420),
                        }
                    )
                except Exception:
                    continue

            if not out:
                return {"ok": False, "error": "Failed to load images"}

            # if we couldn't load enough, still ok; UI will fill placeholders
            return {"ok": True, "images": out}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ---------- Defaults ----------
    def get_default_train_config(self) -> Dict[str, Any]:
        cfg = TrainConfig()
        return {"ok": True, "config": asdict(cfg)}

    def get_default_infer_config(self) -> Dict[str, Any]:
        cfg = InferenceConfig()
        return {"ok": True, "config": asdict(cfg)}

    # ---------- Training ----------
    def start_training(self, config_json: str) -> Dict[str, Any]:
        with self._lock:
            if self._train_thread and self._train_thread.is_alive():
                return {"ok": False, "error": "Training already running"}

            self._stop_flag = False

            try:
                cfg_dict = json.loads(config_json)
                cfg = TrainConfig(**cfg_dict)
            except Exception as e:
                return {"ok": False, "error": f"Invalid config: {e}"}

            base = (cfg.project_base_dir or "").strip()
            name = (cfg.project_name or "").strip()
            if not base:
                return {"ok": False, "error": "Выбери базовую папку для проекта (Project base folder)."}
            if not name:
                return {"ok": False, "error": "Введи имя проекта (Project name)."}

            project_dir = Path(base).expanduser().resolve() / name
            if project_dir.exists():
                return {"ok": False, "error": f"Папка проекта уже существует: {project_dir}"}

            if (cfg.device or "").strip().lower() == "cuda":
                st = get_cuda_status()
                if not st.available:
                    return {"ok": False, "error": st.message}

            ensure_dir(project_dir)
            ensure_dir(project_dir / "checkpoints")
            ensure_dir(project_dir / "models")
            ensure_dir(project_dir / "stats")
            ensure_dir(project_dir / "samples")

            self._last_project_dir = project_dir

            try:
                self._trainer = CycleGANTrainer(cfg, project_dir=project_dir)
            except Exception as e:
                return {"ok": False, "error": f"Failed to init trainer: {e}"}

            def _worker() -> None:
                assert self._trainer is not None
                try:
                    self._trainer.train(stop_checker=lambda: self._stop_flag)
                except Exception as e:
                    self._trainer._log(f"[ERROR] {e}")
                    self._stop_flag = True

            self._train_thread = threading.Thread(target=_worker, daemon=True)
            self._train_thread.start()

            return {"ok": True, "project_dir": str(project_dir)}

    def get_checkpoint_info(self, checkpoint_path: str) -> Dict[str, Any]:
        try:
            p = Path(checkpoint_path).expanduser().resolve()
            if not p.exists():
                return {"ok": False, "error": "Checkpoint file not found"}
            raw = torch.load(p, map_location="cpu")
            if not isinstance(raw, dict) or "config" not in raw or "epoch" not in raw:
                return {"ok": False, "error": "Invalid checkpoint format"}
            cfg = raw["config"]
            epoch = int(raw["epoch"])
            project_dir = None
            if p.parent.name == "checkpoints":
                project_dir = str(p.parent.parent)
            return {
                "ok": True,
                "epoch": epoch,
                "project_dir": project_dir,
                "domain_a_dir": cfg.get("domain_a_dir", ""),
                "domain_b_dir": cfg.get("domain_b_dir", ""),
                "device": cfg.get("device", "cpu"),
                "image_size": cfg.get("image_size", 256),
                "batch_size": cfg.get("batch_size", 1),
                "residual_blocks": cfg.get("residual_blocks", 9),
                "epochs_total": cfg.get("epochs", None),
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        try:
            p = Path(model_path).expanduser().resolve()
            if not p.exists():
                return {"ok": False, "error": "Model file not found"}

            meta, sd = load_generator_payload(p)

            if not meta:
                inf = infer_arch_from_state_dict(sd)
                return {
                    "ok": True,
                    "has_meta": False,
                    "message": "Legacy model file (no meta). Parameters inferred from state_dict.",
                    "residual_blocks": inf.get("residual_blocks"),
                    "use_dropout": inf.get("use_dropout"),
                    "dropout_p": None,
                    "image_size": None,
                }

            return {
                "ok": True,
                "has_meta": True,
                "residual_blocks": meta.get("residual_blocks"),
                "use_dropout": meta.get("use_dropout"),
                "dropout_p": meta.get("dropout_p"),
                "image_size": meta.get("image_size"),
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def resume_training(self, checkpoint_path: str) -> Dict[str, Any]:
        with self._lock:
            if self._train_thread and self._train_thread.is_alive():
                return {"ok": False, "error": "Training already running"}

            self._stop_flag = False

            try:
                ckpt_path = Path(checkpoint_path).expanduser().resolve()
            except Exception:
                return {"ok": False, "error": "Invalid checkpoint path"}

            if not ckpt_path.exists():
                return {"ok": False, "error": "Checkpoint file not found"}

            try:
                trainer = CycleGANTrainer.resume_from_checkpoint(ckpt_path)
            except Exception as e:
                return {"ok": False, "error": f"Failed to load checkpoint: {e}"}

            if (trainer.cfg.device or "").strip().lower() == "cuda":
                st = get_cuda_status()
                if not st.available:
                    return {"ok": False, "error": st.message}

            self._trainer = trainer
            self._last_project_dir = trainer.project_dir

            def _worker() -> None:
                assert self._trainer is not None
                try:
                    self._trainer.train(stop_checker=lambda: self._stop_flag)
                except Exception as e:
                    self._trainer._log(f"[ERROR] {e}")
                    self._stop_flag = True

            self._train_thread = threading.Thread(target=_worker, daemon=True)
            self._train_thread.start()

            return {
                "ok": True,
                "project_dir": str(trainer.project_dir),
                "resumed_from": str(ckpt_path),
            }

    def stop_training(self) -> Dict[str, Any]:
        with self._lock:
            if not (self._train_thread and self._train_thread.is_alive()):
                return {"ok": False, "error": "No active training"}
            self._stop_flag = True
            return {"ok": True}

    def get_training_status(self) -> Dict[str, Any]:
        with self._lock:
            running = bool(self._train_thread and self._train_thread.is_alive())
            if not self._trainer:
                return {"ok": True, "running": running, "status": None}

            status = self._trainer.get_status()
            status["running"] = running
            status["project_dir"] = str(self._last_project_dir) if self._last_project_dir else None
            return {"ok": True, "running": running, "status": status}

    # ---------- Inference ----------
    def run_inference(self, config_json: str) -> Dict[str, Any]:
        try:
            cfg_dict = json.loads(config_json)
            cfg = InferenceConfig(**cfg_dict)
        except Exception as e:
            return {"ok": False, "error": f"Invalid config: {e}"}

        out_dir_str = (cfg.output_dir or "").strip()
        if not out_dir_str:
            return {"ok": False, "error": "Output folder обязателен. Выбери папку для сохранения результатов."}

        if (cfg.device or "").strip().lower() == "cuda":
            st = get_cuda_status()
            if not st.available:
                return {"ok": False, "error": st.message}

        input_path = Path(cfg.input_path).expanduser().resolve()
        if not input_path.exists():
            return {"ok": False, "error": "Input path not found"}

        out_dir = Path(out_dir_str).expanduser().resolve()
        ensure_dir(out_dir)

        try:
            outputs = run_inference_images(cfg, output_dir=out_dir)
        except Exception as e:
            return {"ok": False, "error": f"Inference failed: {e}"}

        previews = []
        for pair in outputs[: min(12, len(outputs))]:
            try:
                img_in = Image.open(pair[0]).convert("RGB")
                img_out = Image.open(pair[1]).convert("RGB")
                previews.append(
                    {
                        "in_path": str(pair[0]),
                        "out_path": str(pair[1]),
                        "in_img": _pil_to_data_url(img_in),
                        "out_img": _pil_to_data_url(img_out),
                    }
                )
            except Exception:
                continue

        return {"ok": True, "output_dir": str(out_dir), "count": len(outputs), "previews": previews}


def run_web() -> None:
    api = Api()
    window = webview.create_window(
        title="Style Transfer CycleGAN - Web UI",
        url=str(Path(WEB_DIR / "index.html")),
        js_api=api,
        width=1200,
        height=800,
        min_size=(1000, 700),
    )
    api.set_window(window)

    gui = None
    try:
        import qtpy  # noqa: F401
        gui = "qt"
    except Exception:
        try:
            import gi  # noqa: F401
            gui = "gtk"
        except Exception:
            gui = None

    if gui:
        webview.start(debug=False, gui=gui)
    else:
        webview.start(debug=False)