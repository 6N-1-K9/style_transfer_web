import base64
import io
import json
import random
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import webview
from PIL import Image

from styler.config import InferenceConfig, TrainConfig
from styler.device import get_cuda_status
from styler.inference import run_inference_images
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


def _infer_project_dir_from_model_path(model_path: Path) -> Optional[Path]:
    """
    Expected:
      <project_dir>/models/G_A2B_epoch_XXXX.pth
    """
    try:
        if model_path.parent.name == "models":
            return model_path.parent.parent
    except Exception:
        pass
    return None


def _read_train_config_from_project(project_dir: Path) -> Optional[Dict[str, Any]]:
    try:
        cfg_path = project_dir / "train_config.json"
        if not cfg_path.exists():
            return None
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            return raw
    except Exception:
        return None
    return None


def _sample_images_as_data_urls(folder: Path, k: int) -> List[Dict[str, str]]:
    imgs = list_images(folder, recursive=True)
    if not imgs:
        return []

    if len(imgs) <= k:
        chosen = imgs
    else:
        chosen = random.sample(imgs, k)

    out: List[Dict[str, str]] = []
    for p in chosen:
        try:
            img = Image.open(p).convert("RGB")
            # downscale a bit for UI memory, keep square-ish cropping by center-crop
            # (we keep it simple: just thumbnail to 256x256 preserving aspect)
            img.thumbnail((256, 256))
            out.append({"path": str(p), "data_url": _pil_to_data_url(img)})
        except Exception:
            continue
    return out


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

    # ---------- Defaults ----------
    def get_default_train_config(self) -> Dict[str, Any]:
        cfg = TrainConfig()
        return {"ok": True, "config": asdict(cfg)}

    def get_default_infer_config(self) -> Dict[str, Any]:
        cfg = InferenceConfig()
        return {"ok": True, "config": asdict(cfg)}

    # ---------- Small helpers for UI (already used by train preview / resume) ----------
    def get_dataset_preview(self, folder: str, k: int = 3) -> Dict[str, Any]:
        try:
            p = Path(folder).expanduser().resolve()
            if not p.exists():
                return {"ok": False, "error": "Folder not found"}
            imgs = _sample_images_as_data_urls(p, int(k))
            return {"ok": True, "count": len(imgs), "images": imgs}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # NOTE: you likely already have get_model_info/get_resume_details in your repo.
    # If not — these calls will fail from JS.
    # (Оставляю как есть: в проекте они уже должны быть.)

    # ---------- NEW: Infer tab training datasets preview ----------
    def get_infer_training_datasets_preview(self, model_path: str, k: int = 4) -> Dict[str, Any]:
        """
        Returns:
          a_status: "ok" | "not_found" | "unknown"
          b_status: "ok" | "not_found" | "unknown"
          a_images: [{path, data_url}, ...]
          b_images: [{path, data_url}, ...]
          domain_a_dir, domain_b_dir (if known)
        """
        try:
            mp = Path(model_path).expanduser().resolve()
            if not mp.exists():
                return {"ok": False, "error": "Model file not found"}

            project_dir = _infer_project_dir_from_model_path(mp)
            domain_a_dir = ""
            domain_b_dir = ""

            if project_dir:
                cfg = _read_train_config_from_project(project_dir)
                if cfg:
                    domain_a_dir = str(cfg.get("domain_a_dir", "") or "")
                    domain_b_dir = str(cfg.get("domain_b_dir", "") or "")

            # If still empty — try reading from model dict (future-proof)
            if not domain_a_dir or not domain_b_dir:
                try:
                    raw = torch.load(mp, map_location="cpu")
                    if isinstance(raw, dict):
                        meta = raw.get("meta") or raw.get("metadata") or {}
                        if isinstance(meta, dict):
                            domain_a_dir = domain_a_dir or str(meta.get("domain_a_dir", "") or "")
                            domain_b_dir = domain_b_dir or str(meta.get("domain_b_dir", "") or "")
                except Exception:
                    pass

            out: Dict[str, Any] = {
                "ok": True,
                "domain_a_dir": domain_a_dir,
                "domain_b_dir": domain_b_dir,
                "a_status": "unknown",
                "b_status": "unknown",
                "a_images": [],
                "b_images": [],
            }

            # A
            if domain_a_dir:
                pa = Path(domain_a_dir).expanduser()
                if pa.exists() and pa.is_dir():
                    imgs_a = _sample_images_as_data_urls(pa.resolve(), int(k))
                    if imgs_a:
                        out["a_status"] = "ok"
                        out["a_images"] = imgs_a
                    else:
                        # directory exists but empty/no readable images -> treat as not_found-like UX
                        out["a_status"] = "not_found"
                else:
                    out["a_status"] = "not_found"

            # B
            if domain_b_dir:
                pb = Path(domain_b_dir).expanduser()
                if pb.exists() and pb.is_dir():
                    imgs_b = _sample_images_as_data_urls(pb.resolve(), int(k))
                    if imgs_b:
                        out["b_status"] = "ok"
                        out["b_images"] = imgs_b
                    else:
                        out["b_status"] = "not_found"
                else:
                    out["b_status"] = "not_found"

            return out
        except Exception as e:
            return {"ok": False, "error": str(e)}

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

            return {"ok": True, "project_dir": str(trainer.project_dir), "resumed_from": str(ckpt_path)}

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
        for pair in outputs[: min(16, len(outputs))]:
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
        title="Style Transfer Web",
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