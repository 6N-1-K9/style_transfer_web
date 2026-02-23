from pathlib import Path

from .config import InferenceConfig, TrainConfig
from .device import get_cuda_status
from .inference import run_inference_images
from .trainer import CycleGANTrainer
from .utils import ensure_dir, now_ts


def _prompt(msg: str, default: str = "") -> str:
    if default:
        s = input(f"{msg} [{default}]: ").strip()
        return s if s else default
    return input(f"{msg}: ").strip()


def _prompt_int(msg: str, default: int) -> int:
    s = _prompt(msg, str(default))
    try:
        return int(s)
    except Exception:
        return default


def _prompt_float(msg: str, default: float) -> float:
    s = _prompt(msg, str(default))
    try:
        return float(s)
    except Exception:
        return default


def _prompt_bool(msg: str, default: bool) -> bool:
    d = "y" if default else "n"
    s = _prompt(msg + " (y/n)", d).lower()
    return s in ("y", "yes", "1", "true", "t")


def _validate_device(device: str) -> str:
    dev = (device or "cpu").strip().lower()
    if dev == "cuda":
        st = get_cuda_status()
        if not st.available:
            print("\n[WARN] Ты выбрал GPU (cuda), но он недоступен.")
            print("[WARN]", st.message)
            print("[WARN] Переключаюсь на CPU.\n")
            return "cpu"
    return "cpu" if dev != "cuda" else "cuda"


def run_cli() -> None:
    print("\nConsole mode")
    print("1) Train (CycleGAN)")
    print("2) Inference (apply generator)")
    print("0) Back\n")
    mode = _prompt("Choose", "1")

    if mode == "1":
        cfg = TrainConfig()

        cfg.domain_a_dir = _prompt("Domain A folder (objects)", cfg.domain_a_dir)
        cfg.domain_b_dir = _prompt("Domain B folder (styles)", cfg.domain_b_dir)
        cfg.image_size = _prompt_int("Image size", cfg.image_size)
        cfg.batch_size = _prompt_int("Batch size", cfg.batch_size)
        cfg.epochs = _prompt_int("Epochs", cfg.epochs)

        cfg.residual_blocks = _prompt_int("Residual blocks", cfg.residual_blocks)
        cfg.use_dropout = _prompt_bool("Use dropout", cfg.use_dropout)
        cfg.dropout_p = _prompt_float("Dropout p", cfg.dropout_p)

        cfg.lr = _prompt_float("Learning rate", cfg.lr)
        cfg.lambda_cycle = _prompt_float("Lambda cycle", cfg.lambda_cycle)
        cfg.lambda_identity = _prompt_float("Lambda identity", cfg.lambda_identity)

        cfg.lr_decay_start = _prompt_int("LR decay start epoch", cfg.lr_decay_start)
        cfg.lr_decay_end = _prompt_int("LR decay end epoch", cfg.lr_decay_end)
        cfg.final_lr_ratio = _prompt_float("Final LR ratio", cfg.final_lr_ratio)

        cfg.use_replay_buffer = _prompt_bool("Use replay buffer", cfg.use_replay_buffer)
        cfg.replay_buffer_size = _prompt_int("Replay buffer size", cfg.replay_buffer_size)
        cfg.gradient_clip_norm = _prompt_float("Gradient clip norm (0=off)", cfg.gradient_clip_norm)

        cfg.early_stopping = _prompt_bool("Early stopping", cfg.early_stopping)
        cfg.early_stopping_patience = _prompt_int("Early stopping patience", cfg.early_stopping_patience)
        cfg.early_stopping_min_delta = _prompt_float("Early stopping min delta", cfg.early_stopping_min_delta)
        cfg.early_stopping_metric = _prompt(
            "Early stopping metric (G_total/cycle/identity/adv)", cfg.early_stopping_metric
        )

        cfg.device = _validate_device(_prompt("Device (cpu/cuda)", cfg.device))

        run_dir = Path("runs") / f"run_{now_ts()}"
        ensure_dir(run_dir)

        trainer = CycleGANTrainer(cfg, run_dir=run_dir)
        trainer.train(stop_checker=lambda: False)

        print(f"\nDone. Run directory: {run_dir}")
        print("Checkpoints in:", run_dir / "checkpoints")
        return

    if mode == "2":
        cfg = InferenceConfig()
        cfg.model_path = _prompt("Generator .pth path", cfg.model_path)
        cfg.input_path = _prompt("Input image or folder", cfg.input_path)
        cfg.output_dir = _prompt("Output folder (empty=auto)", cfg.output_dir)

        cfg.image_size = _prompt_int("Image size", cfg.image_size)
        cfg.residual_blocks = _prompt_int("Residual blocks", cfg.residual_blocks)
        cfg.use_dropout = _prompt_bool("Use dropout (usually no)", cfg.use_dropout)
        cfg.dropout_p = _prompt_float("Dropout p", cfg.dropout_p)

        cfg.device = _validate_device(_prompt("Device (cpu/cuda)", cfg.device))

        out_dir = Path(cfg.output_dir).expanduser().resolve() if cfg.output_dir else (Path("runs") / f"infer_{now_ts()}")
        ensure_dir(out_dir)

        outputs = run_inference_images(cfg, output_dir=out_dir)
        print(f"\nSaved {len(outputs)} images to: {out_dir}")
        return

    print("Back.")
