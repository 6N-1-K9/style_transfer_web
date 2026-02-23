import json
import random
import re
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import TrainConfig
from .datasets import UnpairedImageDataset
from .device import resolve_device
from .models import Discriminator, StyleGenerator
from .utils import ensure_dir


class ReplayBuffer:
    def __init__(self, max_size: int = 50) -> None:
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, batch: torch.Tensor) -> torch.Tensor:
        out = []
        for item in batch:
            item = item.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(item)
                out.append(item)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.max_size - 1)
                    old = self.data[idx].clone()
                    self.data[idx] = item
                    out.append(old)
                else:
                    out.append(item)
        return torch.cat(out, dim=0)


class LinearDecayLR:
    def __init__(self, base_lr: float, start_epoch: int, end_epoch: int, final_ratio: float) -> None:
        self.base_lr = base_lr
        self.start = max(0, start_epoch)
        self.end = max(self.start + 1, end_epoch)
        self.final_ratio = max(0.0, min(final_ratio, 1.0))

    def lr_for_epoch(self, epoch: int) -> float:
        if epoch < self.start:
            return self.base_lr
        if epoch >= self.end:
            return self.base_lr * self.final_ratio
        t = (epoch - self.start) / float(self.end - self.start)
        target = self.base_lr * (1.0 - t) + (self.base_lr * self.final_ratio) * t
        return target


class CycleGANTrainer:
    """
    project_dir structure:
      project_dir/
        checkpoints/
        models/
        stats/
        samples/
        train_config.json
    """

    MODEL_A2B_RE = re.compile(r"^G_A2B_epoch_(\d{4})\.pth$")
    MODEL_B2A_RE = re.compile(r"^G_B2A_epoch_(\d{4})\.pth$")

    def __init__(self, cfg: TrainConfig, project_dir: Path) -> None:
        self.cfg = cfg
        self.project_dir = project_dir
        ensure_dir(self.project_dir)

        self.checkpoints_dir = self.project_dir / "checkpoints"
        self.models_dir = self.project_dir / "models"
        self.stats_dir = self.project_dir / "stats"
        self.samples_dir = self.project_dir / "samples"

        ensure_dir(self.checkpoints_dir)
        ensure_dir(self.models_dir)
        ensure_dir(self.stats_dir)
        ensure_dir(self.samples_dir)

        self._stats_losses_csv = self.stats_dir / "losses.csv"
        self._stats_lr_csv = self.stats_dir / "lr.csv"
        self._stats_logs_txt = self.stats_dir / "logs.txt"

        self.stats_to_save = set([s.strip() for s in (cfg.stats_to_save or []) if s.strip()])

        self.device = resolve_device(cfg.device)

        # Models
        self.G_A2B = StyleGenerator(cfg.residual_blocks, cfg.use_dropout, cfg.dropout_p).to(self.device)
        self.G_B2A = StyleGenerator(cfg.residual_blocks, cfg.use_dropout, cfg.dropout_p).to(self.device)
        self.D_A = Discriminator().to(self.device)
        self.D_B = Discriminator().to(self.device)

        # Losses
        self.adv = nn.MSELoss()
        self.l1 = nn.L1Loss()

        # Optimizers
        self.opt_G = optim.Adam(
            list(self.G_A2B.parameters()) + list(self.G_B2A.parameters()),
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
        )
        self.opt_D_A = optim.Adam(self.D_A.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
        self.opt_D_B = optim.Adam(self.D_B.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))

        self.lr_sched = LinearDecayLR(cfg.lr, cfg.lr_decay_start, cfg.lr_decay_end, cfg.final_lr_ratio)

        self.fake_A_buf = ReplayBuffer(cfg.replay_buffer_size) if cfg.use_replay_buffer else None
        self.fake_B_buf = ReplayBuffer(cfg.replay_buffer_size) if cfg.use_replay_buffer else None

        # Data
        ds = UnpairedImageDataset(
            domain_a_dir=Path(cfg.domain_a_dir).expanduser().resolve(),
            domain_b_dir=Path(cfg.domain_b_dir).expanduser().resolve(),
            image_size=cfg.image_size,
            recursive=cfg.recursive_search,
            max_images_a=cfg.max_images_a,
            max_images_b=cfg.max_images_b,
        )
        self.loader = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            drop_last=True,
        )

        self._steps_per_epoch = max(1, len(self.loader))
        self._start_epoch = 0  # 0-based
        self._global_step = 0

        self._status: Dict[str, object] = {
            "epoch": 0,
            "step": 0,
            "total_steps": cfg.epochs * self._steps_per_epoch,
            "lr": cfg.lr,
            "logs": [],
            "last_losses": {},
            "early_stopped": False,
            "saved_last": None,
            "project_dir": str(self.project_dir),
            "resumed_from": None,
        }

        self._best_metric: Optional[float] = None
        self._bad_epochs: int = 0

        (self.project_dir / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
        self._init_stats_files()

    # ---------------- Resume helpers ----------------
    @staticmethod
    def _infer_project_dir_from_checkpoint(checkpoint_path: Path) -> Path:
        p = checkpoint_path.expanduser().resolve()
        if p.parent.name != "checkpoints":
            raise ValueError("Checkpoint path must be inside <project_dir>/checkpoints/")
        return p.parent.parent

    @classmethod
    def resume_from_checkpoint(cls, checkpoint_path: Path) -> "CycleGANTrainer":
        ckpt_path = checkpoint_path.expanduser().resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(str(ckpt_path))

        project_dir = cls._infer_project_dir_from_checkpoint(ckpt_path)

        raw = torch.load(ckpt_path, map_location="cpu")
        if not isinstance(raw, dict) or "config" not in raw or "epoch" not in raw:
            raise ValueError("Invalid checkpoint format (missing 'config' or 'epoch')")

        cfg = TrainConfig(**raw["config"])
        trainer = cls(cfg, project_dir=project_dir)

        trainer.G_A2B.load_state_dict(raw["G_A2B"], strict=True)
        trainer.G_B2A.load_state_dict(raw["G_B2A"], strict=True)
        trainer.D_A.load_state_dict(raw["D_A"], strict=True)
        trainer.D_B.load_state_dict(raw["D_B"], strict=True)

        raw2 = torch.load(ckpt_path, map_location=trainer.device)
        trainer.opt_G.load_state_dict(raw2["opt_G"])
        trainer.opt_D_A.load_state_dict(raw2["opt_D_A"])
        trainer.opt_D_B.load_state_dict(raw2["opt_D_B"])

        saved_epoch_1based = int(raw["epoch"])
        trainer._start_epoch = max(0, saved_epoch_1based)
        trainer._global_step = trainer._start_epoch * trainer._steps_per_epoch

        trainer._status["resumed_from"] = str(ckpt_path)
        trainer._status["epoch"] = trainer._start_epoch
        trainer._status["step"] = trainer._global_step
        trainer._status["total_steps"] = trainer.cfg.epochs * trainer._steps_per_epoch

        trainer._log(f"Resuming from checkpoint: {ckpt_path}")
        trainer._log(f"Project dir: {project_dir}")
        trainer._log(f"Start epoch (next): {trainer._start_epoch + 1}/{trainer.cfg.epochs}")

        return trainer

    # ---------------- Stats / logs ----------------
    def _init_stats_files(self) -> None:
        if "losses_csv" in self.stats_to_save and not self._stats_losses_csv.exists():
            self._stats_losses_csv.write_text("epoch,G_total,D_total,cycle,identity,adv\n", encoding="utf-8")
        if "lr_csv" in self.stats_to_save and not self._stats_lr_csv.exists():
            self._stats_lr_csv.write_text("epoch,lr\n", encoding="utf-8")

    def _append_losses_csv(self, epoch_1based: int, avg: Dict[str, float]) -> None:
        if "losses_csv" not in self.stats_to_save:
            return
        line = (
            f"{epoch_1based},"
            f"{avg.get('G_total',0.0):.6f},{avg.get('D_total',0.0):.6f},"
            f"{avg.get('cycle',0.0):.6f},{avg.get('identity',0.0):.6f},{avg.get('adv',0.0):.6f}\n"
        )
        with self._stats_losses_csv.open("a", encoding="utf-8") as f:
            f.write(line)

    def _append_lr_csv(self, epoch_1based: int, lr: float) -> None:
        if "lr_csv" not in self.stats_to_save:
            return
        with self._stats_lr_csv.open("a", encoding="utf-8") as f:
            f.write(f"{epoch_1based},{lr:.10f}\n")

    def _append_logs_txt(self, msg: str) -> None:
        if "logs_txt" not in self.stats_to_save:
            return
        with self._stats_logs_txt.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def _log(self, msg: str) -> None:
        logs = self._status["logs"]
        assert isinstance(logs, list)
        logs.append(msg)
        if len(logs) > self.cfg.max_log_lines:
            del logs[: len(logs) - self.cfg.max_log_lines]
        self._append_logs_txt(msg)

    def get_status(self) -> Dict[str, object]:
        return dict(self._status)

    def _set_lr(self, lr: float) -> None:
        for pg in self.opt_G.param_groups:
            pg["lr"] = lr
        for pg in self.opt_D_A.param_groups:
            pg["lr"] = lr
        for pg in self.opt_D_B.param_groups:
            pg["lr"] = lr
        self._status["lr"] = lr

    # ---------------- Model saving (models/) ----------------
    def _meta(self) -> Dict[str, object]:
        return {
            "residual_blocks": self.cfg.residual_blocks,
            "use_dropout": self.cfg.use_dropout,
            "dropout_p": self.cfg.dropout_p,
            "image_size": self.cfg.image_size,
        }

    def _parse_model_epochs(self, which: str) -> List[Tuple[int, Path]]:
        out: List[Tuple[int, Path]] = []
        for p in self.models_dir.glob("*.pth"):
            m = self.MODEL_A2B_RE.match(p.name) if which == "A2B" else self.MODEL_B2A_RE.match(p.name)
            if not m:
                continue
            try:
                ep = int(m.group(1))
                out.append((ep, p))
            except Exception:
                continue
        out.sort(key=lambda x: x[0])
        return out

    def _cleanup_keep_last_files(self) -> None:
        """
        Keep only last K *saved model files* (by epoch number) in models/.
        Works for:
          - interval only
          - keep_last only (dense epochs)
          - interval + keep_last (sparse epochs): keeps last K saved epochs
        """
        if not self.cfg.models_keep_last_enabled:
            return
        k = int(self.cfg.models_keep_last_count or 0)
        if k <= 0:
            return

        # A2B cleanup
        a = self._parse_model_epochs("A2B")
        if len(a) > k:
            for ep, path in a[: len(a) - k]:
                try:
                    path.unlink()
                except Exception:
                    pass

        # B2A cleanup only if enabled
        if self.cfg.save_b2a_models:
            b = self._parse_model_epochs("B2A")
            if len(b) > k:
                for ep, path in b[: len(b) - k]:
                    try:
                        path.unlink()
                    except Exception:
                        pass

    def _should_save_models_this_epoch(self, epoch_1based: int) -> bool:
        """
        Priority:
          - if interval enabled: save only on interval (and last epoch always)
          - else if keep_last enabled: save every epoch (because otherwise last K cannot exist)
          - else: don't save
        """
        if self.cfg.models_save_interval_enabled:
            n = int(self.cfg.models_save_interval_epochs or 1)
            if n < 1:
                n = 1
            if epoch_1based >= self.cfg.epochs:
                return True
            return (epoch_1based % n) == 0

        if self.cfg.models_keep_last_enabled:
            return True  # every epoch

        return False

    def _save_models_if_needed(self, epoch_1based: int) -> bool:
        if not self._should_save_models_this_epoch(epoch_1based):
            return False

        meta = self._meta()

        # Save A2B
        torch.save(
            {"state_dict": self.G_A2B.state_dict(), "meta": meta},
            self.models_dir / f"G_A2B_epoch_{epoch_1based:04d}.pth",
        )

        # Save B2A only if enabled
        if self.cfg.save_b2a_models:
            torch.save(
                {"state_dict": self.G_B2A.state_dict(), "meta": meta},
                self.models_dir / f"G_B2A_epoch_{epoch_1based:04d}.pth",
            )

        # After saving, apply keep-last policy if enabled (keeps last K saved files)
        self._cleanup_keep_last_files()
        return True

    # ---------------- Checkpoint saving (checkpoints/) ----------------
    def _delete_old_checkpoints_except(self, keep_path: Path) -> None:
        try:
            for p in self.checkpoints_dir.glob("epoch_*.pth"):
                if p.resolve() != keep_path.resolve():
                    try:
                        p.unlink()
                    except Exception:
                        pass
        except Exception:
            pass

    def _save_checkpoint(self, epoch_1based: int) -> Optional[Path]:
        if not self.cfg.save_checkpoints:
            return None

        ckpt = {
            "epoch": epoch_1based,
            "config": asdict(self.cfg),
            "G_A2B": self.G_A2B.state_dict(),
            "G_B2A": self.G_B2A.state_dict(),
            "D_A": self.D_A.state_dict(),
            "D_B": self.D_B.state_dict(),
            "opt_G": self.opt_G.state_dict(),
            "opt_D_A": self.opt_D_A.state_dict(),
            "opt_D_B": self.opt_D_B.state_dict(),
            "meta": self._meta(),
        }
        path = self.checkpoints_dir / f"epoch_{epoch_1based:04d}.pth"
        torch.save(ckpt, path)

        if self.cfg.keep_only_latest_checkpoint:
            self._delete_old_checkpoints_except(path)

        self._status["saved_last"] = str(path)
        return path

    def _should_save_checkpoint_epoch(self, epoch_1based: int) -> bool:
        if not self.cfg.save_checkpoints:
            return False
        if epoch_1based >= self.cfg.epochs:
            return True
        n = int(self.cfg.checkpoint_interval_epochs or 1)
        if n < 1:
            n = 1
        return (epoch_1based % n) == 0

    def _maybe_clip(self, params) -> None:
        if self.cfg.gradient_clip_norm and self.cfg.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=self.cfg.gradient_clip_norm)

    # ---------------- Training loop ----------------
    def train(self, stop_checker: Optional[Callable[[], bool]] = None) -> None:
        stop_checker = stop_checker or (lambda: False)

        def ones_like(pred: torch.Tensor) -> torch.Tensor:
            return torch.ones_like(pred, device=pred.device)

        def zeros_like(pred: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(pred, device=pred.device)

        self._log(f"Device: {self.device}")
        self._log(f"Steps/epoch: {self._steps_per_epoch}")

        self._log(
            f"Checkpoint saving: enabled={self.cfg.save_checkpoints}, "
            f"interval={self.cfg.checkpoint_interval_epochs}, "
            f"keep_only_latest={self.cfg.keep_only_latest_checkpoint}"
        )
        self._log(
            f"Model saving: interval_enabled={self.cfg.models_save_interval_enabled}, "
            f"interval={self.cfg.models_save_interval_epochs}, "
            f"keep_last_enabled={self.cfg.models_keep_last_enabled}, "
            f"keep_last_count={self.cfg.models_keep_last_count}"
        )
        self._log(f"Save B2A models: {self.cfg.save_b2a_models}")

        if self._start_epoch > 0:
            self._log(f"Resume: starting from epoch {self._start_epoch + 1}")

        total_steps = self._global_step
        self._status["total_steps"] = self.cfg.epochs * self._steps_per_epoch
        self._status["step"] = total_steps
        self._status["epoch"] = self._start_epoch

        for epoch in range(self._start_epoch, self.cfg.epochs):
            if stop_checker():
                self._log("Training stopped by user.")
                break

            self._status["epoch"] = epoch + 1

            lr = self.lr_sched.lr_for_epoch(epoch)
            self._set_lr(lr)

            sum_losses = {"G_total": 0.0, "D_total": 0.0, "cycle": 0.0, "identity": 0.0, "adv": 0.0}
            n_batches = 0

            for batch_i, (real_A, real_B, _, _) in enumerate(self.loader):
                if stop_checker():
                    self._log("Training stopped by user.")
                    break

                n_batches += 1
                total_steps += 1
                self._status["step"] = total_steps

                real_A = real_A.to(self.device)
                real_B = real_B.to(self.device)

                # --- Generators ---
                self.opt_G.zero_grad(set_to_none=True)

                idt_A = self.G_B2A(real_A)
                idt_B = self.G_A2B(real_B)
                loss_idt = (self.l1(idt_A, real_A) + self.l1(idt_B, real_B)) * self.cfg.lambda_identity

                fake_B = self.G_A2B(real_A)
                pred_fake_B = self.D_B(fake_B)
                loss_gan_A2B = self.adv(pred_fake_B, ones_like(pred_fake_B))

                fake_A = self.G_B2A(real_B)
                pred_fake_A = self.D_A(fake_A)
                loss_gan_B2A = self.adv(pred_fake_A, ones_like(pred_fake_A))

                loss_adv = loss_gan_A2B + loss_gan_B2A

                rec_A = self.G_B2A(fake_B)
                rec_B = self.G_A2B(fake_A)
                loss_cycle = (self.l1(rec_A, real_A) + self.l1(rec_B, real_B)) * self.cfg.lambda_cycle

                loss_G = loss_adv + loss_cycle + loss_idt
                loss_G.backward()
                self._maybe_clip(list(self.G_A2B.parameters()) + list(self.G_B2A.parameters()))
                self.opt_G.step()

                # --- Discriminator A ---
                self.opt_D_A.zero_grad(set_to_none=True)
                pred_real_A = self.D_A(real_A)
                loss_D_real_A = self.adv(pred_real_A, ones_like(pred_real_A))

                fake_A_det = fake_A.detach()
                if self.fake_A_buf is not None:
                    fake_A_det = self.fake_A_buf.push_and_pop(fake_A_det)
                pred_fake_A2 = self.D_A(fake_A_det)
                loss_D_fake_A = self.adv(pred_fake_A2, zeros_like(pred_fake_A2))

                loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5
                loss_D_A.backward()
                self._maybe_clip(self.D_A.parameters())
                self.opt_D_A.step()

                # --- Discriminator B ---
                self.opt_D_B.zero_grad(set_to_none=True)
                pred_real_B = self.D_B(real_B)
                loss_D_real_B = self.adv(pred_real_B, ones_like(pred_real_B))

                fake_B_det = fake_B.detach()
                if self.fake_B_buf is not None:
                    fake_B_det = self.fake_B_buf.push_and_pop(fake_B_det)
                pred_fake_B2 = self.D_B(fake_B_det)
                loss_D_fake_B = self.adv(pred_fake_B2, zeros_like(pred_fake_B2))

                loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5
                loss_D_B.backward()
                self._maybe_clip(self.D_B.parameters())
                self.opt_D_B.step()

                loss_D = loss_D_A + loss_D_B

                sum_losses["G_total"] += float(loss_G.item())
                sum_losses["D_total"] += float(loss_D.item())
                sum_losses["cycle"] += float(loss_cycle.item())
                sum_losses["identity"] += float(loss_idt.item())
                sum_losses["adv"] += float(loss_adv.item())

                if total_steps % 10 == 0:
                    self._status["last_losses"] = {
                        "G_total": float(loss_G.item()),
                        "D_total": float(loss_D.item()),
                        "cycle": float(loss_cycle.item()),
                        "identity": float(loss_idt.item()),
                        "adv": float(loss_adv.item()),
                    }

                if total_steps % 50 == 0:
                    self._log(
                        f"Epoch {epoch+1}/{self.cfg.epochs} | "
                        f"Step {batch_i+1}/{self._steps_per_epoch} | "
                        f"LR {lr:.6f} | "
                        f"G {loss_G.item():.4f} D {loss_D.item():.4f} "
                        f"(cycle {loss_cycle.item():.4f}, id {loss_idt.item():.4f})"
                    )

            if n_batches > 0:
                avg = {k: v / n_batches for k, v in sum_losses.items()}
            else:
                avg = sum_losses

            self._status["last_losses"] = avg
            self._append_losses_csv(epoch + 1, avg)
            self._append_lr_csv(epoch + 1, lr)

            self._log(
                "Epoch %d done. Avg losses: %s"
                % (epoch + 1, ", ".join([f"{k}={v:.4f}" for k, v in avg.items()]))
            )

            epoch_1based = epoch + 1

            # Model saving: interval, keep-last, or both (keep-last keeps last K saved files)
            saved_models = self._save_models_if_needed(epoch_1based)
            if saved_models:
                self._log(f"Saved model weights to models/ for epoch {epoch_1based}")
            else:
                self._log("Model saving skipped this epoch (per settings).")

            # Checkpoint saving
            if self._should_save_checkpoint_epoch(epoch_1based):
                ckpt_path = self._save_checkpoint(epoch_1based)
                if ckpt_path is not None:
                    self._log(f"Saved checkpoint: {ckpt_path}")
                else:
                    self._log("Checkpoint saving disabled.")

            # Early stopping
            if self.cfg.early_stopping:
                metric_name = self.cfg.early_stopping_metric
                metric_val = float(avg.get(metric_name, avg.get("G_total", 0.0)))

                improved = False
                if self._best_metric is None:
                    improved = True
                else:
                    improved = (self._best_metric - metric_val) > self.cfg.early_stopping_min_delta

                if improved:
                    self._best_metric = metric_val
                    self._bad_epochs = 0
                    self._log(f"EarlyStopping: improved {metric_name} -> {metric_val:.4f}")
                else:
                    self._bad_epochs += 1
                    self._log(f"EarlyStopping: no improvement ({self._bad_epochs}/{self.cfg.early_stopping_patience})")

                if self._bad_epochs >= self.cfg.early_stopping_patience:
                    self._status["early_stopped"] = True
                    self._log("EarlyStopping: triggered. Stopping training.")
                    break

        self._log("Training finished.")