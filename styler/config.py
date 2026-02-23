from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainConfig:
    # Datasets
    domain_a_dir: str = ""
    domain_b_dir: str = ""

    image_size: int = 256
    batch_size: int = 1
    epochs: int = 50

    max_images_a: Optional[int] = None
    max_images_b: Optional[int] = None
    recursive_search: bool = True

    # Model
    residual_blocks: int = 9
    use_dropout: bool = False
    dropout_p: float = 0.5

    # Optimization
    lr: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999

    lambda_cycle: float = 10.0
    lambda_identity: float = 5.0

    # LR decay
    lr_decay_start: int = 25
    lr_decay_end: int = 50
    final_lr_ratio: float = 0.05

    # Replay buffer
    use_replay_buffer: bool = True
    replay_buffer_size: int = 50

    # Misc
    num_workers: int = 0
    max_log_lines: int = 400
    gradient_clip_norm: float = 0.0

    # Early stopping
    early_stopping: bool = False
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    early_stopping_metric: str = "G_total"

    # Device
    device: str = "cpu"  # "cpu" | "cuda"

    # Project folder (chosen in Web UI)
    project_base_dir: str = ""
    project_name: str = ""

    # Statistics saving selection
    # allowed: "losses_csv", "lr_csv", "logs_txt"
    stats_to_save: List[str] = field(default_factory=lambda: ["losses_csv", "lr_csv"])

    # Save B2A generator weights separately (disk saver)
    save_b2a_models: bool = True

    # Checkpoint saving controls
    save_checkpoints: bool = True
    checkpoint_interval_epochs: int = 1
    keep_only_latest_checkpoint: bool = False

    # NEW: model (weights) saving controls (models/ folder)
    models_save_interval_enabled: bool = True
    models_save_interval_epochs: int = 1

    models_keep_last_enabled: bool = False
    models_keep_last_count: int = 5


@dataclass
class InferenceConfig:
    model_path: str = ""
    input_path: str = ""

    # REQUIRED in Web UI
    output_dir: str = ""

    image_size: int = 256
    residual_blocks: int = 9
    use_dropout: bool = False
    dropout_p: float = 0.0

    device: str = "cpu"
    direction: str = "A2B"  # reserved