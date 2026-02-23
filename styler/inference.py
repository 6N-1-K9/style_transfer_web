from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torchvision import transforms

from .config import InferenceConfig
from .device import resolve_device
from .models import StyleGenerator
from .utils import ensure_dir, list_images


def load_generator_payload(model_path: Path) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """
    Load a generator model and return (meta, state_dict).

    Supported formats:
      1) legacy: plain state_dict
      2) current: {"state_dict": ..., "meta": {...}}
    """
    p = Path(model_path).expanduser().resolve()
    raw = torch.load(p, map_location="cpu")

    if isinstance(raw, dict) and "state_dict" in raw:
        meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
        sd = raw["state_dict"]
        if not isinstance(sd, dict):
            raise ValueError("Invalid model format: state_dict is not a dict")
        return meta, sd

    if isinstance(raw, dict):
        # Legacy: file is just a state_dict
        return {}, raw

    raise ValueError("Invalid model format (expected a state_dict or {state_dict, meta}).")


def infer_arch_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Best-effort inference of architecture params from a legacy state_dict.

    We infer:
      - residual_blocks: count of unique model.<idx>.block.* groups
      - use_dropout: heuristic based on presence of ".block.5.weight"
    """
    import re

    idxs = set()
    has_dropout = False

    for k in state_dict.keys():
        m = re.match(r"^model\.(\d+)\.block\.", k)
        if m:
            idxs.add(int(m.group(1)))
        if ".block.5.weight" in k:
            has_dropout = True

    out: Dict[str, Any] = {}
    if idxs:
        out["residual_blocks"] = len(idxs)
    out["use_dropout"] = has_dropout
    return out


@torch.inference_mode()
def run_inference_images(cfg: InferenceConfig, output_dir: Path) -> List[Tuple[Path, Path]]:
    device = resolve_device(cfg.device)

    # IMPORTANT:
    # Architecture-critical parameters MUST match the saved model.
    # UI-provided residual_blocks / dropout are ignored if model meta exists.
    model_path = Path(cfg.model_path).expanduser().resolve()
    meta, state_dict = load_generator_payload(model_path)

    if not meta:
        meta = infer_arch_from_state_dict(state_dict)

    n_resblocks = int(meta.get("residual_blocks", cfg.residual_blocks))
    use_dropout = bool(meta.get("use_dropout", cfg.use_dropout))
    dropout_p = float(meta.get("dropout_p", cfg.dropout_p))

    gen = StyleGenerator(
        n_residual_blocks=n_resblocks,
        use_dropout=use_dropout,
        dropout_p=dropout_p,
    ).to(device)
    gen.eval()
    gen.load_state_dict(state_dict, strict=True)

    t = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    def inv_norm(x: torch.Tensor) -> torch.Tensor:
        return (x * 0.5) + 0.5

    input_path = Path(cfg.input_path).expanduser().resolve()
    if input_path.is_dir():
        inputs = list_images(input_path, recursive=True)
    else:
        inputs = [input_path]

    # IMPORTANT: save directly into output_dir, no extra subfolders
    ensure_dir(output_dir)

    outputs: List[Tuple[Path, Path]] = []
    for p in inputs:
        img = Image.open(p).convert("RGB")
        x = t(img).unsqueeze(0).to(device)
        y = gen(x).cpu().squeeze(0)
        y = inv_norm(y).clamp(0, 1)

        out_img = transforms.ToPILImage()(y)

        # flat structure: files only
        out_path = output_dir / f"{p.stem}_styled.png"
        out_img.save(out_path)
        outputs.append((p, out_path))

    return outputs