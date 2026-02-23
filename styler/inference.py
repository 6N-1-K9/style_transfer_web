from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torchvision import transforms

from .config import InferenceConfig
from .device import resolve_device
from .models import StyleGenerator
from .utils import ensure_dir, list_images


@torch.inference_mode()
def run_inference_images(cfg: InferenceConfig, output_dir: Path) -> List[Tuple[Path, Path]]:
    device = resolve_device(cfg.device)

    gen = StyleGenerator(
        n_residual_blocks=cfg.residual_blocks,
        use_dropout=cfg.use_dropout,
        dropout_p=cfg.dropout_p,
    ).to(device)
    gen.eval()

    state = torch.load(Path(cfg.model_path).expanduser().resolve(), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        gen.load_state_dict(state["state_dict"], strict=True)
    else:
        gen.load_state_dict(state, strict=True)

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
