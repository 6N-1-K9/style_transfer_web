from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import list_images


class UnpairedImageDataset(Dataset):
    def __init__(
        self,
        domain_a_dir: Path,
        domain_b_dir: Path,
        image_size: int = 256,
        recursive: bool = True,
        max_images_a: Optional[int] = None,
        max_images_b: Optional[int] = None,
    ) -> None:
        self.domain_a = list_images(domain_a_dir, recursive=recursive)
        self.domain_b = list_images(domain_b_dir, recursive=recursive)

        if max_images_a is not None:
            self.domain_a = self.domain_a[: max_images_a]
        if max_images_b is not None:
            self.domain_b = self.domain_b[: max_images_b]

        if len(self.domain_a) == 0 or len(self.domain_b) == 0:
            raise ValueError("Domain A or B has no images")

        self.t = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self) -> int:
        return max(len(self.domain_a), len(self.domain_b))

    def _load(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self.t(img)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        a_path = self.domain_a[idx % len(self.domain_a)]
        b_path = self.domain_b[idx % len(self.domain_b)]
        return self._load(a_path), self._load(b_path), str(a_path), str(b_path)
