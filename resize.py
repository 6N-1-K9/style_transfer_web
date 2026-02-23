from pathlib import Path
from PIL import Image

# ====== НАСТРОЙКИ ======
INPUT_DIR = Path(r"/home/n1k/Загрузки/styles")     # папка с исходными изображениями
OUTPUT_DIR = Path(r"/home/n1k/Загрузки/styles_resized")   # куда сохранять (структура будет такая же)
SIZE = (256, 256)

# "fit_crop" = без искажений (resize по меньшей стороне + center crop)
# "stretch"  = просто растянуть до 256x256 (может исказить)
MODE = "fit_crop"

# Какие расширения считать картинками
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# =======================

def resize_fit_crop(img: Image.Image, size=(256, 256)) -> Image.Image:
    """Resize сохраняя пропорции + центр-кроп до нужного размера."""
    img = img.convert("RGB")
    target_w, target_h = size
    w, h = img.size

    scale = max(target_w / w, target_h / h)  # чтобы покрыть целевую область
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))

def resize_stretch(img: Image.Image, size=(256, 256)) -> Image.Image:
    """Просто растянуть до размера."""
    img = img.convert("RGB")
    return img.resize(size, Image.Resampling.LANCZOS)

def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = [p for p in INPUT_DIR.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    print(f"Найдено изображений: {len(files)}")

    for src_path in files:
        rel = src_path.relative_to(INPUT_DIR)
        dst_path = OUTPUT_DIR / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(src_path) as img:
                if MODE == "stretch":
                    out = resize_stretch(img, SIZE)
                else:
                    out = resize_fit_crop(img, SIZE)

                # сохраняем с тем же расширением; для JPEG чуть повысим качество
                if dst_path.suffix.lower() in {".jpg", ".jpeg"}:
                    out.save(dst_path, quality=95, optimize=True)
                else:
                    out.save(dst_path)

        except Exception as e:
            print(f"[WARN] Не смог обработать: {src_path} -> {e}")

    print("Готово.")

if __name__ == "__main__":
    main()
