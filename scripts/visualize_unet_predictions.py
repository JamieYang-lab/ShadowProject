from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from datasets.unet_shadow_dataset import UNetShadowDataset
from models.unet_shadow_refiner import UNetShadowRefiner


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "visualizations" / "unet_shadow_refiner" / "predictions"


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def _to_image(array: np.ndarray) -> Image.Image:
    array = np.asarray(array)
    if array.ndim == 2:
        return Image.fromarray(np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)).convert("RGB")
    array = np.transpose(array, (1, 2, 0)) if array.shape[0] in {1, 3} else array
    return Image.fromarray(np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)).convert("RGB")


def _load_optional_mask(path_value, size: tuple[int, int]) -> np.ndarray:
    if isinstance(path_value, (list, tuple)):
        path_value = path_value[0] if path_value else None
    if path_value is None:
        return np.zeros((size[1], size[0]), dtype=np.float32)
    path = Path(path_value)
    if not path.exists():
        return np.zeros((size[1], size[0]), dtype=np.float32)
    with Image.open(path) as image:
        return (np.asarray(image.convert("L").resize(size, Image.Resampling.NEAREST), dtype=np.float32) > 0).astype(np.float32)


def _add_title(panel: Image.Image, title: str) -> Image.Image:
    title_h = 24
    out = Image.new("RGB", (panel.width, panel.height + title_h), (18, 20, 24))
    out.paste(panel, (0, title_h))
    draw = ImageDraw.Draw(out)
    draw.text((8, 6), title, fill=(235, 235, 235), font=ImageFont.load_default())
    return out


def save_prediction_panel(sample: dict, logits: torch.Tensor, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model_input = sample["input"].detach().cpu().numpy()
    target = sample["shadow_mask"].detach().cpu().numpy()[0]
    pred = torch.sigmoid(logits.detach().cpu())[0].numpy()
    error = np.abs(pred - target)
    receiver = _load_optional_mask(sample.get("receiver_mask_path"), (target.shape[1], target.shape[0]))

    panels = [
        ("composite", _to_image(model_input[0:3])),
        ("object", _to_image(model_input[3])),
        ("sg prior", _to_image(model_input[7])),
        ("receiver", _to_image(receiver)),
        ("gt shadow", _to_image(target)),
        ("prediction", _to_image(pred)),
        ("abs error", _to_image(error)),
    ]
    titled = [_add_title(panel, title) for title, panel in panels]
    gap = 8
    width = sum(panel.width for panel in titled) + gap * (len(titled) - 1)
    height = max(panel.height for panel in titled)
    board = Image.new("RGB", (width, height), (18, 20, 24))
    x = 0
    for panel in titled:
        board.paste(panel, (x, 0))
        x += panel.width + gap
    board.save(output)
    return output


def load_checkpoint_model(checkpoint_path: str | Path, device: torch.device) -> UNetShadowRefiner:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    input_channels = int(checkpoint.get("input_channels", 18)) if isinstance(checkpoint, dict) else 18
    model = UNetShadowRefiner(input_channels=input_channels)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def visualize_predictions(
    config_path: str | Path,
    checkpoint_path: str | Path,
    max_samples: int = 10,
    size: int = 512,
    device_name: str = "cuda",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    device = resolve_device(device_name)
    dataset = UNetShadowDataset(config_path, size=size, strict=False, limit=max_samples)
    model = load_checkpoint_model(checkpoint_path, device)
    output_root = Path(output_dir)
    written: list[Path] = []
    with torch.no_grad():
        for sample in dataset:
            inputs = sample["input"].unsqueeze(0).to(device)
            logits = model(inputs)[0]
            path = output_root / f"{sample['sample_id']}_prediction.png"
            written.append(save_prediction_panel(sample, logits, path))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize U-Net shadow mask predictions.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "dataset_config.yaml"))
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "unet_shadow_refiner" / "best.pt"))
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    written = visualize_predictions(args.config, args.checkpoint, args.max_samples, args.size, args.device)
    print(f"Wrote {len(written)} U-Net prediction visualizations.")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
