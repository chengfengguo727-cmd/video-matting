from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch

from ..io.frames import iter_video_frames, probe_size


REPO = "PeterL1n/RobustVideoMatting"
DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "models" / "rvm_mobilenetv3.pth"


def auto_downsample_ratio(width: int, height: int) -> float:
    long_edge = max(width, height)
    if long_edge >= 3000:
        return 0.125
    if long_edge >= 1500:
        return 0.25
    if long_edge >= 900:
        return 0.375
    return 0.5


def pick_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_rvm_model(
    variant: str = "mobilenetv3",
    weights_path: Optional[str | Path] = None,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    if device is None:
        device = pick_device()

    model = torch.hub.load(REPO, variant, trust_repo=True)

    chosen = Path(weights_path) if weights_path else DEFAULT_WEIGHTS
    if chosen.exists():
        state = torch.load(str(chosen), map_location="cpu")
        model.load_state_dict(state)

    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def iter_matte_frames(
    model: torch.nn.Module,
    video_path: str | Path,
    device: Optional[torch.device] = None,
    downsample_ratio: Optional[float] = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if device is None:
        device = next(model.parameters()).device

    if downsample_ratio is None:
        w, h = probe_size(video_path)
        downsample_ratio = auto_downsample_ratio(w, h)

    rec = [None, None, None, None]
    for frame_rgb in iter_video_frames(video_path):
        src = (
            torch.from_numpy(frame_rgb)
            .to(device, non_blocking=True)
            .float()
            .div_(255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        fgr, pha, *rec = model(src, *rec, downsample_ratio)
        fgr_np = (
            fgr[0].clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
        )
        pha_np = pha[0, 0].clamp(0, 1).mul(255).byte().cpu().numpy()
        yield fgr_np, pha_np
