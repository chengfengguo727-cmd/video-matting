from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Iterator

import av
import cv2
import numpy as np


def probe_fps(video_path: str | Path) -> float:
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.base_rate or Fraction(30, 1)
        return float(rate)
    finally:
        container.close()


def probe_size(video_path: str | Path) -> tuple[int, int]:
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        return int(stream.width), int(stream.height)
    finally:
        container.close()


def iter_video_frames(video_path: str | Path) -> Iterator[np.ndarray]:
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            yield frame.to_ndarray(format="rgb24")
    finally:
        container.close()


def write_png(path: str | Path, image: np.ndarray) -> None:
    if image.ndim == 3 and image.shape[2] == 3:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        bgra = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        bgr = bgra
    else:
        bgr = image
    ok = cv2.imwrite(str(path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise IOError(f"Failed to write PNG: {path}")


def read_png(path: str | Path, with_alpha: bool = False) -> np.ndarray:
    flag = cv2.IMREAD_UNCHANGED if with_alpha else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise IOError(f"Failed to read PNG: {path}")
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return img


def list_png_seq(directory: str | Path) -> list[Path]:
    p = Path(directory)
    return sorted(p.glob("*.png"))


def read_png_seq_iter(directory: str | Path, with_alpha: bool = False) -> Iterator[np.ndarray]:
    for f in list_png_seq(directory):
        yield read_png(f, with_alpha=with_alpha)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def frame_filename(index: int) -> str:
    return f"frame_{index:06d}.png"
