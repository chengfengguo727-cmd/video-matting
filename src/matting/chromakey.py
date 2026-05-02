from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ChromaKeyParams:
    h_min: int = 35
    h_max: int = 85
    s_min: int = 60
    v_min: int = 60
    softness: float = 1.5


def build_chromakey_alpha(rgb: np.ndarray, params: ChromaKeyParams) -> np.ndarray:
    """Return uint8 [H,W] alpha — 255 for non-green (foreground), 0 for green."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([params.h_min, params.s_min, params.v_min], dtype=np.uint8)
    upper = np.array([params.h_max, 255, 255], dtype=np.uint8)
    green_mask = cv2.inRange(hsv, lower, upper)  # 255 where green
    alpha = 255 - green_mask
    if params.softness > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=params.softness)
    return alpha


def fuse_alpha(pha_rvm: np.ndarray, pha_key: np.ndarray, mode: str = "min") -> np.ndarray:
    """Combine RVM alpha and chromakey alpha.

    'min'  — pixel is foreground only if BOTH agree (cleans RVM stray green)
    'max'  — pixel is foreground if EITHER agrees (recovers fine hair)
    'mean' — soft compromise
    """
    if pha_rvm.shape != pha_key.shape:
        pha_key = cv2.resize(pha_key, (pha_rvm.shape[1], pha_rvm.shape[0]))
    if mode == "min":
        return np.minimum(pha_rvm, pha_key)
    if mode == "max":
        return np.maximum(pha_rvm, pha_key)
    if mode == "mean":
        return ((pha_rvm.astype(np.uint16) + pha_key.astype(np.uint16)) // 2).astype(np.uint8)
    raise ValueError(f"Unknown fuse mode: {mode}")
