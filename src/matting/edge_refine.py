from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import cv2
import numpy as np

from .chromakey import ChromaKeyParams, build_chromakey_alpha, fuse_alpha


@dataclass
class RefineParams:
    feather: float = 1.5
    choke: int = 1
    despill: float = 1.0
    open_kernel: int = 0
    use_chromakey: bool = True
    chromakey: ChromaKeyParams = None  # type: ignore[assignment]
    fuse_mode: str = "min"
    clean_threshold: int = 4
    decontaminate: bool = False
    decontam_strength: float = 0.5
    bg_color: tuple[int, int, int] = (0, 255, 0)  # green screen
    decontam_alpha_min: float = 0.05  # avoid div-by-tiny-alpha
    decontam_alpha_max: float = 0.95  # apply only to edge pixels

    def __post_init__(self) -> None:
        if self.chromakey is None:
            self.chromakey = ChromaKeyParams()

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RefineParams":
        ck = d.get("chromakey") or {}
        bg = d.get("bg_color", (0, 255, 0))
        return cls(
            feather=float(d.get("feather", 1.5)),
            choke=int(d.get("choke", 1)),
            despill=float(d.get("despill", 1.0)),
            open_kernel=int(d.get("open_kernel", 0)),
            use_chromakey=bool(d.get("use_chromakey", True)),
            chromakey=ChromaKeyParams(**ck) if ck else ChromaKeyParams(),
            fuse_mode=str(d.get("fuse_mode", "min")),
            clean_threshold=int(d.get("clean_threshold", 4)),
            decontaminate=bool(d.get("decontaminate", False)),
            decontam_strength=float(d.get("decontam_strength", 0.5)),
            bg_color=tuple(bg),  # type: ignore[arg-type]
            decontam_alpha_min=float(d.get("decontam_alpha_min", 0.05)),
            decontam_alpha_max=float(d.get("decontam_alpha_max", 0.95)),
        )


def choke_alpha(pha: np.ndarray, choke: int) -> np.ndarray:
    if choke <= 0:
        return pha
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.erode(pha, k, iterations=int(choke))


def feather_alpha(pha: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return pha
    return cv2.GaussianBlur(pha, (0, 0), sigmaX=float(sigma))


def open_alpha(pha: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 0:
        return pha
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    return cv2.morphologyEx(pha, cv2.MORPH_OPEN, k)


def despill_rgb(rgb: np.ndarray, pha: np.ndarray, strength: float) -> np.ndarray:
    """Clamp green channel: g ← g - strength · max(g - max(r,b), 0).

    Applied uniformly across all alpha values (not just edges) — green spill on
    fully-opaque hair lit by green screen is just as visible as edge spill.
    strength=1.0 fully clamps green ≤ max(r,b) (the natural ceiling for any
    non-green object); strength>1 over-suppresses (push toward neutral).
    """
    if strength <= 0:
        return rgb
    rgb_f = rgb.astype(np.float32)
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    rb_max = np.maximum(r, b)
    g_excess = np.clip(g - rb_max, 0, None)

    g_new = g - float(strength) * g_excess
    out = rgb_f.copy()
    out[..., 1] = np.clip(g_new, 0, 255)
    return out.astype(np.uint8)


def decontaminate_edges(
    rgb: np.ndarray,
    pha: np.ndarray,
    bg_color: tuple[int, int, int] = (0, 255, 0),
    alpha_min: float = 0.05,
    alpha_max: float = 0.95,
    strength: float = 0.5,
) -> np.ndarray:
    """Remove background-color contamination from partially-transparent edges.

    Solves observed = α·fg + (1-α)·bg  for fg. To avoid the magenta artifact
    (over-subtracting more green than actually exists in dark hair pixels):
    1. cap the per-channel subtraction to the channel's own value
    2. blend result with original by `strength` (0=no change, 1=full)
    Default strength=0.5 is a safe middle; pump up for stubborn green, dial
    down if you see pink/magenta tint.
    """
    if strength <= 0:
        return rgb

    a = pha.astype(np.float32) / 255.0
    a3 = a[..., None]
    bg = np.array(bg_color, dtype=np.float32)

    rgb_f = rgb.astype(np.float32)
    contrib = (1.0 - a3) * bg                   # what bg would contribute
    contrib_capped = np.minimum(contrib, rgb_f)  # don't subtract more than actually present
    a_safe = np.clip(a3, alpha_min, 1.0)
    fg_solved = (rgb_f - contrib_capped) / a_safe
    fg_solved = np.clip(fg_solved, 0, 255)

    blended = rgb_f * (1.0 - float(strength)) + fg_solved * float(strength)

    edge_mask = (a > alpha_min) & (a < alpha_max)
    out = rgb_f.copy()
    out[edge_mask] = blended[edge_mask]
    return out.astype(np.uint8)


def premultiply(rgb: np.ndarray, pha: np.ndarray) -> np.ndarray:
    a = (pha.astype(np.float32) / 255.0)[..., None]
    return np.clip(rgb.astype(np.float32) * a, 0, 255).astype(np.uint8)


def refine_frame(
    fgr_rgb: np.ndarray,
    pha: np.ndarray,
    params: RefineParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply chromakey fusion + choke + feather + despill + transparent-area cleanup.

    Returns (fgr_rgb, pha). Pixels with alpha <= clean_threshold get RGB zeroed —
    invisible after compositing, but lets the encoder compress those large flat
    regions efficiently (huge bitrate savings vs. leaving the original green RGB).
    """
    pha_out = pha
    if params.use_chromakey:
        pha_key = build_chromakey_alpha(fgr_rgb, params.chromakey)
        pha_out = fuse_alpha(pha_out, pha_key, mode=params.fuse_mode)

    pha_out = open_alpha(pha_out, params.open_kernel)
    pha_out = choke_alpha(pha_out, params.choke)
    pha_out = feather_alpha(pha_out, params.feather)

    fgr_out = despill_rgb(fgr_rgb, pha_out, params.despill)

    if params.decontaminate:
        fgr_out = decontaminate_edges(
            fgr_out, pha_out,
            bg_color=params.bg_color,
            alpha_min=params.decontam_alpha_min,
            alpha_max=params.decontam_alpha_max,
            strength=params.decontam_strength,
        )

    if params.clean_threshold >= 0:
        mask = pha_out <= int(params.clean_threshold)
        if mask.any():
            fgr_out = fgr_out.copy()
            fgr_out[mask] = 0

    return fgr_out, pha_out


def composite_checkerboard(rgb: np.ndarray, pha: np.ndarray, tile: int = 16) -> np.ndarray:
    """For preview: composite over a gray checkerboard so transparency is visible."""
    h, w = pha.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    pattern = (((xx // tile) + (yy // tile)) % 2).astype(np.uint8)
    bg = np.where(pattern[..., None] == 0, 200, 150).astype(np.uint8)
    bg = np.repeat(bg, 3, axis=2)
    a = (pha.astype(np.float32) / 255.0)[..., None]
    out = rgb.astype(np.float32) * a + bg.astype(np.float32) * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)
