from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional


VP9 = "vp9"
VP8 = "vp8"


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    raise RuntimeError(
        "ffmpeg not found on PATH. Run scripts\\install_ffmpeg.ps1 or install manually."
    )


def _video_codec_args(codec: str, crf: int) -> list[str]:
    """Return libvpx-vp9 or libvpx (VP8) args, both encoding alpha into a yuva420p WebM stream.

    Notes on flags:
    - `-auto-alt-ref 0` is REQUIRED for alpha in both VP8 and VP9 libvpx encoders.
    - `-row-mt 1` and `-tile-columns N` are intentionally OMITTED for VP9: in some
      FFmpeg builds (observed on Gyan 8.x) they cause libvpx-vp9 to silently drop
      the alpha plane while keeping the `alpha_mode=1` tag, producing an opaque file.
    """
    if codec == VP9:
        return [
            "-c:v", "libvpx-vp9",
            "-pix_fmt", "yuva420p",
            "-metadata:s:v:0", "alpha_mode=1",
            "-auto-alt-ref", "0",
            "-b:v", "0",
            "-crf", str(int(crf)),
        ]
    if codec == VP8:
        return [
            "-c:v", "libvpx",
            "-pix_fmt", "yuva420p",
            "-metadata:s:v:0", "alpha_mode=1",
            "-auto-alt-ref", "0",
            "-b:v", "0",
            "-crf", str(int(crf)),
            "-quality", "good",
            "-cpu-used", "1",
        ]
    raise ValueError(f"Unknown codec: {codec!r} (expected 'vp9' or 'vp8')")


def encode_webm_from_rgba_pngs(
    rgba_dir: str | Path,
    fps: float,
    out_path: str | Path,
    crf: int = 22,
    threads: int = 8,
    pattern: str = "frame_%06d.png",
    audio_source: Optional[str | Path] = None,
    audio_bitrate: str = "128k",
    codec: str = VP9,
) -> None:
    """Encode a sequence of 4-channel RGBA PNGs into WebM with alpha. No alphamerge."""
    rgba_pattern = str(Path(rgba_dir) / pattern)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        find_ffmpeg(),
        "-y",
        "-framerate", f"{fps:.6f}",
        "-i", rgba_pattern,
    ]
    if audio_source is not None:
        cmd += ["-i", str(audio_source), "-map", "0:v", "-map", "1:a?"]

    cmd += _video_codec_args(codec, crf)
    cmd += ["-threads", str(int(threads))]

    if audio_source is not None:
        cmd += ["-c:a", "libopus", "-b:a", audio_bitrate, "-shortest"]

    cmd += [str(out_path)]

    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed (exit {proc.returncode})")


def encode_webm_from_pngs(
    fgr_dir: str | Path,
    pha_dir: str | Path,
    fps: float,
    out_path: str | Path,
    crf: int = 22,
    threads: int = 8,
    pattern: str = "frame_%06d.png",
    audio_source: Optional[str | Path] = None,
    audio_bitrate: str = "128k",
    codec: str = VP9,
) -> None:
    """Legacy: combine separate fgr/ + pha/ via FFmpeg alphamerge. Some FFmpeg builds
    drop the alpha plane silently with this approach; prefer encode_webm_from_rgba_pngs.
    """
    fgr_pattern = str(Path(fgr_dir) / pattern)
    pha_pattern = str(Path(pha_dir) / pattern)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        find_ffmpeg(),
        "-y",
        "-framerate", f"{fps:.6f}",
        "-i", fgr_pattern,
        "-framerate", f"{fps:.6f}",
        "-i", pha_pattern,
    ]
    if audio_source is not None:
        cmd += ["-i", str(audio_source)]

    cmd += [
        "-filter_complex", "[0:v][1:v]alphamerge,format=yuva420p[v]",
        "-map", "[v]",
    ]
    if audio_source is not None:
        cmd += ["-map", "2:a?"]

    cmd += _video_codec_args(codec, crf)
    cmd += ["-threads", str(int(threads))]

    if audio_source is not None:
        cmd += ["-c:a", "libopus", "-b:a", audio_bitrate, "-shortest"]

    cmd += [str(out_path)]

    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed (exit {proc.returncode})")


def open_ffmpeg_rgba_pipe(
    width: int,
    height: int,
    fps: float,
    out_path: str | Path,
    crf: int = 22,
    threads: int = 8,
    audio_source: Optional[str | Path] = None,
    audio_bitrate: str = "128k",
    codec: str = VP9,
) -> subprocess.Popen:
    """Spawn ffmpeg reading raw RGBA frames from stdin, encoding to WebM yuva420p (VP9 or VP8)."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cmd: list[str] = [
        find_ffmpeg(),
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgba",
        "-s", f"{int(width)}x{int(height)}",
        "-r", f"{fps:.6f}",
        "-i", "-",
    ]
    if audio_source is not None:
        cmd += ["-i", str(audio_source), "-map", "0:v", "-map", "1:a?"]

    cmd += _video_codec_args(codec, crf)
    cmd += ["-threads", str(int(threads))]

    if audio_source is not None:
        cmd += ["-c:a", "libopus", "-b:a", audio_bitrate, "-shortest"]

    cmd += [str(out_path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)
