from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from tqdm import tqdm

from .encode.vp9_encoder import encode_webm_from_rgba_pngs, find_ffmpeg, open_ffmpeg_rgba_pipe
from .io.frames import (
    ensure_dir,
    frame_filename,
    list_png_seq,
    probe_fps,
    read_png,
    write_png,
)
from .matting.chromakey import ChromaKeyParams
from .matting.edge_refine import RefineParams, refine_frame
from .matting.rvm_runner import iter_matte_frames, load_rvm_model

app = typer.Typer(add_completion=False, help="MP4 → WebM (VP9 + alpha) green-screen matting pipeline.")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKDIR_ROOT = PROJECT_ROOT / "workdir"


def _job_dir(job: str) -> Path:
    return ensure_dir(WORKDIR_ROOT / job)


def _save_meta(job_path: Path, fps: float, source: Optional[Path] = None) -> None:
    data: dict = {"fps": fps}
    if source is not None:
        data["source"] = str(Path(source).resolve())
    (job_path / "meta.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def _load_meta(job_path: Path) -> dict:
    p = job_path / "meta.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _load_params(path: Optional[Path]) -> RefineParams:
    if path is None or not path.exists():
        return RefineParams()
    data = json.loads(path.read_text(encoding="utf-8"))
    return RefineParams.from_dict(data)


@app.command()
def matte(
    input: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    job: str = typer.Option(..., help="Job name; outputs go to workdir/<job>/"),
    downsample: Optional[float] = typer.Option(None, help="RVM downsample_ratio (auto if omitted)"),
    weights: Optional[Path] = typer.Option(None, help="Path to rvm_mobilenetv3.pth"),
) -> None:
    """Run RVM and write fgr/ + pha/ PNG sequences."""
    job_path = _job_dir(job)
    fgr_dir = ensure_dir(job_path / "fgr")
    pha_dir = ensure_dir(job_path / "pha")

    typer.echo("Loading RVM model ...")
    model = load_rvm_model(weights_path=weights)

    fps = probe_fps(input)
    _save_meta(job_path, fps, source=input)

    typer.echo(f"Running matting on {input} (fps={fps:.3f}) ...")
    n = 0
    for fgr_rgb, pha in tqdm(
        iter_matte_frames(model, input, downsample_ratio=downsample),
        desc="matte",
        unit="f",
    ):
        write_png(fgr_dir / frame_filename(n), fgr_rgb)
        write_png(pha_dir / frame_filename(n), pha)
        n += 1
    typer.echo(f"Wrote {n} frames to {job_path}")


@app.command()
def refine(
    job: str = typer.Option(..., help="Job name created by `matte`."),
    feather: float = typer.Option(1.5),
    choke: int = typer.Option(1),
    despill: float = typer.Option(1.0, help="Green channel clamp 0..2; 1.0 = clamp g≤max(r,b), >1 over-suppresses"),
    open_kernel: int = typer.Option(0, help="Morphological open kernel size; 0 = off"),
    chromakey: bool = typer.Option(True, "--chromakey/--no-chromakey"),
    fuse_mode: str = typer.Option("min"),
    decontaminate: bool = typer.Option(
        False, "--decontaminate/--no-decontaminate",
        help="Solve fg=(rgb-(1-α)·bg)/α at edges. Default OFF — only enable for stubborn green halo, dial strength carefully (can produce magenta artifacts).",
    ),
    decontam_strength: float = typer.Option(
        0.5, "--decontam-strength",
        help="Blend factor 0..1; lower if you see pink/magenta tint",
    ),
    bg_r: int = typer.Option(0, "--bg-r", help="Background RGB used by decontamination (default green: 0,255,0)"),
    bg_g: int = typer.Option(255, "--bg-g"),
    bg_b: int = typer.Option(0, "--bg-b"),
) -> None:
    """Apply OpenCV edge refinement on workdir/<job>/fgr+pha → workdir/<job>/refined/."""
    job_path = _job_dir(job)
    fgr_dir = job_path / "fgr"
    pha_dir = job_path / "pha"
    out_dir = ensure_dir(job_path / "refined")

    if not fgr_dir.exists() or not pha_dir.exists():
        raise typer.BadParameter(f"Missing fgr/ or pha/ in {job_path}. Run `matte` first.")

    params = RefineParams(
        feather=feather,
        choke=choke,
        despill=despill,
        open_kernel=open_kernel,
        use_chromakey=chromakey,
        chromakey=ChromaKeyParams(),
        fuse_mode=fuse_mode,
        decontaminate=decontaminate,
        decontam_strength=decontam_strength,
        bg_color=(bg_r, bg_g, bg_b),
    )
    (job_path / "params.json").write_text(json.dumps(params.to_dict(), indent=2), encoding="utf-8")

    fgr_files = list_png_seq(fgr_dir)
    pha_files = list_png_seq(pha_dir)
    if len(fgr_files) != len(pha_files):
        raise typer.BadParameter("fgr/ and pha/ frame counts differ; rerun `matte`.")

    for f_fgr, f_pha in tqdm(list(zip(fgr_files, pha_files)), desc="refine", unit="f"):
        fgr_rgb = read_png(f_fgr)
        pha = read_png(f_pha)
        if pha.ndim == 3:
            pha = pha[..., 0]
        fgr_out, pha_out = refine_frame(fgr_rgb, pha, params)
        rgba = np.dstack([fgr_out, pha_out]).astype(np.uint8)
        write_png(out_dir / f_fgr.name, rgba)


@app.command()
def encode(
    job: str = typer.Option(...),
    output: Path = typer.Option(..., "--output", "-o"),
    crf: int = typer.Option(32, help="VP9 CRF, 0-63. 32 sweet spot for matting; lower = bigger/cleaner"),
    threads: int = typer.Option(8),
    use_refined: bool = typer.Option(True, "--refined/--raw", help="Use refined/ if available"),
    audio: Optional[Path] = typer.Option(
        None, "--audio", help="Audio source (default: original input recorded by `matte`). Pass empty to drop."
    ),
    no_audio: bool = typer.Option(False, "--no-audio", help="Drop audio entirely."),
    audio_bitrate: str = typer.Option("128k", "--audio-bitrate"),
    codec: str = typer.Option("vp9", "--codec", help="Video codec: vp9 or vp8 (vp8 is more reliable for alpha)"),
) -> None:
    """Encode WebM (VP9 + alpha) from workdir/<job>/."""
    job_path = _job_dir(job)
    meta = _load_meta(job_path)
    fps = float(meta.get("fps", 30.0))

    refined_dir = job_path / "refined"
    if use_refined and refined_dir.exists() and any(refined_dir.glob("*.png")):
        rgba_dir = refined_dir
    else:
        rgba_dir = ensure_dir(job_path / "rgba_raw")
        fgr_dir = job_path / "fgr"
        pha_dir = job_path / "pha"
        if not fgr_dir.exists() or not pha_dir.exists():
            raise typer.BadParameter(f"Missing fgr/ or pha/ in {job_path}.")
        fgr_files = list_png_seq(fgr_dir)
        pha_files = list_png_seq(pha_dir)
        if len(fgr_files) != len(pha_files):
            raise typer.BadParameter("fgr/ and pha/ frame counts differ; rerun `matte`.")
        typer.echo(f"Merging fgr+pha → {rgba_dir} ({len(fgr_files)} frames) ...")
        for f_fgr, f_pha in tqdm(list(zip(fgr_files, pha_files)), desc="merge", unit="f"):
            fgr_rgb = read_png(f_fgr)
            pha = read_png(f_pha)
            if pha.ndim == 3:
                pha = pha[..., 0]
            fgr_rgb = fgr_rgb.copy()
            fgr_rgb[pha <= 4] = 0  # zero invisible RGB → smaller encoded file
            rgba = np.dstack([fgr_rgb, pha]).astype(np.uint8)
            write_png(rgba_dir / f_fgr.name, rgba)

    audio_source: Optional[Path] = None
    if not no_audio:
        if audio is not None:
            audio_source = audio
        elif "source" in meta and Path(meta["source"]).exists():
            audio_source = Path(meta["source"])
        else:
            typer.echo("Note: no audio source on record; output will be silent. Pass --audio to override.")

    typer.echo(
        f"Encoding {rgba_dir.name} → {output} "
        f"(codec={codec}, fps={fps:.3f}, crf={crf}, audio={'on' if audio_source else 'off'}) ..."
    )
    encode_webm_from_rgba_pngs(
        rgba_dir, fps, output,
        crf=crf, threads=threads,
        audio_source=audio_source, audio_bitrate=audio_bitrate,
        codec=codec,
    )
    typer.echo(f"Wrote {output}")


@app.command(name="all")
def run_all(
    input: Path = typer.Argument(..., exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output", "-o"),
    params_file: Optional[Path] = typer.Option(None, "--params", help="JSON of RefineParams"),
    downsample: Optional[float] = typer.Option(None),
    weights: Optional[Path] = typer.Option(None),
    crf: int = typer.Option(32, help="VP9 CRF, 0-63. 32 sweet spot for matting; lower = bigger/cleaner"),
    threads: int = typer.Option(8),
    no_audio: bool = typer.Option(False, "--no-audio", help="Drop audio."),
    audio_bitrate: str = typer.Option("128k", "--audio-bitrate"),
    codec: str = typer.Option("vp9", "--codec", help="Video codec: vp9 or vp8 (vp8 is more reliable for alpha)"),
) -> None:
    """Streaming one-shot pipeline: RVM → refine → FFmpeg pipe → WebM (no PNG on disk)."""
    typer.echo("Loading RVM model ...")
    model = load_rvm_model(weights_path=weights)
    params = _load_params(params_file)

    fps = probe_fps(input)
    audio_source: Optional[Path] = None if no_audio else input
    typer.echo(
        f"Streaming pipeline on {input} (fps={fps:.3f}, audio={'on' if audio_source else 'off'}) ..."
    )

    pipe_proc = None
    n = 0
    try:
        for fgr_rgb, pha in tqdm(
            iter_matte_frames(model, input, downsample_ratio=downsample),
            desc="all",
            unit="f",
        ):
            fgr_out, pha_out = refine_frame(fgr_rgb, pha, params)
            rgba = np.dstack([fgr_out, pha_out]).astype(np.uint8)

            if pipe_proc is None:
                h, w = rgba.shape[:2]
                pipe_proc = open_ffmpeg_rgba_pipe(
                    w, h, fps, output,
                    crf=crf, threads=threads,
                    audio_source=audio_source, audio_bitrate=audio_bitrate,
                    codec=codec,
                )

            assert pipe_proc.stdin is not None
            pipe_proc.stdin.write(rgba.tobytes())
            n += 1
    finally:
        if pipe_proc is not None and pipe_proc.stdin is not None:
            pipe_proc.stdin.close()
            pipe_proc.wait()

    typer.echo(f"Wrote {n} frames to {output}")


@app.command()
def verify(
    webm: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    keep_frame: bool = typer.Option(False, "--keep-frame", help="Keep the extracted frame.png"),
) -> None:
    """Verify a WebM has a working alpha channel.

    Decodes one frame using libvpx-vp9 (required: FFmpeg's native vp9 decoder silently
    drops alpha) and reports whether the resulting PNG has 4 channels with varying alpha.
    """
    import shutil
    import subprocess
    import tempfile

    import cv2

    ffmpeg = find_ffmpeg()
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        ffmpeg_dir = Path(ffmpeg).parent
        candidate = ffmpeg_dir / ("ffprobe.exe" if ffmpeg.lower().endswith(".exe") else "ffprobe")
        ffprobe = str(candidate) if candidate.exists() else ffmpeg

    typer.echo(f"=== {webm} ===")

    probe = subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", str(webm)],
        capture_output=True, text=True,
    )
    for line in probe.stdout.splitlines():
        if any(k in line.lower() for k in ("codec_name", "pix_fmt", "alpha", "duration", "width", "height")):
            typer.echo(f"  {line}")

    with tempfile.TemporaryDirectory() as td:
        frame_path = Path(td) / "frame.png"
        out = subprocess.run(
            [ffmpeg, "-y", "-c:v", "libvpx-vp9", "-i", str(webm),
             "-vframes", "1", "-update", "1", str(frame_path)],
            capture_output=True, text=True,
        )
        if not frame_path.exists():
            typer.echo("ERROR: frame extraction failed")
            typer.echo(out.stderr[-400:])
            raise typer.Exit(code=1)

        img = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
        channels = img.shape[-1] if img.ndim == 3 else 1
        typer.echo(f"\nDecoded frame: shape={img.shape}, channels={channels}")

        if channels == 4:
            alpha = img[..., -1]
            unique = sorted(set(alpha.flatten().tolist()))
            varying = len(unique) > 1
            min_a, max_a = int(alpha.min()), int(alpha.max())
            typer.echo(f"Alpha range: [{min_a}, {max_a}] ({len(unique)} unique values)")
            if varying and min_a < 64:
                typer.secho("[OK] Alpha channel works.", fg=typer.colors.GREEN)
            elif varying:
                typer.secho("[?] Alpha varies but stays >=64; might be encoded but with no fully-transparent pixels in this frame.", fg=typer.colors.YELLOW)
            else:
                typer.secho("[FAIL] Alpha is all-{}: not actually transparent.".format(unique[0]), fg=typer.colors.RED)
        else:
            typer.secho("[FAIL] Decoded frame has no alpha channel - encode lost it OR libvpx decoder failed.", fg=typer.colors.RED)

        if keep_frame:
            kept = webm.with_suffix(".verify_frame.png")
            shutil.copy(frame_path, kept)
            typer.echo(f"Frame saved to {kept}")


if __name__ == "__main__":
    app()
