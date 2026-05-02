from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np

from .encode.vp9_encoder import encode_webm_from_pngs
from .io.frames import ensure_dir, list_png_seq, read_png
from .matting.chromakey import ChromaKeyParams
from .matting.edge_refine import RefineParams, composite_checkerboard, refine_frame


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKDIR_ROOT = PROJECT_ROOT / "workdir"


def _load_preview_frame(job_path: Path, idx: int) -> tuple[np.ndarray, np.ndarray]:
    fgr_files = list_png_seq(job_path / "fgr")
    pha_files = list_png_seq(job_path / "pha")
    if not fgr_files:
        raise FileNotFoundError(f"No PNGs in {job_path / 'fgr'} — run `python -m src.cli matte` first")
    idx = max(0, min(idx, len(fgr_files) - 1))
    fgr = read_png(fgr_files[idx])
    pha = read_png(pha_files[idx])
    if pha.ndim == 3:
        pha = pha[..., 0]
    return fgr, pha


def build_app(job: str) -> gr.Blocks:
    job_path = WORKDIR_ROOT / job
    if not job_path.exists():
        raise FileNotFoundError(
            f"workdir/{job} not found. First run: python -m src.cli matte INPUT.mp4 --job {job}"
        )

    n_frames = len(list_png_seq(job_path / "fgr"))

    def render(
        frame_idx: int,
        feather: float,
        choke: int,
        despill: float,
        open_kernel: int,
        use_chromakey: bool,
        h_min: int,
        h_max: int,
        s_min: int,
        v_min: int,
        ck_softness: float,
        fuse_mode: str,
    ):
        fgr, pha = _load_preview_frame(job_path, int(frame_idx))
        params = RefineParams(
            feather=float(feather),
            choke=int(choke),
            despill=float(despill),
            open_kernel=int(open_kernel),
            use_chromakey=bool(use_chromakey),
            chromakey=ChromaKeyParams(
                h_min=int(h_min),
                h_max=int(h_max),
                s_min=int(s_min),
                v_min=int(v_min),
                softness=float(ck_softness),
            ),
            fuse_mode=str(fuse_mode),
        )
        fgr_out, pha_out = refine_frame(fgr, pha, params)
        preview = composite_checkerboard(fgr_out, pha_out)
        return fgr, preview, pha_out

    def save_params(*args) -> str:
        (
            feather,
            choke,
            despill,
            open_kernel,
            use_chromakey,
            h_min,
            h_max,
            s_min,
            v_min,
            ck_softness,
            fuse_mode,
        ) = args
        params = RefineParams(
            feather=float(feather),
            choke=int(choke),
            despill=float(despill),
            open_kernel=int(open_kernel),
            use_chromakey=bool(use_chromakey),
            chromakey=ChromaKeyParams(
                h_min=int(h_min),
                h_max=int(h_max),
                s_min=int(s_min),
                v_min=int(v_min),
                softness=float(ck_softness),
            ),
            fuse_mode=str(fuse_mode),
        )
        path = job_path / "params.json"
        path.write_text(json.dumps(params.to_dict(), indent=2), encoding="utf-8")
        return f"Saved → {path}"

    def render_webm(crf: int, *args) -> str:
        msg = save_params(*args)
        cmd = [
            sys.executable, "-m", "src.cli", "refine",
            "--job", job,
            "--feather", str(args[0]),
            "--choke", str(int(args[1])),
            "--despill", str(args[2]),
            "--open-kernel", str(int(args[3])),
            "--chromakey" if args[4] else "--no-chromakey",
            "--fuse-mode", str(args[10]),
        ]
        r = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if r.returncode != 0:
            return f"refine failed (exit {r.returncode})"

        out_path = job_path / f"{job}.webm"
        cmd2 = [
            sys.executable, "-m", "src.cli", "encode",
            "--job", job,
            "-o", str(out_path),
            "--crf", str(int(crf)),
        ]
        r2 = subprocess.run(cmd2, cwd=str(PROJECT_ROOT))
        if r2.returncode != 0:
            return f"{msg}\nencode failed (exit {r2.returncode})"
        return f"{msg}\nWrote {out_path}"

    with gr.Blocks(title=f"video-matting · {job}") as demo:
        gr.Markdown(f"### Edge refinement preview — job `{job}` ({n_frames} frames)")

        with gr.Row():
            frame_idx = gr.Slider(0, max(0, n_frames - 1), value=0, step=1, label="Preview frame")

        with gr.Row():
            with gr.Column():
                feather = gr.Slider(0, 5, value=1.5, step=0.1, label="Feather (Gaussian sigma)")
                choke = gr.Slider(0, 5, value=1, step=1, label="Choke (erode iterations)")
                despill = gr.Slider(0, 2, value=1.0, step=0.05, label="Despill strength (1.0 = clamp g≤max(r,b))")
                open_kernel = gr.Slider(0, 9, value=0, step=1, label="Open kernel (denoise; 0=off)")

            with gr.Column():
                use_chromakey = gr.Checkbox(value=True, label="Enable green chromakey")
                h_min = gr.Slider(0, 179, value=35, step=1, label="Hue min")
                h_max = gr.Slider(0, 179, value=85, step=1, label="Hue max")
                s_min = gr.Slider(0, 255, value=60, step=1, label="Saturation min")
                v_min = gr.Slider(0, 255, value=60, step=1, label="Value min")
                ck_softness = gr.Slider(0, 5, value=1.5, step=0.1, label="Chromakey softness")
                fuse_mode = gr.Radio(["min", "max", "mean"], value="min", label="Fuse mode")

        with gr.Row():
            crf = gr.Slider(10, 40, value=22, step=1, label="VP9 CRF (lower = higher quality)")

        with gr.Row():
            btn_preview = gr.Button("Preview")
            btn_save = gr.Button("Save params.json")
            btn_render = gr.Button("Render WebM (refine + encode)", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)

        with gr.Row():
            in_img = gr.Image(label="Input fgr", height=360)
            preview_img = gr.Image(label="Preview (over checkerboard)", height=360)
            alpha_img = gr.Image(label="Alpha", height=360)

        params_inputs = [
            feather, choke, despill, open_kernel,
            use_chromakey,
            h_min, h_max, s_min, v_min, ck_softness,
            fuse_mode,
        ]

        btn_preview.click(
            render,
            inputs=[frame_idx, *params_inputs],
            outputs=[in_img, preview_img, alpha_img],
        )

        btn_save.click(save_params, inputs=params_inputs, outputs=status)
        btn_render.click(render_webm, inputs=[crf, *params_inputs], outputs=status)

        demo.load(
            render,
            inputs=[frame_idx, *params_inputs],
            outputs=[in_img, preview_img, alpha_img],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_app(args.job)
    demo.launch(server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
