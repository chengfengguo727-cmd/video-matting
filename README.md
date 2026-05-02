# video-matting

把「人物 + 綠色背景（含少數雜點）」的 MP4 影片自動轉成「**含 alpha 透明度**的 WebM」。

```
input.mp4 ──► RVM (AI matting) ──► 綠幕色鍵雙重保險 ──► OpenCV 邊緣修飾 ──► FFmpeg VP9/yuva420p ──► out.webm
```

- **去背模型**：[Robust Video Matting (RVM)](https://github.com/PeterL1n/RobustVideoMatting) — `mobilenetv3` 變體，PyTorch + CUDA
- **雙重保險**：RVM alpha 與 HSV 綠幕色鍵以 `min()` 融合，砍掉 RVM 漏掉的綠色雜點
- **邊緣修飾**：OpenCV 模擬 AE 的 Refine Soft Matte / Matte Choker / Spill Suppressor（feather / choke / despill）
- **編碼**：FFmpeg `libvpx-vp9` + `yuva420p` + `alpha_mode=1`，產出可在 Chrome / OBS 直接使用的透明 WebM

## 環境需求

- Windows 10 / 11
- Python 3.11+
- NVIDIA GPU（建議 ≥6 GB VRAM；GTX 1070 8GB 已驗證）
- CUDA 11.8 相容驅動

## 安裝步驟

從 PowerShell 進入專案目錄：

```powershell
cd D:\claude_code\video-matting

# 1) 建立 venv 並安裝 PyTorch (CUDA 11.8) + 其餘依賴
#    腳本會自動建立 .\venv\ 並把所有東西裝在裡面，不污染全域 Python
.\scripts\install_torch_cu118.ps1

# 2) 安裝 FFmpeg（系統層級，已裝可跳過）
.\scripts\install_ffmpeg.ps1
# 裝完重啟 PowerShell 讓 PATH 生效

# 3) 下載 RVM 模型權重
.\scripts\download_rvm_weights.ps1
```

### 啟用 venv

每次新開 PowerShell 進入專案後都要先啟用：

```powershell
.\venv\Scripts\Activate.ps1
```

啟用後提示字元前面會出現 `(venv)`。後續所有 `python -m src.cli ...` 都會走 venv。
若 PowerShell 報錯 `script execution disabled`，先執行一次：
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

不想啟用也可直接呼叫 venv 的 Python：
```powershell
.\venv\Scripts\python.exe -m src.cli all .\input.mp4 -o .\out.webm
```

### 驗證

```powershell
.\venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"   # True
ffmpeg -version
```

## 使用方式

### 一鍵模式（不落地中間檔，最快）

```powershell
python -m src.cli all .\input.mp4 -o .\out.webm
```

可帶 `--params params.json` 套用先前在 Gradio 調好的參數，或 `--downsample 0.25` 手動指定。

### 分步模式（搭配 Gradio 互動調參）

```powershell
# 1) RVM 推論，輸出 PNG 序列到 workdir/<job>/fgr+pha
python -m src.cli matte .\input.mp4 --job demo

# 2) 開瀏覽器拉桿即時預覽，存 params.json
python -m src.gradio_app --job demo
#   → http://127.0.0.1:7860 ，調好按 Save params.json，或直接按 Render WebM 一鍵跑完

# 3) 套用參數做邊緣修飾
python -m src.cli refine --job demo --feather 1.5 --choke 1 --despill 0.6

# 4) 編成 WebM
python -m src.cli encode --job demo -o demo.webm --crf 22
```

## 輸出驗證

### 一鍵自動驗證

```powershell
python -m src.cli verify demo.webm
```

會依序檢查 codec / pix_fmt / ALPHA_MODE tag，再用 libvpx 解一張幀出來確認 alpha 通道有真實漸層值，最後印 `[OK] Alpha channel works.`。

### 直觀目視驗證

把 `demo.webm` 直接拖進 Chrome / Edge 一個空白分頁，背景應透出瀏覽器底色（不是黑、不是綠）。OBS 加為「媒體源」也是去背素材。

### 用 ffprobe 確認 alpha 標籤

```powershell
ffprobe -v error -show_streams demo.webm | Select-String "codec_name|pix_fmt|alpha"
# 應該看到：
#   codec_name=vp9
#   pix_fmt=yuv420p
#   TAG:ALPHA_MODE=1
```

> **注意 `pix_fmt=yuv420p` 是正常的** — libvpx-vp9 把 alpha 編在 BlockAdditional 旁帶，主視訊流仍報 yuv420p。重點是有 `ALPHA_MODE=1`。

### 抽一張幀檢查（必須用對解碼器）

```powershell
ffmpeg -y -c:v libvpx-vp9 -i demo.webm -vframes 1 -update 1 frame.png
```

`-c:v libvpx-vp9` **必須放在 `-i` 之前**，強制用 libvpx 解碼。FFmpeg 預設用自家原生 VP9 解碼器，會默默把 alpha 丟掉，導致 `frame.png` 變成 3 通道，誤判成 alpha 沒被編碼。用 libvpx 解出來才會是 4 通道 RGBA。

把 `frame.png` 拖進 Chrome → 背景透明 = 全鏈成功。

## 專案結構

```
video-matting\
├── scripts\          PowerShell 安裝腳本
├── models\           RVM 權重 (.pth)
├── src\
│   ├── cli.py        Typer CLI 入口（matte / refine / encode / all）
│   ├── gradio_app.py 互動 editor
│   ├── io\frames.py
│   ├── matting\rvm_runner.py
│   ├── matting\chromakey.py
│   ├── matting\edge_refine.py
│   └── encode\vp9_encoder.py
└── workdir\<job>\    fgr/ pha/ refined/ refined_pha/ params.json meta.json
```

## 參數調整指引

| 參數 | 範圍 | 預設 | 用途 |
|---|---|---|---|
| `feather` | 0–5 | 1.5 | Gaussian sigma；髮絲不夠軟時調高 |
| `choke` | 0–5 | 1 | 把人物剪影往內縮，邊緣綠暈嚴重時調 2 |
| `despill` | 0–2 | **1.0** | 1.0 = clamp `g ≤ max(r,b)`；>1 過度抑制（推向中性灰）|
| `open_kernel` | 0–9 | 0 | 去人物外小白點；雜點多綠幕可設 3 |
| `chromakey` HSV | — | H[35,85] S>60 V>60 | 框出綠色範圍，多數綠幕直接用預設 |
| `fuse_mode` | min/max/mean | min | min 最乾淨；人物頭髮細節易被砍時改 mean |
| `decontaminate` | on/off | **off** | 數學式扣除背景綠汙染（解 `fg=(rgb-(1-α)·bg)/α`）|
| `decontam-strength` | 0–1 | 0.5 | decontaminate 開啟時的融合強度，出現粉紅就降低 |
| `crf` | 0–63 | **32** | 32 = 去背素材甜蜜點；20 接近無損但檔很大 |

## 常見問題

- **`torch.cuda.is_available()` 為 False**：驅動未到 CUDA 11.8 等級，或安裝了 CPU-only 的 torch。重新跑 `install_torch_cu118.ps1`。
- **WebM 在 Chrome 顯示黑底不透明**：FFmpeg 缺少 `-auto-alt-ref 0` 或 `alpha_mode=1`，或播放器太舊。本專案 `vp9_encoder.py` 已正確設定。
- **驗證 alpha 用 `ffmpeg -i out.webm frame.png` 顯示沒透明**：FFmpeg 預設用原生 VP9 解碼會丟 alpha，必須加 `-c:v libvpx-vp9` 在 `-i` 之前；或直接用 `python -m src.cli verify out.webm`。
- **頭髮邊緣有綠暈**：先試 `--despill 1.5`；還不行加 `--choke 2 --feather 2.0`；最後才動 `--decontaminate --decontam-strength 0.3`。
- **decontaminate 開了出現粉紅/洋紅邊緣**：背景色非純綠 + 過度扣除。降 `--decontam-strength` 到 0.2，或設定實際背景色 `--bg-r --bg-g --bg-b`。
- **人物邊緣太鋒利**：把 `feather` 調到 2.0–3.0。
- **檔案太大**：`--crf 36` 或更高；對去背素材用途差異視覺上幾乎看不出來。

## 授權

本專案為內部工具範例，RVM 模型權重請遵守 [RVM 原始 repo](https://github.com/PeterL1n/RobustVideoMatting) 的授權條款。
