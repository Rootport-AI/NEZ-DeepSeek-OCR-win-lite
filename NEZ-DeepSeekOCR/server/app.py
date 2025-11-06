# app.py
# FastAPI + DeepSeek-OCR サーバー（SSE進捗 & 処理中プレビュー対応）
#追加：起動直後に「TFを使わない／Torchを使う」を固定
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import io
import re
import json
import base64
import asyncio
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

import torch
from PIL import Image, ImageOps
from transformers import AutoModel, AutoTokenizer

# python二重起動の原因特定用
import sys, os
print(f"[BOOT] pid={os.getpid()} exe={sys.executable}", flush=True)

#デバッグログの文字化け防止
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# PyInstaller対応
from pathlib import Path
import sys, os

def resource_path(rel: str) -> str:
    """開発時: カレント / 凍結時: _MEIPASS から相対パスを返す"""
    base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return str(Path(base) / rel)

APP_TITLE = "DeepSeek-OCR Minimal Server"
MODEL_ID = os.getenv("DEEPSEEK_OCR_MODEL", "./DeepSeek-OCR")

DEFAULT_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
DEFAULT_BASE_SIZE = int(os.getenv("BASE_SIZE", "1024"))
DEFAULT_IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))
DEFAULT_CROP_MODE = True
DEFAULT_TEST_COMPRESS = True

app = FastAPI(title=APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_tokenizer = None
_model = None

def _has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False

def get_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32

def load_model():
    global _tokenizer, _model
    if _model is not None:
        return
    device, dtype = get_device_and_dtype()
    attn_impl = "flash_attention_2" if _has_flash_attn() and device == "cuda" else "eager"
    _tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, local_files_only=True
    )
    _model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=dtype,
        use_safetensors=True,
        _attn_implementation=attn_impl,
        local_files_only=True,
    )
    if device == "cuda":
        _model = _model.cuda()
    _model.eval()

@app.get("/healthz")
def healthz():
    return {"status": "ok", "model": MODEL_ID}

# ========= 既存の同期エンドポイント（互換用。SSE版が下にあります） =========

@app.post("/ocr/image")
async def ocr_image(
    file: UploadFile = File(...),
    prompt: Optional[str] = None,
    base_size: Optional[int] = None,
    image_size: Optional[int] = None,
    crop_mode: Optional[bool] = None,
    test_compress: Optional[bool] = None,
):
    try:
        load_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load failed: {e}")

    # 入力画像を一時保存（パス渡しが堅い）
    try:
        raw = await file.read()
        suffix = ".png"
        try:
            from PIL import Image as PILImage
            fmt = PILImage.open(io.BytesIO(raw)).format
            if fmt:
                suffix = "." + fmt.lower()
        except Exception:
            pass
        tmp_dir = Path.cwd() / "_tmp_inputs"
        tmp_dir.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, dir=tmp_dir, suffix=suffix) as tf:
            tf.write(raw)
            tmp_path = Path(tf.name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image or temp save failed: {e}")

    prm = prompt or DEFAULT_PROMPT
    bs = int(base_size) if base_size else DEFAULT_BASE_SIZE
    isz = int(image_size) if image_size else DEFAULT_IMAGE_SIZE
    cm = DEFAULT_CROP_MODE if crop_mode is None else bool(crop_mode)
    tc = False if test_compress is None else bool(test_compress)

    out_dir = (Path.cwd() / "_ocr_out")
    out_dir.mkdir(exist_ok=True)
    out_dir_str = out_dir.resolve().as_posix()

    try:
        _ = _model.infer(
            _tokenizer,
            prompt=prm,
            image_file=str(tmp_path),
            output_path=out_dir_str,
            base_size=bs,
            image_size=isz,
            crop_mode=cm,
            save_results=True,   # 公式整形
            test_compress=tc,
            eval_mode=False,
        )
        md_path = Path(out_dir_str) / "result.mmd"
        text = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    finally:
        try:
            if "tmp_path" in locals() and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    return JSONResponse({"text": text})

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

def _iter_images_in_folder(folder: Path) -> List[Path]:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    def _nkey(p: Path):
        import re as _re
        parts = _re.split(r"(\d+)", p.name)
        return [int(t) if t.isdigit() else t.lower() for t in parts]
    return sorted(files, key=_nkey)

@app.post("/ocr/folder")
def ocr_folder(
    folder_path: str = Form(...),
    prompt: Optional[str] = Form(None),
    base_size: Optional[int] = Form(None),
    image_size: Optional[int] = Form(None),
    crop_mode: Optional[bool] = Form(None),
    test_compress: Optional[bool] = Form(None),
    mode: Optional[str] = Form("save_results"),
):
    try:
        load_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load failed: {e}")

    try:
        folder = Path(folder_path).expanduser().resolve(strict=True)
        if not folder.is_dir():
            raise HTTPException(status_code=400, detail="folder_path is not a directory.")
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="folder_path not found.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid folder_path: {e}")

    files = _iter_images_in_folder(folder)
    if not files:
        return JSONResponse({"text": "", "count": 0, "files": []})

    prm = prompt or DEFAULT_PROMPT
    bs = int(base_size) if base_size else DEFAULT_BASE_SIZE
    isz = int(image_size) if image_size else DEFAULT_IMAGE_SIZE
    cm = DEFAULT_CROP_MODE if crop_mode is None else bool(crop_mode)
    tc = False if test_compress is None else bool(test_compress)

    SEP = "--- --- ---  "
    accumulator: List[str] = []
    succeeded = 0

    out_base = (Path.cwd() / "_ocr_out")
    out_base.mkdir(exist_ok=True)

    for idx, img_path in enumerate(files, start=1):
        subdir = out_base / f"batch_{os.getpid()}_{idx}"
        subdir.mkdir(exist_ok=True)
        subdir_posix = subdir.resolve().as_posix()
        try:
            if mode == "eval":
                text_or_dict = _model.infer(
                    _tokenizer,
                    prompt=prm,
                    image_file=str(img_path),
                    base_size=bs,
                    image_size=isz,
                    crop_mode=cm,
                    save_results=False,
                    test_compress=tc,
                    eval_mode=True,
                )
                if isinstance(text_or_dict, str):
                    ocr_text = text_or_dict
                elif isinstance(text_or_dict, dict):
                    ocr_text = text_or_dict.get("text") or text_or_dict.get("markdown") or ""
                else:
                    ocr_text = str(text_or_dict)
            else:
                _ = _model.infer(
                    _tokenizer,
                    prompt=prm,
                    image_file=str(img_path),
                    output_path=subdir_posix,
                    base_size=bs,
                    image_size=isz,
                    crop_mode=cm,
                    save_results=True,
                    test_compress=tc,
                    eval_mode=False,
                )
                md_path = subdir / "result.mmd"
                ocr_text = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
            succeeded += 1
        except Exception as e:
            ocr_text = f"[ERROR] Inference failed: {e}"

        # 区切り → ファイル名 → 空行2つ → テキスト → 空行
        accumulator.append(SEP)
        accumulator.append(img_path.name)
        accumulator.append("")
        accumulator.append("")
        accumulator.append(ocr_text.strip())
        accumulator.append("")

        try:
            for p in subdir.iterdir():
                p.unlink(missing_ok=True)
            subdir.rmdir()
        except Exception:
            pass

    result_text = "\n".join(accumulator).rstrip() + "\n"
    return JSONResponse(
        {"text": result_text, "count": len(files), "succeeded": succeeded,
         "files": [p.name for p in files]}
    )

# ===================== ここからSSEジョブ版（進捗/サムネ） =====================

JOBS: Dict[str, Dict[str, Any]] = {}  # {job_id: {"queue": asyncio.Queue, "done": bool, "result": str, "backlog": list[str], "subscribers": int}}

def _new_job() -> str:
    import uuid
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"queue": asyncio.Queue(), "done": False, "result": "", "backlog": [], "subscribers": 0}
    return job_id

def _queue(job_id) -> asyncio.Queue:
    return JOBS[job_id]["queue"]

async def _emit(job_id: str, payload: Dict[str, Any]):
    msg = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    # まず履歴に積む（後着接続にも再送できるように）
    JOBS[job_id]["backlog"].append(msg)
    # リアルタイム接続があればキューにも流す
    if JOBS[job_id]["subscribers"] > 0:
        await _queue(job_id).put(msg)

def _thumb_dataurl_from_bytes(b: bytes, max_side: int = 320) -> str:
    im = Image.open(io.BytesIO(b))
    im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size
    if max(w, h) > max_side:
        if w >= h:
            new_w = max_side
            new_h = int(h * (max_side / w))
        else:
            new_h = max_side
            new_w = int(w * (max_side / h))
        im = im.resize((new_w, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=75)
    enc = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{enc}"

def _thumb_dataurl_from_path(path: Path, max_side: int = 320) -> str:
    with open(path, "rb") as f:
        return _thumb_dataurl_from_bytes(f.read(), max_side=max_side)

@app.post("/jobs/image")
async def jobs_image(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    base_size: Optional[int] = Form(None),
    image_size: Optional[int] = Form(None),
    crop_mode: Optional[bool] = Form(None),
    test_compress: Optional[bool] = Form(None),
    mode: Optional[str] = Form("save_results"),  # "save_results" or "eval"
):
    try:
        load_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load failed: {e}")

    raw = await file.read()
    job_id = _new_job()
    asyncio.create_task(_run_job_image(job_id, raw, {
        "prompt": prompt or DEFAULT_PROMPT,
        "base_size": int(base_size) if base_size else DEFAULT_BASE_SIZE,
        "image_size": int(image_size) if image_size else DEFAULT_IMAGE_SIZE,
        "crop_mode": DEFAULT_CROP_MODE if crop_mode is None else bool(crop_mode),
        "test_compress": False if test_compress is None else bool(test_compress),
        "mode": mode,
    }))
    return {"job_id": job_id}

async def _run_job_image(job_id: str, raw: bytes, cfg: Dict[str, Any]):
    queue = _queue(job_id)
    try:
        await _emit(job_id, {"type": "start", "pct": 0, "status": "start"})

        # サムネを先に送る
        thumb = _thumb_dataurl_from_bytes(raw, 320)
        await _emit(job_id, {"type": "preview", "thumb": thumb})

        # フェーズ: load/preprocess
        await _emit(job_id, {"type": "progress", "pct": 5, "status": "preprocess"})

        # ★ EXIF補正してから一時保存（OCRも正しい向きで）
        suffix = ".png"
        try:
            probe = Image.open(io.BytesIO(raw))
            if probe.format:
                suffix = "." + probe.format.lower()
            im = ImageOps.exif_transpose(probe).convert("RGB")
        except Exception:
            im = Image.open(io.BytesIO(raw)).convert("RGB")
        tmp_dir = Path.cwd() / "_tmp_inputs"
        tmp_dir.mkdir(exist_ok=True)
        # 一時ファイル名のみ確保（閉じてからパスに対して保存する）
        with tempfile.NamedTemporaryFile(delete=False, dir=tmp_dir, suffix=suffix) as tf:
            tmp_path = Path(tf.name)
        # 保存フォーマットは元画像の format を優先、無ければ PNG
        save_fmt = ("PNG")
        if 'probe' in locals() and getattr(probe, "format", None):
            save_fmt = probe.format
        im.save(tmp_path, format=save_fmt)

        # encode 開始
        await _emit(job_id, {"type": "progress", "pct": 25, "status": "encode"})

        # 推論
        def _infer():
            if cfg["mode"] == "eval":
                return _model.infer(
                    _tokenizer,
                    prompt=cfg["prompt"],
                    image_file=str(tmp_path),
                    base_size=cfg["base_size"],
                    image_size=cfg["image_size"],
                    crop_mode=cfg["crop_mode"],
                    save_results=False,
                    test_compress=cfg["test_compress"],
                    eval_mode=True,
                )
            else:
                out_dir = (Path.cwd() / "_ocr_out_jobs")
                out_dir.mkdir(exist_ok=True)
                out_posix = out_dir.resolve().as_posix()
                _ = _model.infer(
                    _tokenizer,
                    prompt=cfg["prompt"],
                    image_file=str(tmp_path),
                    output_path=out_posix,
                    base_size=cfg["base_size"],
                    image_size=cfg["image_size"],
                    crop_mode=cfg["crop_mode"],
                    save_results=True,
                    test_compress=cfg["test_compress"],
                    eval_mode=False,
                )
                md_path = Path(out_posix) / "result.mmd"
                return md_path.read_text(encoding="utf-8") if md_path.exists() else ""

        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(None, _infer)

        # decode ~ post
        await _emit(job_id, {"type": "progress", "pct": 95, "status": "postprocess"})

        if isinstance(res, str):
            text = res
        elif isinstance(res, dict):
            text = res.get("text") or res.get("markdown") or ""
        else:
            text = str(res)

        await _emit(job_id, {"type": "done", "pct": 100, "status": "done", "text": text})
        JOBS[job_id]["result"] = text
    except Exception as e:
        await _emit(job_id, {"type": "error", "pct": 100, "status": "error", "message": str(e)})
    finally:
        JOBS[job_id]["done"] = True
        # 入力一時ファイル掃除
        try:
            if "tmp_path" in locals() and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

@app.post("/jobs/folder")
async def jobs_folder(
    folder_path: str = Form(...),
    prompt: Optional[str] = Form(None),
    base_size: Optional[int] = Form(None),
    image_size: Optional[int] = Form(None),
    crop_mode: Optional[bool] = Form(None),
    test_compress: Optional[bool] = Form(None),
    mode: Optional[str] = Form("save_results"),
):
    try:
        load_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load failed: {e}")

    # パス検証
    try:
        folder = Path(folder_path).expanduser().resolve(strict=True)
        if not folder.is_dir():
            raise HTTPException(status_code=400, detail="folder_path is not a directory.")
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="folder_path not found.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid folder_path: {e}")

    files = _iter_images_in_folder(folder)

    job_id = _new_job()
    # ★ async 関数内なので create_task を安全に呼べる
    asyncio.create_task(_run_job_folder(job_id, files, {
        "prompt": prompt or DEFAULT_PROMPT,
        "base_size": int(base_size) if base_size else DEFAULT_BASE_SIZE,
        "image_size": int(image_size) if image_size else DEFAULT_IMAGE_SIZE,
        "crop_mode": DEFAULT_CROP_MODE if crop_mode is None else bool(crop_mode),
        "test_compress": False if test_compress is None else bool(test_compress),
        "mode": mode,
    }))
    return {"job_id": job_id, "total": len(files)}

async def _run_job_folder(job_id: str, files: List[Path], cfg: Dict[str, Any]):
    queue = _queue(job_id)
    total = len(files)
    SEP = "--- --- ---  "
    acc: List[str] = []
    try:
        await _emit(job_id, {"type": "start", "pct": 0, "status": "start", "total": total})

        done_files = 0
        for idx, img_path in enumerate(files, start=1):
            # プレビュー（サムネ）
            try:
                thumb = _thumb_dataurl_from_path(img_path, 320)
            except Exception:
                thumb = None
            await _emit(job_id, {
                "type": "file_start",
                "pct": (done_files / total * 100.0) if total else 0.0,
                "status": "file_start",
                "filename": img_path.name,
                "current_index": idx - 1,
                "total": total,
                "thumb": thumb
            })

            # 単枚%： 0.0〜1.0 の内部進捗
            async def _emit_infile(pct_in: float, status: str):
                overall = ((done_files + pct_in) / total * 100.0) if total else 100.0
                await _emit(job_id, {
                    "type": "progress",
                    "pct": overall,
                    "status": status,
                    "filename": img_path.name,
                    "current_index": idx - 1,
                    "total": total
                })

            await _emit_infile(0.05, "preprocess")

            # 推論
            def _infer_one() -> str:
                if cfg["mode"] == "eval":
                    res = _model.infer(
                        _tokenizer,
                        prompt=cfg["prompt"],
                        image_file=str(img_path),
                        base_size=cfg["base_size"],
                        image_size=cfg["image_size"],
                        crop_mode=cfg["crop_mode"],
                        save_results=False,
                        test_compress=cfg["test_compress"],
                        eval_mode=True,
                    )
                    if isinstance(res, str):
                        return res
                    elif isinstance(res, dict):
                        return res.get("text") or res.get("markdown") or ""
                    else:
                        return str(res)
                else:
                    out_dir = (Path.cwd() / "_ocr_out_jobs")
                    out_dir.mkdir(exist_ok=True)
                    out_posix = out_dir.resolve().as_posix()
                    _ = _model.infer(
                        _tokenizer,
                        prompt=cfg["prompt"],
                        image_file=str(img_path),
                        output_path=out_posix,
                        base_size=cfg["base_size"],
                        image_size=cfg["image_size"],
                        crop_mode=cfg["crop_mode"],
                        save_results=True,
                        test_compress=cfg["test_compress"],
                        eval_mode=False,
                    )
                    md_path = Path(out_posix) / "result.mmd"
                    return md_path.read_text(encoding="utf-8") if md_path.exists() else ""

            loop = asyncio.get_running_loop()
            await _emit_infile(0.25, "encode")
            text = await loop.run_in_executor(None, _infer_one)
            await _emit_infile(0.95, "postprocess")

            # 連結：区切り → ファイル名 → 空行2つ → テキスト → 空行
            acc.append(SEP)
            acc.append(img_path.name)
            acc.append("")
            acc.append("")
            acc.append(text.strip())
            acc.append("")

            done_files += 1
            await _emit(job_id, {
                "type": "file_done",
                "pct": (done_files / total * 100.0) if total else 100.0,
                "status": "file_done",
                "filename": img_path.name,
                "current_index": idx - 1,
                "total": total
            })

        final_text = "\n".join(acc).rstrip() + "\n"
        await _emit(job_id, {"type": "done", "pct": 100, "status": "done", "text": final_text})
        JOBS[job_id]["result"] = final_text
    except Exception as e:
        await _emit(job_id, {"type": "error", "pct": 100, "status": "error", "message": str(e)})
    finally:
        JOBS[job_id]["done"] = True

@app.get("/jobs/stream/{job_id}")
async def jobs_stream(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="job not found")

    async def event_generator():
        # 接続カウント
        JOBS[job_id]["subscribers"] += 1
        try:
            # まずバックログを再送（過去の file_start/preview を確実に見せる）
            for msg in JOBS[job_id]["backlog"]:
                yield msg

            queue = _queue(job_id)
            # ここからはリアルタイム分を受け渡し
            # （_emit は subscribers>0 のときだけ queue に入れる）
            while True:
                try:
                    msg = await queue.get()
                    yield msg
                    # done/error の後は閉じる
                    try:
                        payload = json.loads(msg[len("data: "):-2])
                        if payload.get("type") in ("done", "error"):
                            break
                    except Exception:
                        pass
                except asyncio.CancelledError:
                    break
        finally:
            JOBS[job_id]["subscribers"] = max(0, JOBS[job_id]["subscribers"] - 1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# 静的配信とルート/PyInstaller対応済み
static_dir = resource_path("static")
app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")

@app.get("/")
def root():
    return FileResponse(str(Path(static_dir) / "index.html"))

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

if __name__ == "__main__":
    import uvicorn
    # Supervisorをバイパスして“同一プロセス内”で起動する
    config = uvicorn.Config(app=app, host="127.0.0.1", port=8000, reload=False)
    server = uvicorn.Server(config)
    server.run()
