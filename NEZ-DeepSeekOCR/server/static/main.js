// static/main.js
const fileInput = document.getElementById("fileInput");
const dropzone = document.getElementById("dropzone");
const runBtn = document.getElementById("runBtn");
const runFolderBtn = document.getElementById("runFolderBtn");
const folderPathInput = document.getElementById("folderPath");

const output = document.getElementById("output");
const statusEl = document.getElementById("status");
const copyBtn = document.getElementById("copyBtn");
const pingBtn = document.getElementById("pingBtn");
const healthEl = document.getElementById("health");
const previewWrap = document.getElementById("preview");
const previewImg = document.getElementById("previewImg");
const promptTA = document.getElementById("prompt");

// Loader refs
const loader = document.getElementById("loader");
const progBar = document.getElementById("progBar");
const progText = document.getElementById("progText");

let currentFile = null;
let isBusy = false;
let es = null; // EventSource

/* ---------- ボタン活性/非活性の一元管理 ---------- */
function updateButtons() {
  const hasFolder = folderPathInput?.value?.trim().length > 0;
  const hasFile = !!currentFile;
  runFolderBtn.disabled = isBusy || !hasFolder;
  runBtn.disabled = isBusy || hasFolder || !hasFile;
  copyBtn.disabled = !output.value;
}
function setStatus(msg) { statusEl.textContent = msg; }

/* ---------- プレビュー表示 ---------- */
function showPreview(file) {
  const emptyEl = document.getElementById("previewEmpty");  
  if (!file) {
    if (emptyEl) emptyEl.hidden = false;
    if (previewImg) {
      previewImg.src = "";
      previewImg.hidden = true; // 画像未設定時はアイコン/altを出さない
    }
    return;
  }
  // 1024x1024以内に収まるようにプレビューだけ縮小（推論は元ファイルを使用）
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    const maxSide = 1024;
    const w = img.naturalWidth, h = img.naturalHeight;
    let tw = w, th = h;
    if (w > maxSide || h > maxSide) {
      if (w >= h) { tw = maxSide; th = Math.round(h * (maxSide / w)); }
      else { th = maxSide; tw = Math.round(w * (maxSide / h)); }
    }
    const canvas = document.createElement("canvas");
    canvas.width = tw; canvas.height = th;
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(img, 0, 0, tw, th);
    canvas.toBlob((blob) => {
        if (!blob) {
        previewImg.onload = () => {
          if (emptyEl) emptyEl.hidden = true;
          previewImg.hidden = false;
          };
        previewImg.src = url;
        return;
      }
      const purl = URL.createObjectURL(blob);
      previewImg.onload = () => { 
        URL.revokeObjectURL(purl); 
        previewImg.hidden = false; 
      };
      previewImg.src = purl;
      if (emptyEl) emptyEl.hidden = true;
      URL.revokeObjectURL(url);
    }, "image/jpeg", 0.95);
  };
  img.onerror = () => { 
    previewImg.src = url; 
    if (emptyEl) emptyEl.hidden = true; 
    previewImg.hidden = false; 
  };
  img.src = url;
}

/* ---------- Loader（実値をSSEで更新） ---------- */
function setProgress(p) {
  const v = Math.max(0, Math.min(100, Number.isFinite(p) ? p : 0));
  progBar.style.width = `${v}%`;
  progText.textContent = `${Math.floor(v)}%`;
}
function openLoader() {
  loader.hidden = false;
  setProgress(0);
}
function closeLoader() {
  loader.hidden = true;
  setProgress(0);
}

/* ---------- ドラッグ&ドロップ（全画面） ---------- */
const overlay = document.getElementById("globalDropOverlay");
const overlayNote = document.getElementById("overlayNote");
const openFileBtn = document.getElementById("openFileBtn");

if (openFileBtn) openFileBtn.addEventListener("click", () => fileInput.click());

let dragCounter = 0;
function isFileDrag(e) {
  const dt = e.dataTransfer;
  if (!dt) return false;
  return Array.from(dt.types || []).includes("Files");
}

window.addEventListener("dragenter", (e) => {
  if (!isFileDrag(e)) return;
  e.preventDefault();
  dragCounter++;
  overlayNote.textContent = "";
  overlay.hidden = false;
});

window.addEventListener("dragover", (e) => {
  if (!isFileDrag(e)) return;
  e.preventDefault();
});

window.addEventListener("dragleave", (e) => {
  if (!isFileDrag(e)) return;
  e.preventDefault();
  dragCounter = Math.max(0, dragCounter - 1);
  if (dragCounter === 0) overlay.hidden = true;
});

window.addEventListener("drop", (e) => {
  if (!isFileDrag(e)) return;
  e.preventDefault();
  const files = e.dataTransfer.files;
  if (!files || files.length === 0) { overlay.hidden = true; return; }
  if (files.length > 1) {
    overlayNote.textContent = "複数のファイルがドロップされました。先頭の1枚のみ読み込みます。";
  }
  const file = files[0];
  if (file && file.type && file.type.startsWith("image/")) {
    currentFile = file; showPreview(file);
    updateButtons();
  }
  // オーバーレイは少しだけ注釈を見せてから閉じる
  setTimeout(() => { overlay.hidden = true; dragCounter = 0; }, overlayNote.textContent ? 900 : 0);
});

window.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !overlay.hidden) {
    overlay.hidden = true; dragCounter = 0;
  }
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) { currentFile = file; showPreview(file); }
  updateButtons();
  try { fileInput.value = ""; } catch {}
});

folderPathInput.addEventListener("input", updateButtons);

/* ---------- SSE ハンドリング ---------- */
function attachSSE(jobId, onDone) {
  if (es) { es.close(); es = null; }
  es = new EventSource(`/jobs/stream/${jobId}`);
  es.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data);
      const t = data.type;

      if (t === "start") {
        openLoader();
        setProgress(data.pct ?? 0);
        setStatus("開始");
      }
      if (t === "preview" && data.thumb) {
        previewImg.src = data.thumb;
        const emptyEl = document.getElementById("previewEmpty");
        if (emptyEl) emptyEl.hidden = true;
        previewImg.hidden = false;             
        previewWrap.classList.remove("hidden");
      }
      if (t === "file_start") {
        setStatus(`処理中: ${data.filename} (${(data.current_index + 1)}/${data.total})`);
        if (data.thumb) {
          previewImg.src = data.thumb;
          const emptyEl = document.getElementById("previewEmpty");
          if (emptyEl) emptyEl.hidden = true;
          previewImg.hidden = false;               // ★ フォルダOCRでも確実に表示
          previewWrap.classList.remove("hidden");
        }
        setProgress(data.pct ?? 0);
      }
      if (t === "progress") {
        if (typeof data.pct === "number") setProgress(data.pct);
        if (data.filename && data.total) {
          setStatus(`処理中: ${data.filename} (${(data.current_index + 1)}/${data.total})`);
        } else if (data.status) {
          setStatus(`処理中: ${data.status}`);
        }
      }
      if (t === "file_done") {
        if (typeof data.pct === "number") setProgress(data.pct);
      }
      if (t === "done") {
        setProgress(100);
        if (data.text) output.value = data.text;
        setStatus("完了");
        copyBtn.disabled = !output.value;
        closeLoader();
        es.close(); es = null;
        onDone?.();
      }
      if (t === "error") {
        setStatus(`エラー: ${data.message || "unknown"}`);
        closeLoader();
        es.close(); es = null;
        onDone?.(new Error(data.message || "error"));
      }
    } catch (e) {
      // パース失敗は握りつぶす（SSEは複数行になることもあるため）
    }
  };
  es.onerror = () => {
    // ストリーム断はリトライしない（ジョブ1回ごとに開く設計）
  };
}

/* ---------- 単枚OCR（SSEジョブ版） ---------- */
runBtn.addEventListener("click", async () => {
  if (!currentFile || runBtn.disabled) return;
  isBusy = true; updateButtons();
  output.value = ""; copyBtn.disabled = true;
  setStatus("送信中…");

  const form = new FormData();
  form.append("file", currentFile);
  form.append("prompt", promptTA.value);
  form.append("base_size", document.getElementById("baseSize").value);
  form.append("image_size", document.getElementById("imageSize").value);
  form.append("crop_mode", document.getElementById("cropMode").checked ? "true" : "false");
  form.append("test_compress", document.getElementById("testCompress").checked ? "true" : "false");
  form.append("mode", "save_results"); // 体裁重視

  try {
    const res = await fetch("/jobs/image", { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const { job_id } = await res.json();
    attachSSE(job_id, () => { isBusy = false; updateButtons(); });
  } catch (e) {
    setStatus(`エラー: ${e.message}`);
    isBusy = false; updateButtons();
  }
});

/* ---------- フォルダ一括OCR（SSEジョブ版） ---------- */
runFolderBtn.addEventListener("click", async () => {
  const folderPath = folderPathInput.value.trim();
  if (!folderPath || runFolderBtn.disabled) return;

  isBusy = true; updateButtons();
  output.value = ""; copyBtn.disabled = true;
  setStatus("フォルダOCRを実行中…");

  const form = new FormData();
  form.append("folder_path", folderPath);
  form.append("prompt", promptTA.value);
  form.append("base_size", document.getElementById("baseSize").value);
  form.append("image_size", document.getElementById("imageSize").value);
  form.append("crop_mode", document.getElementById("cropMode").checked ? "true" : "false");
  form.append("test_compress", document.getElementById("testCompress").checked ? "true" : "false");
  form.append("mode", "save_results"); // 体裁重視の既定

  try {
    const res = await fetch("/jobs/folder", { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const { job_id } = await res.json();
    attachSSE(job_id, () => { isBusy = false; updateButtons(); });
  } catch (e) {
    setStatus(`エラー: ${e.message}`);
    isBusy = false; updateButtons();
  }
});

/* ---------- クリップボード ---------- */
copyBtn.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(output.value);
    setStatus("コピーしました");
  } catch {
    setStatus("コピー失敗");
  }
});

/* ---------- ヘルスチェック ---------- */
pingBtn.addEventListener("click", async () => {
  healthEl.textContent = "確認中…";
  try {
    const res = await fetch("/healthz");
    const data = await res.json();
    healthEl.textContent = `OK (${data.model})`;
  } catch {
    healthEl.textContent = "失敗";
  }
});

/* ---------- プロンプト・プリセット＆強調表示 ---------- */
function normalizePrompt(s) {
  if (!s) return "";
  return s
    .replace(/\u3000/g, " ")
    .replace(/\r\n?/g, "\n")
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();
}
function updatePresetHighlights() {
  const current = normalizePrompt(promptTA.value);
  document.querySelectorAll(".preset-btn").forEach((btn) => {
    const p = btn.getAttribute("data-prompt") || "";
    const match = current === normalizePrompt(p);
    btn.classList.toggle("active", match);
    btn.setAttribute("aria-pressed", match ? "true" : "false");
  });
}
document.querySelectorAll(".preset-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const p = btn.getAttribute("data-prompt") || "";
    promptTA.value = p;
    promptTA.classList.add("flash");
    setTimeout(() => promptTA.classList.remove("flash"), 250);
    updatePresetHighlights();
  });
});
let _timer = null;
promptTA.addEventListener("input", () => {
  clearTimeout(_timer);
  _timer = setTimeout(updatePresetHighlights, 80);
});

/* ---------- 初期化 ---------- */
document.addEventListener("DOMContentLoaded", () => {
  updateButtons();
  updatePresetHighlights();
});
