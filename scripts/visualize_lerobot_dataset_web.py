#!/usr/bin/env python3
"""
在浏览器中可视化任意 LeRobot 风格 parquet 数据集（按 episode_index / frame_index 切分）。

用法:
  python scripts/visualize_lerobot_dataset_web.py --dataset /path/to/dataset
  python scripts/visualize_lerobot_dataset_web.py --dataset data/rlbench_reach_target --port 8765
  python scripts/visualize_lerobot_dataset_web.py --dataset ... --episode 3

依赖: pyarrow, pillow, numpy
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pyarrow.parquet as pq
except ImportError:
    print("需要安装 pyarrow: pip install pyarrow", file=sys.stderr)
    raise

try:
    from PIL import Image
except ImportError:
    print("需要安装 pillow: pip install pillow", file=sys.stderr)
    raise


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _collect_parquet_files(dataset_dir: str) -> List[str]:
    pattern = os.path.join(dataset_dir, "data", "**", "*.parquet")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"未找到 parquet: {pattern}（需为 LeRobot 目录结构 dataset/data/**/*.parquet）")
    return files


def _image_dict_to_bytes(img_cell: Any, dataset_dir: str, parquet_path: str) -> Optional[bytes]:
    if img_cell is None:
        return None
    if isinstance(img_cell, dict):
        b = img_cell.get("bytes")
        if b:
            return bytes(b)
        rel = img_cell.get("path")
        if rel and isinstance(rel, str):
            for base in (
                dataset_dir,
                os.path.dirname(parquet_path),
                os.path.join(dataset_dir, "data"),
            ):
                cand = rel if os.path.isabs(rel) else os.path.normpath(os.path.join(base, rel))
                if os.path.isfile(cand):
                    with open(cand, "rb") as f:
                        return f.read()
    if isinstance(img_cell, (bytes, bytearray)):
        return bytes(img_cell)
    return None


def _to_jpeg_bytes(b: bytes, max_side: int = 512) -> Tuple[str, bytes]:
    im = Image.open(io.BytesIO(b)).convert("RGB")
    w, h = im.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    im.save(out, format="JPEG", quality=88)
    return "image/jpeg", out.getvalue()


def _infer_data_columns(schema_names: List[str]) -> List[str]:
    """可序列化到 JSON 的列（排除图像与索引列）。"""
    skip_exact = {
        "episode_index",
        "frame_index",
        "index",
        "timestamp",
        "task_index",
        "episode_id",
    }
    out: List[str] = []
    for name in schema_names:
        if name in skip_exact:
            continue
        if name.startswith("observation.images."):
            continue
        out.append(name)
    return sorted(out)


def _serialize_cell(cell: Any) -> Any:
    if cell is None:
        return None
    if isinstance(cell, (bool, int, float, str)):
        return cell
    if isinstance(cell, (list, tuple)):
        try:
            return [float(x) for x in cell]
        except (TypeError, ValueError):
            return [str(x) for x in cell]
    if isinstance(cell, np.ndarray):
        return cell.astype(np.float64).tolist()
    if isinstance(cell, dict):
        return {str(k): _serialize_cell(v) for k, v in cell.items()}
    return str(cell)


class DatasetIndex:
    """episode_id -> 按 frame_index 排序的 (parquet_path, row_index) 列表。"""

    def __init__(self, dataset_dir: str):
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.parquet_files = _collect_parquet_files(self.dataset_dir)
        self._tables: Dict[str, Any] = {}
        ep_rows: Dict[int, List[Dict[str, Any]]] = {}
        for fp in self.parquet_files:
            t = pq.read_table(fp, columns=["episode_index", "frame_index"])
            eps = t.column("episode_index").to_pylist()
            frs = t.column("frame_index").to_pylist()
            for row_i, (ep, fr) in enumerate(zip(eps, frs)):
                ep_rows.setdefault(int(ep), []).append(
                    {"path": fp, "row": row_i, "frame_index": int(fr)}
                )
        self.episode_ids = sorted(ep_rows.keys())
        self._by_ep: Dict[int, List[Dict[str, Any]]] = {}
        for ep in self.episode_ids:
            self._by_ep[ep] = sorted(ep_rows[ep], key=lambda x: x["frame_index"])

        schema = pq.read_schema(self.parquet_files[0])
        self.camera_keys = sorted(
            c.replace("observation.images.", "")
            for c in schema.names
            if c.startswith("observation.images.")
        )
        self.data_columns = _infer_data_columns(list(schema.names))

    def _get_table(self, path: str):
        if path not in self._tables:
            self._tables[path] = pq.read_table(path)
        return self._tables[path]

    def num_frames(self, episode_id: int) -> int:
        return len(self._by_ep[episode_id])

    def get_frame(self, episode_id: int, frame_ord: int) -> Dict[str, Any]:
        rows = self._by_ep[episode_id]
        if frame_ord < 0 or frame_ord >= len(rows):
            raise IndexError("frame_ord out of range")
        ref = rows[frame_ord]
        tbl = self._get_table(ref["path"])
        r = ref["row"]
        out: Dict[str, Any] = {
            "episode_index": int(tbl.column("episode_index")[r].as_py()),
            "frame_index": int(tbl.column("frame_index")[r].as_py()),
        }
        if "timestamp" in tbl.column_names:
            out["timestamp"] = tbl.column("timestamp")[r].as_py()

        for col in self.data_columns:
            if col not in tbl.column_names:
                continue
            cell = tbl.column(col)[r].as_py()
            try:
                out[col] = _serialize_cell(cell)
            except Exception:
                out[col] = str(cell)

        for ck in self.camera_keys:
            col = f"observation.images.{ck}"
            cell = tbl.column(col)[r].as_py()
            b = _image_dict_to_bytes(cell, self.dataset_dir, ref["path"])
            out[f"_img_bytes_{ck}"] = b
        return out


def make_handler(index: DatasetIndex, default_episode: Optional[int]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, obj: Any, code: int = 200):
            b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self._send(code, b, "application/json; charset=utf-8")

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path

            if path == "/" or path == "/index.html":
                de = default_episode if default_episode is not None else index.episode_ids[0]
                html = (
                    PAGE_HTML.replace("__DEFAULT_EP__", str(de))
                    .replace("__CAMERAS_JSON__", json.dumps(index.camera_keys))
                    .replace("__TITLE__", os.path.basename(index.dataset_dir.rstrip(os.sep)))
                )
                self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
                return

            if path == "/api/meta":
                self._send_json(
                    {
                        "dataset_dir": index.dataset_dir,
                        "dataset_name": os.path.basename(index.dataset_dir.rstrip(os.sep)),
                        "episode_ids": index.episode_ids,
                        "cameras": index.camera_keys,
                        "data_fields": index.data_columns,
                        "num_parquet_files": len(index.parquet_files),
                    }
                )
                return

            m = re.match(r"^/api/episode/(\d+)/info$", path)
            if m:
                ep = int(m.group(1))
                if ep not in index._by_ep:
                    self._send_json({"error": "unknown episode"}, 404)
                    return
                self._send_json({"episode_id": ep, "num_frames": index.num_frames(ep)})
                return

            m = re.match(r"^/api/episode/(\d+)/frame/(\d+)/data$", path)
            if m:
                ep = int(m.group(1))
                fi = int(m.group(2))
                try:
                    row = index.get_frame(ep, fi)
                except (KeyError, IndexError) as e:
                    self._send_json({"error": str(e)}, 404)
                    return
                payload = {k: v for k, v in row.items() if not k.startswith("_img_bytes_")}
                self._send_json(payload)
                return

            m = re.match(r"^/api/episode/(\d+)/frame/(\d+)/image/([^/]+)$", path)
            if m:
                ep = int(m.group(1))
                fi = int(m.group(2))
                cam = m.group(3)
                if cam not in index.camera_keys:
                    self._send_json({"error": "unknown camera"}, 404)
                    return
                try:
                    row = index.get_frame(ep, fi)
                except (KeyError, IndexError):
                    self.send_error(404)
                    return
                raw = row.get(f"_img_bytes_{cam}")
                if not raw:
                    self._send_json({"error": "no image bytes"}, 404)
                    return
                try:
                    ctype, body = _to_jpeg_bytes(raw)
                except Exception as e:
                    self._send_json({"error": f"decode image: {e}"}, 500)
                    return
                self._send(200, body, ctype)
                return

            self.send_error(404)

    return Handler


PAGE_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>__TITLE__ — LeRobot 可视化</title>
  <style>
    :root { --bg:#0f1419; --panel:#1a2332; --text:#e7ecf3; --accent:#5b9cf5; --muted:#8b9cb3; }
    * { box-sizing: border-box; }
    body { font-family: ui-sans-serif, system-ui, sans-serif; margin:0; background:var(--bg); color:var(--text); }
    header { padding:1rem 1.25rem; background:var(--panel); border-bottom:1px solid #2a3544; }
    h1 { margin:0; font-size:1.1rem; font-weight:600; }
    .sub { color:var(--muted); font-size:0.85rem; margin-top:0.35rem; word-break: break-all; }
    main { padding:1rem 1.25rem; max-width:1400px; margin:0 auto; }
    .controls { display:flex; flex-wrap:wrap; gap:1rem; align-items:flex-end; margin-bottom:1rem; }
    label { color:var(--muted); font-size:0.8rem; display:block; margin-bottom:0.25rem; }
    select, input[type=range] { accent-color: var(--accent); }
    select { background:#243044; color:var(--text); border:1px solid #3d4f66; border-radius:6px; padding:0.4rem 0.6rem; }
    .frame-readout { font-variant-numeric: tabular-nums; min-width:8rem; color:var(--muted); padding-bottom:0.35rem; }
    /* 向量数据在相机下方，固定高度，避免播放时整页跳动 */
    .vector-panel {
      display:grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap:0.75rem;
      margin-top:1rem;
      min-height: 14rem;
    }
    .vec-card {
      background:var(--panel);
      border:1px solid #2a3544;
      border-radius:8px;
      padding:0.5rem 0.75rem 0.75rem;
      display:flex;
      flex-direction:column;
      min-height: 13rem;
    }
    .vec-card label { font-size:0.72rem; color:var(--muted); margin-bottom:0.35rem; word-break: break-all; }
    .vec-card pre {
      flex:1;
      margin:0;
      min-height: 10rem;
      max-height: 14rem;
      overflow:auto;
      background:#121820;
      border:1px solid #2a3544;
      border-radius:6px;
      padding:0.5rem;
      font-size:0.68rem;
      line-height:1.35;
    }
    .grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(280px,1fr)); gap:0.75rem; }
    .card { background:var(--panel); border-radius:8px; overflow:hidden; border:1px solid #2a3544; }
    .card h3 { margin:0; padding:0.5rem 0.75rem; font-size:0.75rem; color:var(--muted); font-weight:500; }
    .card .img-wrap {
      position:relative;
      width:100%;
      aspect-ratio: 1;
      background:#0a0e14;
    }
    .card img {
      position:absolute;
      inset:0;
      width:100%;
      height:100%;
      object-fit: contain;
      display:block;
    }
    button { background:var(--accent); color:#0a0e14; border:none; padding:0.45rem 0.9rem; border-radius:6px; font-weight:600; cursor:pointer; }
    button:hover { filter:brightness(1.08); }
    .err { color:#f87171; font-size:0.85rem; margin-bottom:0.5rem; }
  </style>
</head>
<body>
  <header>
    <h1>LeRobot 数据集 — Episode 浏览器</h1>
    <div class="sub" id="datasetPath"></div>
  </header>
  <main>
    <div id="err" class="err"></div>
    <div class="controls">
      <div>
        <label>Episode</label>
        <select id="epSelect"></select>
      </div>
      <div style="flex:1; min-width:200px;">
        <label>帧 <span id="frameLabel"></span></label>
        <input type="range" id="frameSlider" min="0" max="0" value="0"/>
      </div>
      <div class="frame-readout" id="frameReadout"></div>
      <div>
        <label>&nbsp;</label>
        <button type="button" id="btnPlay">播放</button>
      </div>
    </div>
    <div class="grid" id="imgGrid"></div>
    <div class="vector-panel" id="vectorPanel"></div>
  </main>
  <script>
    const DEFAULT_EP = __DEFAULT_EP__;
    let CAMERAS = __CAMERAS_JSON__;
    let DATA_FIELDS = [];
    let meta = null;
    let playTimer = null;
    let numFrames = 0;
    const fieldPre = {};
    let frameToken = 0;

    function setImageSrcStable(img, url, token) {
      const loader = new Image();
      loader.onload = () => {
        if (token === frameToken) img.src = url;
      };
      loader.onerror = () => {
        if (token === frameToken) img.alt = 'load error';
      };
      loader.src = url;
    }

    function buildVectorPanel(fields) {
      const panel = document.getElementById('vectorPanel');
      panel.innerHTML = '';
      Object.keys(fieldPre).forEach(k => delete fieldPre[k]);
      fields.forEach((field, idx) => {
        const card = document.createElement('div');
        card.className = 'vec-card';
        const lab = document.createElement('label');
        lab.textContent = field;
        const pre = document.createElement('pre');
        pre.id = 'vec_pre_' + idx;
        pre.textContent = '—';
        card.appendChild(lab);
        card.appendChild(pre);
        panel.appendChild(card);
        fieldPre[field] = pre;
      });
    }

    function buildImageGrid() {
      const grid = document.getElementById('imgGrid');
      grid.innerHTML = '';
      for (const cam of CAMERAS) {
        const card = document.createElement('div');
        card.className = 'card';
        const h = document.createElement('h3');
        h.textContent = cam;
        const wrap = document.createElement('div');
        wrap.className = 'img-wrap';
        const img = document.createElement('img');
        img.alt = cam;
        img.dataset.cam = cam;
        wrap.appendChild(img);
        card.appendChild(h);
        card.appendChild(wrap);
        grid.appendChild(card);
      }
    }

    async function loadMeta() {
      const r = await fetch('/api/meta');
      meta = await r.json();
      DATA_FIELDS = meta.data_fields || [];
      CAMERAS = meta.cameras || CAMERAS;
      document.getElementById('datasetPath').textContent = meta.dataset_dir;
      const sel = document.getElementById('epSelect');
      sel.innerHTML = '';
      for (const id of meta.episode_ids) {
        const o = document.createElement('option');
        o.value = id;
        o.textContent = 'Episode ' + id;
        sel.appendChild(o);
      }
      sel.value = String(meta.episode_ids.includes(DEFAULT_EP) ? DEFAULT_EP : meta.episode_ids[0]);
      buildVectorPanel(DATA_FIELDS);
      buildImageGrid();
      await onEpisodeChange();
    }

    function setError(msg) {
      document.getElementById('err').textContent = msg || '';
    }

    async function onEpisodeChange() {
      setError('');
      const ep = parseInt(document.getElementById('epSelect').value, 10);
      const r = await fetch('/api/episode/' + ep + '/info');
      const j = await r.json();
      if (j.error) { setError(j.error); return; }
      numFrames = j.num_frames;
      const slider = document.getElementById('frameSlider');
      slider.max = Math.max(0, numFrames - 1);
      slider.value = 0;
      document.getElementById('frameLabel').textContent = '(0 … ' + (numFrames - 1) + ')';
      await showFrame(0);
    }

    async function showFrame(i) {
      setError('');
      const ep = parseInt(document.getElementById('epSelect').value, 10);
      if (numFrames <= 0) return;
      i = Math.max(0, Math.min(numFrames - 1, i));
      const token = ++frameToken;
      document.getElementById('frameSlider').value = i;
      document.getElementById('frameReadout').textContent = 'frame_ord ' + i + ' / ' + (numFrames - 1);

      const urlBase = '/api/episode/' + ep + '/frame/' + i + '/image/';
      document.querySelectorAll('#imgGrid img').forEach((img) => {
        const cam = img.dataset.cam;
        setImageSrcStable(img, urlBase + encodeURIComponent(cam), token);
      });

      try {
        const dr = await fetch('/api/episode/' + ep + '/frame/' + i + '/data');
        const d = await dr.json();
        if (token !== frameToken) return;
        if (d.error) { setError(d.error); return; }
        for (const field of DATA_FIELDS) {
          const pre = fieldPre[field];
          if (!pre) continue;
          if (d[field] !== undefined && d[field] !== null) {
            pre.textContent = typeof d[field] === 'object'
              ? JSON.stringify(d[field], null, 2)
              : String(d[field]);
          } else {
            pre.textContent = '—';
          }
        }
      } catch (e) {
        setError(String(e));
      }
    }

    document.getElementById('epSelect').addEventListener('change', () => onEpisodeChange());
    document.getElementById('frameSlider').addEventListener('input', (e) => {
      showFrame(parseInt(e.target.value, 10));
    });
    document.getElementById('btnPlay').addEventListener('click', () => {
      if (playTimer) {
        clearInterval(playTimer);
        playTimer = null;
        document.getElementById('btnPlay').textContent = '播放';
        return;
      }
      document.getElementById('btnPlay').textContent = '暂停';
      playTimer = setInterval(() => {
        let v = parseInt(document.getElementById('frameSlider').value, 10) + 1;
        if (v >= numFrames) v = 0;
        showFrame(v);
      }, 120);
    });

    loadMeta().catch(e => setError(String(e)));
  </script>
</body>
</html>
"""


def _default_dataset_dir() -> str:
    return os.path.join(_project_root(), "data", "rlbench_reach_target")


def main():
    ap = argparse.ArgumentParser(description="浏览器可视化 LeRobot parquet 数据集")
    ap.add_argument(
        "--dataset",
        type=str,
        default=_default_dataset_dir(),
        help="数据集根目录（含 data/**/*.parquet，需有 episode_index / frame_index 列）",
    )
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--episode", type=int, default=None, help="打开页面时默认选中的 episode_id")
    args = ap.parse_args()

    dataset_dir = os.path.abspath(args.dataset)
    if not os.path.isdir(dataset_dir):
        print(f"目录不存在: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    print("正在索引 parquet（首次会稍慢）…")
    index = DatasetIndex(dataset_dir)
    print(f"  episodes: {len(index.episode_ids)}  id {index.episode_ids[0]}…{index.episode_ids[-1]}")
    print(f"  相机: {', '.join(index.camera_keys) or '(无)'}")
    print(f"  数据列: {len(index.data_columns)} 个")

    handler = make_handler(index, args.episode)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"请在浏览器打开: {url}")
    print("Ctrl+C 结束")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")


if __name__ == "__main__":
    main()
