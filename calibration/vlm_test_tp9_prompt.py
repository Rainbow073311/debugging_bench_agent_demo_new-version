"""Test VLM: find TP9 text + silver test pad above it."""
import base64, json, os, re, sys, cv2
from pathlib import Path

_ENV = Path(__file__).resolve().parent.parent / "new_vlm_agent" / ".env"
if _ENV.exists():
    for _l in _ENV.read_text().splitlines():
        _l = _l.strip()
        if _l and not _l.startswith("#") and "=" in _l:
            k, v = _l.split("=", 1)
            if k.strip() not in os.environ: os.environ[k.strip()] = v.strip()

api_key = os.environ.get("VLM_API_KEY")
base_url = os.environ.get("VLM_BASE_URL", "").rstrip("/")
model = os.environ.get("VLM_MODEL", "qwen3.7-plus")

ultra_img = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/32825/Desktop/new_version_demo/new_vlm_agent/workspace/vlm-agent-runs/case-316-run_msedgxxc_lrie90la/ultra_close_20260804_160730_027970.jpg"
cx = int(sys.argv[2]) if len(sys.argv) > 2 else 910
cy = int(sys.argv[3]) if len(sys.argv) > 3 else 1236
roi_half = int(sys.argv[4]) if len(sys.argv) > 4 else 350

img = cv2.imread(ultra_img)
if img is None:
    print(json.dumps({"ok": False, "error": "cannot read image"}))
    sys.exit(1)

h, w = img.shape[:2]
x1 = max(0, cx - roi_half); y1 = max(0, cy - roi_half)
x2 = min(w, cx + roi_half); y2 = min(h, cy + roi_half)
roi = img[y1:y2, x1:x2]
roi_h, roi_w = roi.shape[:2]

roi_path = str(Path(ultra_img).parent / "ultra_tp9_pad_test.jpg")
cv2.imwrite(roi_path, roi, [cv2.IMWRITE_JPEG_QUALITY, 92])

_, buf = cv2.imencode('.jpg', roi, [cv2.IMWRITE_JPEG_QUALITY, 92])
roi_b64 = base64.b64encode(buf).decode()

prompt = """请直接检查图片。

1. 图片中是否存在文字"TP9"？
2. 如果存在，请描述它周围的元件文字。
3. 返回TP9文字中心的大致归一化坐标，坐标范围0到1000。
4. 在TP9文字的上方，找到银色圆形焊盘测试点，返回该焊盘中心的大致归一化坐标，坐标范围0到1000。
5. 不要根据电路知识猜测，只根据图片回答。"""

import requests as _r
resp = _r.post(f"{base_url}/chat/completions",
    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    json={"model": model, "messages": [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{roi_b64}"}},
    ]}], "max_tokens": 400, "temperature": 0.0},
    timeout=(30, 180))

content = resp.json()["choices"][0]["message"]["content"]
print("=" * 60)
print("VLM Response:")
print(content)
print("=" * 60)

# Parse normalized coords — look for coordinate patterns
coord_patterns = re.findall(r'[\[(]\s*(\d+)\s*[,，]\s*(\d+)\s*[\])]', content or "")
print(f"\nAll coord patterns: {coord_patterns}")

norm_candidates = [(int(x), int(y)) for x, y in coord_patterns
                   if 0 <= int(x) <= 1000 and 0 <= int(y) <= 1000]
if norm_candidates:
    print(f"Valid normalized coords (0-1000): {norm_candidates}")
    for i, (nx, ny) in enumerate(norm_candidates):
        px = int(x1 + nx / 1000.0 * roi_w)
        py = int(y1 + ny / 1000.0 * roi_h)
        label = ["text?", "pad?"][i] if i < 2 else f"coord{i+1}"
        print(f"  [{i}] norm=({nx},{ny}) -> full pixel=({px},{py}) ({label})")
else:
    print("No valid normalized coords found!")

print(f"\nROI: [{x1},{y1},{x2},{y2}] size={roi_w}x{roi_h}")
print(f"ROI saved: {roi_path}")
