"""Intake: raw phone photos -> uniform passport crops + cropped signatures, auto-named by OCR.

Everything runs offline (OpenCV face detection + RapidOCR). Student photos never leave this machine.
Working data lives in  <base>/_intake/<class>/  :  raw_photo/  raw_sign/  out/  records.json
"""
import difflib
import json
import queue
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import engine as E

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None
try:
    from rapidocr_onnxruntime import RapidOCR
except Exception:  # pragma: no cover
    RapidOCR = None

LANCZOS = Image.Resampling.LANCZOS
LOCK = threading.RLock()


def capabilities():
    return {"opencv": cv2 is not None, "ocr": RapidOCR is not None}


# ============================================================ OCR + name extraction
_ocr, _ocr_lock = None, threading.Lock()


def _engine():
    global _ocr
    if _ocr is None and RapidOCR is not None:
        _ocr = RapidOCR()
    return _ocr


def ocr_lines(pil, max_side=1600):
    """OCR -> [{text, conf, box=(x0,y0,x1,y1)}] in the coordinates of `pil`, top-to-bottom."""
    eng = _engine()
    if eng is None:
        return []
    w, h = pil.size
    s = min(1.0, max_side / max(w, h))
    im = pil.convert("RGB")
    if s < 1:
        im = im.resize((round(w * s), round(h * s)), LANCZOS)
    with _ocr_lock:
        res, _ = eng(np.array(im)[:, :, ::-1])
    out = []
    for box, text, score in (res or []):
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        out.append({"text": text, "conf": float(score),
                    "box": (min(xs) / s, min(ys) / s, max(xs) / s, max(ys) / s)})
    out.sort(key=lambda l: (l["box"][1], l["box"][0]))
    return out


CLASS_RE = re.compile(r"^\s*(S|SENIOR|FORM|CLASS|STD)?\s*[.:\-]?\s*[1-6]\s*$", re.I)
STOP = {"NAME", "NAMES", "CLASS", "FORM", "STD", "SENIOR", "NO", "YES", "SIGN", "SIGNATURE", "STUDENT"}


def clean_text(t):
    """Keep only name-like text: drops class markers (S4, S.4), numbers, label words."""
    t = t.upper()
    if CLASS_RE.match(t):
        return ""
    t = re.sub(r"[^A-Z'\- ]", " ", t)
    return " ".join(x for x in t.split() if x not in STOP and len(x) > 1)


def sim(a, b):
    a, b = E.norm(a), E.norm(b)
    r1 = difflib.SequenceMatcher(None, a, b).ratio()
    r2 = difflib.SequenceMatcher(None, " ".join(sorted(a.split())), " ".join(sorted(b.split()))).ratio()
    return max(r1, r2)


def candidates_from_lines(lines, min_y=None):
    cand = []
    for l in lines:
        if min_y is not None and (l["box"][1] + l["box"][3]) / 2 < min_y:
            continue
        c = clean_text(l["text"])
        if len(c.replace(" ", "")) >= 3:
            cand.append([c, round(l["conf"], 3)])
    return cand


def name_from_candidates(cand, roster):
    """cand: [[text, conf], ...] in reading order. Returns dict(name, conf, via, flags)."""
    if not cand:
        return {"name": "", "conf": 0.0, "via": "none", "flags": ["no_name"]}
    strings = [c for c, _ in cand]
    strings += [cand[i][0] + " " + cand[i + 1][0] for i in range(len(cand) - 1)]
    if len(cand) > 2:
        strings.append(" ".join(c for c, _ in cand[:3]))
    if roster:
        best = (0.0, "")
        for s in strings:
            for r in roster:
                sc = sim(s, r)
                if sc > best[0]:
                    best = (sc, r)
        if best[0] >= 0.62:
            return {"name": E.norm(best[1]), "conf": round(best[0], 2), "via": "roster",
                    "flags": [] if best[0] >= 0.85 else ["check_name"]}
    i = max(range(len(cand)), key=lambda k: len(cand[k][0].replace(" ", "")))
    name, conf = cand[i][0], cand[i][1]
    if len(name.split()) == 1:  # single word: try to join the neighbouring line
        for j in (i + 1, i - 1):
            if 0 <= j < len(cand) and cand[j][1] > 0.5:
                name = (name + " " + cand[j][0]) if j > i else (cand[j][0] + " " + name)
                break
    return {"name": E.norm(name), "conf": round(conf * 0.9, 2), "via": "ocr",
            "flags": ["not_in_list" if roster else "check_name"]}


# ============================================================ passport photos
_casc = None


def _cascades():
    global _casc
    if _casc is None:
        _casc = []
        for n in ("default", "alt2", "alt"):
            c = cv2.CascadeClassifier(cv2.data.haarcascades + f"haarcascade_frontalface_{n}.xml")
            if not c.empty():
                _casc.append(c)
    return _casc


def detect_face(pil):
    """Largest frontal face as (x, y, w, h) in pil coordinates, or None."""
    if cv2 is None:
        return None
    g = np.array(pil.convert("L"))
    s = min(1.0, 900 / max(g.shape))
    if s < 1:
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
    ms = max(30, int(0.07 * min(g.shape)))
    for c in _cascades():
        for mn in (5, 3):
            f = c.detectMultiScale(g, 1.1, mn, minSize=(ms, ms))
            if len(f):
                x, y, w, h = max(f, key=lambda r: r[2] * r[3])
                return (x / s, y / s, w / s, h / s)
    return None


def orient_by_face(img):
    """Try upright first, then sideways / upside-down. Returns (image, face, extra_rotation_cw)."""
    for rot in (0, 90, 270, 180):
        im = img if rot == 0 else img.rotate(-rot, expand=True)
        f = detect_face(im)
        if f:
            return im, f, rot
    return img, None, 0


def make_passport(img, ic, zoom=1.0):
    """Crop head + shoulders + upper chest, passport style, to the configured size."""
    ow, oh = ic["photo_size"]
    im, face, _ = orient_by_face(img)
    W, H = im.size
    flags = []
    if face:
        fx, fy, fw, fh = face
        ch = fh * ic["face_scale"] * zoom
        cw = ch * ow / oh
        if ch > H or cw > W:  # not enough room: shrink the window, keep the aspect
            k = min(H / ch, W / cw)
            ch, cw = ch * k, cw * k
        left = min(max(fx + fw / 2 - cw / 2, 0), W - cw)
        top = min(max(fy - ic["top_ratio"] * ch, 0), H - ch)
    else:
        flags.append("no_face")
        ch = min(H, W * oh / ow)
        cw = ch * ow / oh
        left, top = (W - cw) / 2, (H - ch) * 0.1
    box = (round(left), round(top), round(left + cw), round(top + ch))
    out = im.crop(box).resize((ow, oh), LANCZOS)
    if (box[2] - box[0]) < 0.35 * ow:
        flags.append("low_res")
    return {"img": out, "oriented": im, "face": face, "flags": flags}


# ============================================================ signatures
def text_rotation(pil):
    """Clockwise degrees (0/90/180/270) needed to make a photographed sheet upright.

    Uses the OCR detector's text-box shapes (tall boxes = sideways page) and its angle classifier
    (flipped text = upside down). Falls back to 0 if the OCR internals are unavailable."""
    eng = _engine()
    if eng is None:
        return 0
    try:
        w, h = pil.size
        s = min(1.0, 1400 / max(w, h))
        im = pil.convert("RGB")
        if s < 1:
            im = im.resize((round(w * s), round(h * s)), LANCZOS)
        with _ocr_lock:
            img, _, _ = eng.preprocess(np.array(im)[:, :, ::-1])
            img, _ = eng.maybe_add_letterbox(img, {})
            boxes, _ = eng.auto_text_det(img)
            if boxes is None or len(boxes) == 0:
                return 0
            _, cls, _ = eng.text_cls(eng.get_crop_img_list(img, boxes))
        votes = {0: 0.0, 90: 0.0, 180: 0.0, 270: 0.0}
        for b, (lab, sc) in zip(boxes, cls):
            bw = max(np.linalg.norm(b[0] - b[1]), np.linalg.norm(b[2] - b[3]))
            bh = max(np.linalg.norm(b[0] - b[3]), np.linalg.norm(b[1] - b[2]))
            tall, flipped = bh / max(bw, 1) >= 1.5, str(lab) == "180"
            votes[(90 if flipped else 270) if tall else (180 if flipped else 0)] += bw * bh * float(sc)
        return max(votes, key=votes.get)
    except Exception:
        return 0


def best_text_orientation(img):
    """Rotate a photographed sheet upright. Returns (image, lines)."""
    rot = text_rotation(img)
    im = img.rotate(-rot, expand=True) if rot else img
    return im, ocr_lines(im)


def _fit_canvas(pil, size):
    ow, oh = size
    s = min(ow * 0.96 / pil.width, oh * 0.92 / pil.height)
    pil = pil.resize((max(1, round(pil.width * s)), max(1, round(pil.height * s))), LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(pil, ((ow - pil.width) // 2, (oh - pil.height) // 2))
    return canvas


def make_signature(img, lines, size):
    """Find the signature on a photographed sheet, crop it, whiten the paper, fit to `size`."""
    flags = []
    arr = np.array(img.convert("RGB"))
    H, W = arr.shape[:2]
    sc = min(1.0, 1400 / max(H, W))
    small = cv2.resize(arr, (round(W * sc), round(H * sc)), interpolation=cv2.INTER_AREA) if sc < 1 else arr
    sh, sw = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    # paper brightness map (removes shadows / grey paper): big morphological closing on a small copy
    q = cv2.resize(gray, (max(8, sw // 4), max(8, sh // 4)), interpolation=cv2.INTER_AREA)
    k = max(5, (min(q.shape) // 10) | 1)
    bgq = cv2.morphologyEx(q, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bgq = cv2.GaussianBlur(bgq, (0, 0), k / 2)
    bg = np.maximum(cv2.resize(bgq, (sw, sh), interpolation=cv2.INTER_LINEAR), 1)
    norm = np.clip(gray.astype(np.float32) / bg * 255, 0, 255).astype(np.uint8)
    ink = (norm < 185).astype(np.uint8) * 255

    # keep only ink inside the sheet of paper (ignores table, fingers and the paper's own edge)
    blur = cv2.GaussianBlur(gray, (0, 0), max(3, sw / 150))
    pm = (blur > 0.45 * np.percentile(blur, 90)).astype(np.uint8)
    n0, lab0, st0, _ = cv2.connectedComponentsWithStats(pm, connectivity=4)
    if n0 > 1:
        pm = (lab0 == 1 + int(np.argmax(st0[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    er = max(3, int(0.025 * max(sw, sh)))
    pm = cv2.erode(pm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (er, er)))
    ink[pm == 0] = 0

    # blank out the written name (confident OCR text) so only the signature remains
    pad = int(0.01 * max(sw, sh))
    for l in lines:
        if l["conf"] >= 0.55 and len(clean_text(l["text"]).replace(" ", "")) >= 4:
            x0, y0, x1, y1 = [v * sc for v in l["box"]]
            ink[max(0, int(y0 - pad)):int(y1 + pad), max(0, int(x0 - pad)):int(x1 + pad)] = 0
    # remove ruled lines of exercise paper
    hl = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, sw // 4), 1)))
    ink = cv2.subtract(ink, cv2.dilate(hl, np.ones((3, 1), np.uint8)))
    # drop specks and solid blobs (shadows, fingers)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    ok = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        ok[i] = a >= max(6, 0.00003 * sw * sh) and not (a / (w * h) > 0.6 and a > 0.01 * sw * sh)
    keep = (ok[lab] * 255).astype(np.uint8)

    # merge strokes into clusters, pick the one with the most ink
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(9, int(0.05 * sw)), max(5, int(0.025 * sh))))
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(cv2.dilate(keep, kern), connectivity=8)
    counts = np.bincount(lab2[keep > 0], minlength=n2)
    best, best_ink = None, 0
    for i in range(1, n2):
        x, y, w, h, _a = st2[i]
        if w > 0.9 * sw and h > 0.9 * sh:
            continue
        if counts[i] > best_ink:
            best, best_ink = i, counts[i]
    if best is None or best_ink == 0:
        flags.append("no_signature")
        return _fit_canvas(img.convert("RGB"), size), flags

    ys, xs = np.where((lab2 == best) & (keep > 0))
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    p = int(0.08 * max(x1 - x0, y1 - y0)) + 4
    sx0, sy0, sx1, sy1 = max(0, x0 - p), max(0, y0 - p), min(sw, x1 + p), min(sh, y1 + p)
    X0, Y0, X1, Y1 = int(sx0 / sc), int(sy0 / sc), int(sx1 / sc), int(sy1 / sc)
    crop = arr[Y0:Y1, X0:X1].astype(np.float32)
    bgc = cv2.resize(bg[sy0:sy1, sx0:sx1], (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)
    out = np.clip(crop / bgc[..., None] * 255, 0, 255)
    out = np.clip((out - 30) * 255 / (228 - 30), 0, 255).astype(np.uint8)
    # white out everything except the strokes (+ a thin halo): removes ruled lines, smudges, leftover text
    r = max(5, int(0.012 * max(sw, sh)))
    m = cv2.dilate(keep[sy0:sy1, sx0:sx1], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r)))
    m = cv2.resize(m, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR) > 0
    out[~m] = 255
    return _fit_canvas(Image.fromarray(out), size), flags


# ============================================================ storage
def cdir(cls):
    return E.base_dir() / "_intake" / cls


def _rp(cls):
    return cdir(cls) / "records.json"


def load(cls):
    p = _rp(cls)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"roster": [], "items": []}


def save(cls, data):
    cdir(cls).mkdir(parents=True, exist_ok=True)
    tmp = _rp(cls).with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(_rp(cls))


def _update(cls, iid, **kw):
    with LOCK:
        d = load(cls)
        for it in d["items"]:
            if it["id"] == iid:
                it.update(kw)
        save(cls, d)


def filename_hint(fn):
    stem = Path(fn).stem.replace("_", " ")
    if re.fullmatch(r"[A-Za-z'\- ]+", stem) and len(stem.split()) >= 2:
        return E.norm(stem)
    return ""


# ============================================================ background worker
Q, INFLIGHT, _w, _wl = queue.Queue(), set(), None, threading.Lock()


def enqueue(cls, iid):
    global _w
    with _wl:
        if (cls, iid) in INFLIGHT:
            return
        INFLIGHT.add((cls, iid))
        Q.put((cls, iid))
        if _w is None or not _w.is_alive():
            _w = threading.Thread(target=_loop, daemon=True)
            _w.start()


def _loop():
    while True:
        cls, iid = Q.get()
        again = False
        try:
            again = process_item(cls, iid)
        except Exception as e:
            _update(cls, iid, status="error", flags=["error"], error=str(e))
        finally:
            with _wl:
                INFLIGHT.discard((cls, iid))
        if again:
            enqueue(cls, iid)


def _name_result(cand, roster):
    ex = name_from_candidates(cand, roster)
    ex["cands"], ex["raw"] = cand, " | ".join(c for c, _ in cand)
    return ex


def process_item(cls, iid):
    """Returns True if the item was changed while processing and needs another pass."""
    ic = E.load_config()["intake"]
    with LOCK:
        d = load(cls)
        item = next((i for i in d["items"] if i["id"] == iid), None)
        if not item:
            return False
        item["status"] = "processing"
        save(cls, d)
        roster, rev = d["roster"], item.get("rev", 0)
    img = ImageOps.exif_transpose(Image.open(cdir(cls) / item["raw"])).convert("RGB")
    if max(img.size) > 5000:
        img.thumbnail((5000, 5000), LANCZOS)
    if item["rot"]:
        img = img.rotate(-item["rot"], expand=True)

    flags, ex = [], None
    if item["kind"] == "photo":
        res = make_passport(img, ic, item["zoom"])
        out, flags = res["img"], res["flags"]
        if item["want_name"]:
            min_y = None
            if res["face"]:
                min_y = res["face"][1] + res["face"][3] * 1.5  # only text below the chin (the held paper)
            ex = _name_result(candidates_from_lines(ocr_lines(res["oriented"]), min_y), roster)
    else:
        if cv2 is None:
            raise RuntimeError("opencv-python-headless is not installed")
        oriented, lines = best_text_orientation(img)
        out, flags = make_signature(oriented, lines, ic["sign_size"])
        if item["want_name"]:
            ex = _name_result(candidates_from_lines(lines), roster)
    outp = cdir(cls) / "out" / f"{iid}.jpg"
    outp.parent.mkdir(parents=True, exist_ok=True)
    out.save(outp, "JPEG", quality=98, dpi=(300, 300))

    upd = {"flags": flags, "error": ""}
    if ex is not None:
        upd["ocr"], upd["cands"] = ex["raw"], ex["cands"]
        name = ex["name"]
        nflags = ex["flags"]
        if not name and item.get("hint"):
            name, nflags = item["hint"], ["from_filename"]
        upd.update(name=name, conf=ex["conf"], nflags=nflags, want_name=False, auto_name=True)
    with LOCK:
        d = load(cls)
        cur = next((i for i in d["items"] if i["id"] == iid), None)
        if not cur:
            outp.unlink(missing_ok=True)
            return False
        cur.update(upd)
        changed = cur.get("rev", 0) != rev
        cur["status"] = "queued" if changed else "done"
        cur["v"] = int(time.time() * 1000)
        save(cls, d)
    return changed


# ============================================================ public API (used by the web routes)
REVIEW = {"no_name", "check_name", "not_in_list", "no_face", "no_signature", "error"}


def add_raw(cls, kind, orig_name, writer, name=""):
    ext = Path(orig_name).suffix.lower()
    if ext not in E.IMG_EXT or kind not in ("photo", "sign"):
        return None
    iid = uuid.uuid4().hex[:10]
    rel = f"raw_{kind}/{iid}{ext}"
    dst = cdir(cls) / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer(dst)
    name = E.norm(name)
    item = {"id": iid, "kind": kind, "raw": rel, "orig": orig_name, "name": name, "ocr": "", "cands": [],
            "conf": 0, "flags": [], "nflags": [], "rot": 0, "zoom": 1.0, "rev": 0, "error": "",
            "want_name": not name, "auto_name": False, "hint": filename_hint(orig_name),
            "status": "queued", "v": int(time.time() * 1000)}
    with LOCK:
        d = load(cls)
        d["items"].append(item)
        save(cls, d)
    enqueue(cls, iid)
    return iid


def build_rows(items):
    groups, rows = {}, []
    for it in items:
        k = E.norm(it["name"])
        if not k:
            rows.append({"name": "", "photo": it if it["kind"] == "photo" else None,
                         "sign": it if it["kind"] == "sign" else None, "dup": False})
        else:
            groups.setdefault(k, {"photo": [], "sign": []})[it["kind"]].append(it)
    for k, g in groups.items():
        n = max(len(g["photo"]), len(g["sign"]))
        for i in range(n):
            rows.append({"name": k, "dup": n > 1,
                         "photo": g["photo"][i] if i < len(g["photo"]) else None,
                         "sign": g["sign"][i] if i < len(g["sign"]) else None})
    rows.sort(key=lambda r: (r["name"] != "", r["name"]))
    for r in rows:
        fl = []
        for it in (r["photo"], r["sign"]):
            if it:
                fl += it.get("flags", []) + it.get("nflags", [])
        if r["dup"]:
            fl.append("duplicate")
        if not r["name"]:
            fl.append("no_name")
        if r["photo"] is None:
            fl.append("missing_photo")
        if r["sign"] is None:
            fl.append("missing_sign")
        r["flags"] = sorted(set(fl))
        r["review"] = bool(set(r["flags"]) & (REVIEW | {"duplicate"}))
        r["ids"] = [it["id"] for it in (r["photo"], r["sign"]) if it]
    return rows


def state(cls):
    with LOCK:
        d = load(cls)
    for it in d["items"]:
        if it["status"] in ("queued", "processing"):
            enqueue(cls, it["id"])  # resumes anything left over from a restart; no-op if already queued
    pend = sum(1 for i in d["items"] if i["status"] in ("queued", "processing"))
    rows = build_rows(d["items"])
    return {"rows": rows, "pending": pend, "total": len(d["items"]), "roster": "\n".join(d["roster"]),
            "photos": sum(i["kind"] == "photo" for i in d["items"]),
            "signs": sum(i["kind"] == "sign" for i in d["items"]),
            "review": sum(r["review"] for r in rows)}


def rename(cls, ids, name):
    name = E.norm(name)
    with LOCK:
        d = load(cls)
        for it in d["items"]:
            if it["id"] in ids:
                it.update(name=name, nflags=[], auto_name=False, want_name=False)
        save(cls, d)


def adjust(cls, iid, rot=0, zoom=1.0):
    with LOCK:
        d = load(cls)
        for it in d["items"]:
            if it["id"] == iid:
                it["rot"] = (it["rot"] + rot) % 360
                it["zoom"] = round(min(2.0, max(0.5, it["zoom"] * zoom)), 3)
                it["rev"] += 1
                it["status"] = "queued"
        save(cls, d)
    enqueue(cls, iid)


def replace_raw(cls, kind, orig_name, writer, iid=None, name=""):
    """Swap in a new picture for an existing item (keeps its name), or create one for `name`."""
    if not iid:
        return add_raw(cls, kind, orig_name, writer, name=name)
    ext = Path(orig_name).suffix.lower()
    if ext not in E.IMG_EXT:
        return None
    with LOCK:
        d = load(cls)
        it = next((i for i in d["items"] if i["id"] == iid), None)
        if not it:
            return None
        (cdir(cls) / it["raw"]).unlink(missing_ok=True)
        rel = f"raw_{it['kind']}/{iid}{ext}"
        writer(cdir(cls) / rel)
        it.update(raw=rel, orig=orig_name, rot=0, zoom=1.0, rev=it["rev"] + 1, status="queued",
                  want_name=not it["name"])
        save(cls, d)
    enqueue(cls, iid)
    return iid


def delete(cls, ids):
    with LOCK:
        d = load(cls)
        for it in d["items"]:
            if it["id"] in ids:
                (cdir(cls) / it["raw"]).unlink(missing_ok=True)
                (cdir(cls) / "out" / f"{it['id']}.jpg").unlink(missing_ok=True)
        d["items"] = [i for i in d["items"] if i["id"] not in ids]
        save(cls, d)


def set_roster(cls, text, rematch=False):
    names = [E.norm(x) for x in text.splitlines() if E.norm(x)]
    with LOCK:
        d = load(cls)
        d["roster"] = names
        n = 0
        if rematch:
            for it in d["items"]:
                if it.get("auto_name") and it.get("cands"):
                    ex = name_from_candidates(it["cands"], names)
                    it.update(name=ex["name"], conf=ex["conf"], nflags=ex["flags"])
                    n += 1
        save(cls, d)
    return n


def recrop_all(cls):
    with LOCK:
        d = load(cls)
        for it in d["items"]:
            it["rev"] += 1
            it["status"] = "queued"
            it["want_name"] = False
        save(cls, d)
        ids = [i["id"] for i in d["items"]]
    for i in ids:
        enqueue(cls, i)
    return len(ids)


def safe_name(n):
    return re.sub(r'[\\/:*?"<>|]', "", E.norm(n)).strip()


def commit(cls, overwrite=False):
    """Copy named results into the project: <base>/<cls>/NAME.jpg and <base>/SIGNS/NAME.jpg."""
    base = E.base_dir()
    d = load(cls)
    rows = build_rows([i for i in d["items"] if i["status"] == "done"])
    written, skipped, conflicts = {"photo": 0, "sign": 0}, [], []
    for r in rows:
        if not r["name"]:
            skipped.append("(no name) item skipped")
            continue
        if r["dup"]:
            skipped.append(f"{r['name']}: duplicate name - fix before saving")
            continue
        fn = safe_name(r["name"]) + ".jpg"
        for kind, dest in (("photo", base / cls), ("sign", base / "SIGNS")):
            it = r[kind]
            if not it:
                continue
            dst = dest / fn
            if dst.exists() and not overwrite:
                conflicts.append(f"{dest.name}/{fn}")
                continue
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cdir(cls) / "out" / f"{it['id']}.jpg", dst)
            written[kind] += 1
    return {"written": written, "skipped": skipped, "conflicts": conflicts}
