"""Igara ID Studio - processing engine (no web code in here, so it can be reused by Electron/CLI)."""
import difflib
import io
import json
import os
import re
import shutil
import tempfile
import threading
import uuid
import zipfile
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("IDSTUDIO_CONFIG") or APP_DIR / "config.json")
LOCKED_BASE = os.environ.get("IDSTUDIO_BASE")  # set on a server: the data folder cannot be changed from the browser
IMG_EXT = {".jpg", ".jpeg", ".png"}
SIGN_W, SIGN_H = 360, 135

DEFAULT_CONFIG = {
    "base": "",
    "issue_date": "01/07/2026",
    "font_pt": 7.8,
    "dpi": 300,
    "font_path": "",
    "classes": {
        "S.1": {"id_prefix": "26/ISS-26", "class_text": "S.1 - S.4", "valid": "31/12/2029", "start": 1},
        "S.2": {"id_prefix": "26/ISS-25", "class_text": "S.1 - S.4", "valid": "31/12/2028", "start": 1},
        "S.3": {"id_prefix": "26/ISS-24", "class_text": "S.1 - S.4", "valid": "31/12/2027", "start": 1},
        "S.4": {"id_prefix": "26/ISS-23", "class_text": "S.1 - S.4", "valid": "31/12/2026", "start": 1},
        "S.5": {"id_prefix": "26/ISS-22", "class_text": "S.5 - S.6", "valid": "31/12/2027", "start": 1},
        "S.6": {"id_prefix": "26/ISS-21", "class_text": "S.6", "valid": "31/12/2026", "start": 1},
    },
    "intake": {
        "photo_size": [2605, 3120],   # passport output size (W, H)
        "sign_size": [360, 135],      # signature output size (W, H)
        "face_scale": 2.1,            # crop height = face height x this  (smaller = tighter)
        "top_ratio": 0.215,           # headroom above the detected face, as a share of crop height
    },
    # Pixel positions on the 1011 x 638 template
    "layout": {
        "photo_box": [29, 195, 344, 510],
        "frame_box": [27, 193, 345, 511],
        "frame_w": 3,
        "face_bias": 0.20,
        "sign_box": [752, 518, 950, 598],
        "text_pos": {"name": [510, 293], "id": [518, 349], "class": [518, 408],
                     "issue": [518, 467], "valid": [838, 467]},
        "max_right": {"name": 1000, "id": 1000, "class": 1000, "issue": 706, "valid": 1005},
        "text_color": [16, 38, 66],
    },
}


# ------------------------------------------------------------------ config
def load_config():
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if CONFIG_PATH.exists():
        try:
            saved = json.loads(CONFIG_PATH.read_text())
            for k, v in saved.items():
                if k in ("layout", "intake") and isinstance(v, dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v
        except Exception:
            pass
    if LOCKED_BASE:
        cfg["base"] = LOCKED_BASE
    if not cfg["base"]:
        guess = Path.home() / "Desktop" / "gara id"
        cfg["base"] = str(guess) if guess.exists() else str(Path.home())
    return cfg


def save_config(cfg):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def base_dir(cfg=None):
    return Path((cfg or load_config())["base"]).expanduser()


def allowed_folders(cfg):
    return list(cfg["classes"]) + ["SIGNS"]


# ------------------------------------------------------------------ helpers
def norm(name):
    return re.sub(r"\s+", " ", name).strip().upper()


def list_images(folder: Path):
    if not folder.is_dir():
        return []
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT),
                  key=lambda p: norm(p.stem))


def template_path(base: Path):
    for n in ("empty template.png", "empty_template.png"):
        p = base / "new" / n
        if p.exists():
            return p
    return None


def find_font(user_path=""):
    if user_path and Path(user_path).exists():
        return user_path
    roots = [APP_DIR, APP_DIR / "fonts", Path.home() / ".fonts", Path.home() / ".local/share/fonts",
             Path("/usr/share/fonts"), Path("/usr/local/share/fonts")]
    found = []
    for r in roots:
        if r.exists():
            found += [p for p in r.rglob("*") if p.suffix.lower() in (".ttf", ".otf")
                      and "montserrat" in p.name.lower()
                      and not any(x in p.name.lower() for x in ("italic", "alternates", "subrayada"))]
    if not found:
        return None

    def rank(p):
        n = p.name.lower()
        for i, key in enumerate(("semibold", "bold", "medium", "regular")):
            if key in n:
                return i
        return 4
    found.sort(key=rank)
    return str(found[0])


@lru_cache(maxsize=256)
def load_font(path, size, weight=600):
    if path:
        f = ImageFont.truetype(path, size)
        try:
            f.set_variation_by_axes([weight])
        except Exception:
            pass
        return f
    for fb in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf"):
        try:
            return ImageFont.truetype(fb, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ------------------------------------------------------------------ prepare: clean / rotate / resize
def clean_names(folder: Path, apply: bool):
    """Collapse runs of spaces in file names ('A   B.jpg' -> 'A B.jpg'). Nothing else is changed."""
    out = []
    for p in sorted(folder.iterdir()) if folder.is_dir() else []:
        if not p.is_file() or p.suffix.lower() not in IMG_EXT:
            continue
        new = re.sub(r" {2,}", " ", p.stem).strip() + p.suffix
        if new == p.name:
            continue
        target = p.with_name(new)
        if target.exists():
            out.append({"old": p.name, "new": new, "status": "skipped (target exists)"})
            continue
        if apply:
            p.rename(target)
        out.append({"old": p.name, "new": new, "status": "renamed" if apply else "will rename"})
    return out


def backup(base: Path, folder: str, path: Path):
    dst = base / "_originals" / folder / path.name
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)


def _save_like(img, path: Path, dpi=None):
    if path.suffix.lower() in (".jpg", ".jpeg"):
        kw = {"dpi": (dpi, dpi)} if dpi else {}
        img.convert("RGB").save(path, "JPEG", quality=95, **kw)
    else:
        img.save(path)


def rotate_file(base: Path, folder: str, path: Path, degrees: int):
    """degrees: 90 = left (counter-clockwise), -90 = right (clockwise). Original is backed up first."""
    backup(base, folder, path)
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        out = img.rotate(degrees, expand=True)
    _save_like(out, path)


def resize_signature(base: Path, folder: str, path: Path):
    """Fit inside 360x135 keeping proportions, centred on white. Original is backed up first."""
    backup(base, folder, path)
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((SIGN_W, SIGN_H), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (SIGN_W, SIGN_H), "white")
        canvas.paste(img, ((SIGN_W - img.width) // 2, (SIGN_H - img.height) // 2))
    _save_like(canvas, path, dpi=300)


def restore_originals(base: Path, folder: str):
    src = base / "_originals" / folder
    n = 0
    if src.is_dir():
        for p in src.iterdir():
            shutil.copy2(p, base / folder / p.name)
            n += 1
    return n


def thumbnail_bytes(path: Path, size=(260, 260)):
    with Image.open(path) as img:
        img.draft("RGB", (size[0] * 2, size[1] * 2))
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail(size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=80)
    return buf.getvalue()


# ------------------------------------------------------------------ ID rendering
def sign_index(base: Path):
    return {norm(p.stem): p for p in list_images(base / "SIGNS")}


def find_sign(idx, name):
    if name in idx:
        return idx[name], False
    m = difflib.get_close_matches(name, idx.keys(), n=1, cutoff=0.93)
    return (idx[m[0]], True) if m else (None, False)


def cover_crop(img, size, y_bias):
    tw, th = size
    w, h = img.size
    s = max(tw / w, th / h)
    img = img.resize((max(tw, round(w * s)), max(th, round(h * s))), Image.LANCZOS)
    left = (img.width - tw) // 2
    top = int((img.height - th) * y_bias)
    return img.crop((left, top, left + tw, top + th))


def _paste_signature(card, sig_path, box):
    l, t, r, b = box
    bw, bh = r - l, b - t
    sig = Image.open(sig_path).convert("RGB")
    s = min(bw / sig.width, bh / sig.height)
    sig = sig.resize((round(sig.width * s), round(sig.height * s)), Image.LANCZOS)
    canvas = Image.new("RGB", (bw, bh), "white")
    canvas.paste(sig, ((bw - sig.width) // 2, (bh - sig.height) // 2))
    card.paste(ImageChops.multiply(card.crop(tuple(box)), canvas), (l, t))


def _draw_text(draw, cfg, font_path, key, text):
    lay = cfg["layout"]
    x, y = lay["text_pos"][key]
    size = round(cfg["font_pt"] * cfg["dpi"] / 72)
    font = load_font(font_path, size)
    mr = lay["max_right"].get(key)
    while mr and draw.textlength(text, font=font) > (mr - x) and size > 18:
        size -= 1
        font = load_font(font_path, size)
    draw.text((x, y), text, font=font, fill=tuple(lay["text_color"]), anchor="ls")


def render_card(template: Image.Image, cfg, font_path, photo_path, sign_path, name, id_no, ccfg):
    lay = cfg["layout"]
    card = template.copy()
    pb = lay["photo_box"]
    pw, ph = pb[2] - pb[0], pb[3] - pb[1]
    with Image.open(photo_path) as ph_img:
        ph_img.draft("RGB", (pw * 2, ph * 2))
        ph_img = ImageOps.exif_transpose(ph_img).convert("RGB")
        card.paste(cover_crop(ph_img, (pw, ph), lay["face_bias"]), (pb[0], pb[1]))
    if sign_path:
        _paste_signature(card, sign_path, lay["sign_box"])
    d = ImageDraw.Draw(card)
    d.rectangle(tuple(lay["frame_box"]), outline="black", width=lay["frame_w"])
    _draw_text(d, cfg, font_path, "name", name)
    _draw_text(d, cfg, font_path, "id", id_no)
    _draw_text(d, cfg, font_path, "class", ccfg["class_text"])
    _draw_text(d, cfg, font_path, "issue", cfg["issue_date"])
    _draw_text(d, cfg, font_path, "valid", ccfg["valid"])
    return card


def id_number(ccfg, index):  # index is 0-based position in the sorted list
    return f"{ccfg['id_prefix']}/{index + int(ccfg.get('start', 1)):03d}"


def render_preview(cls, name=None):
    cfg = load_config()
    base = base_dir(cfg)
    tp = template_path(base)
    if not tp:
        raise FileNotFoundError("Template 'empty template.png' not found in <base>/new/")
    photos = list_images(base / cls)
    if not photos:
        raise FileNotFoundError(f"No photos in {cls}")
    names = [norm(p.stem) for p in photos]
    i = names.index(norm(name)) if name and norm(name) in names else 0
    idx = sign_index(base)
    sp, _ = find_sign(idx, names[i])
    card = render_card(Image.open(tp).convert("RGB"), cfg, find_font(cfg["font_path"]),
                       photos[i], sp, names[i], id_number(cfg["classes"][cls], i), cfg["classes"][cls])
    buf = io.BytesIO()
    card.save(buf, "PNG")
    return buf.getvalue()


# ------------------------------------------------------------------ print sheets (A4, 2 x 5 cards per page)
def make_print_pdf(card_paths, out_pdf: Path):
    PW, PH, CW, CH, GAP = 2480, 3508, 1011, 638, 36
    cols, rows = 2, 5
    mx = (PW - cols * CW - (cols - 1) * GAP) // 2
    my = (PH - rows * CH - (rows - 1) * GAP) // 2
    tmp = Path(tempfile.mkdtemp())
    pages = []
    for start in range(0, len(card_paths), cols * rows):
        sheet = Image.new("RGB", (PW, PH), "white")
        for k, cp in enumerate(card_paths[start:start + cols * rows]):
            r, c = divmod(k, cols)
            x, y = mx + c * (CW + GAP), my + r * (CH + GAP)
            with Image.open(cp) as im:
                sheet.paste(im.convert("RGB"), (x, y))
            ImageDraw.Draw(sheet).rectangle((x - 1, y - 1, x + CW, y + CH), outline=(190, 190, 190), width=1)
        p = tmp / f"p{start:05d}.jpg"
        sheet.save(p, "JPEG", quality=95, dpi=(300, 300))
        pages.append(p)
    if pages:
        first = Image.open(pages[0])
        first.save(out_pdf, "PDF", save_all=True, append_images=[Image.open(p) for p in pages[1:]], resolution=300)
    shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ generation jobs
JOBS = {}


def start_generation(classes, pdf=False, limit=None):
    job = {"id": uuid.uuid4().hex[:8], "status": "running", "done": 0, "total": 0,
           "current": "", "problems": [], "outputs": {}}
    JOBS[job["id"]] = job
    threading.Thread(target=_run, args=(job, classes, pdf, limit), daemon=True).start()
    return job


def _run(job, classes, pdf, limit):
    try:
        cfg = load_config()
        base = base_dir(cfg)
        tp = template_path(base)
        if not tp:
            raise FileNotFoundError("Template 'empty template.png' not found in <base>/new/")
        template = Image.open(tp).convert("RGB")
        font_path = find_font(cfg["font_path"])
        idx = sign_index(base)
        plan = {}
        for cls in classes:
            photos = list_images(base / cls)
            plan[cls] = photos[:limit] if limit else photos
        job["total"] = sum(len(v) for v in plan.values())
        for cls, photos in plan.items():
            ccfg = cfg["classes"][cls]
            out_dir = base / "new" / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            made, seen = [], set()
            for i, pp in enumerate(photos):
                name = norm(pp.stem)
                job["current"] = f"{cls} / {name}"
                try:
                    if name in seen:
                        job["problems"].append(f"{cls}/{name}: duplicate name - earlier ID overwritten")
                    seen.add(name)
                    sp, fuzzy = find_sign(idx, name)
                    if not sp:
                        job["problems"].append(f"{cls}/{name}: NO signature found")
                    elif fuzzy:
                        job["problems"].append(f"{cls}/{name}: signature matched approximately -> {sp.name}")
                    card = render_card(template, cfg, font_path, pp, sp, name, id_number(ccfg, i), ccfg)
                    dst = out_dir / f"{name}.png"
                    card.save(dst, dpi=(cfg["dpi"], cfg["dpi"]))
                    made.append(dst)
                except Exception as e:
                    job["problems"].append(f"{cls}/{name}: FAILED ({e})")
                job["done"] += 1
            if pdf and made:
                job["current"] = f"{cls}: building print PDF"
                make_print_pdf(made, out_dir / f"{cls}_print_sheet.pdf")
            job["outputs"][cls] = len(made)
        job["status"] = "done"
    except Exception as e:
        job["status"] = "error"
        job["problems"].append(str(e))


def zip_class(cls):
    cfg = load_config()
    d = base_dir(cfg) / "new" / cls
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in (".png", ".pdf"):
                z.write(p, p.name)
    buf.seek(0)
    return buf


# ------------------------------------------------------------------ status scan
def scan():
    cfg = load_config()
    base = base_dir(cfg)
    idx = sign_index(base)
    folders = []
    for f in allowed_folders(cfg):
        imgs = list_images(base / f)
        row = {"name": f, "exists": (base / f).is_dir(), "count": len(imgs), "missing_signs": [],
               "generated": len(list((base / "new" / f).glob("*.png"))) if (base / "new" / f).is_dir() else 0,
               "has_backup": (base / "_originals" / f).is_dir()}
        if f != "SIGNS":
            row["missing_signs"] = [norm(p.stem) for p in imgs if not find_sign(idx, norm(p.stem))[0]]
        folders.append(row)
    return {"base": str(base), "base_exists": base.is_dir(),
            "template": str(template_path(base)) if template_path(base) else None,
            "font": find_font(cfg["font_path"]), "folders": folders}
