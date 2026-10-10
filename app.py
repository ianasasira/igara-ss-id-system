#!/usr/bin/env python3
"""Igara ID Studio - local web app.  Run:  python3 app.py   then open http://127.0.0.1:5000"""
import argparse
import hmac
import os
import sys
import threading
import webbrowser
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory

import engine as E
from intake_routes import bp as intake_bp

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.register_blueprint(intake_bp)
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024  # one upload request (the UI sends 4 photos at a time)

AUTH_USER, AUTH_PASS = os.environ.get("IDSTUDIO_USER", "admin"), os.environ.get("IDSTUDIO_PASSWORD")


@app.before_request
def _login():
    """HTTP Basic login. Enabled whenever IDSTUDIO_PASSWORD is set (always set it on a server)."""
    if not AUTH_PASS:
        return None
    a = request.authorization
    if a and hmac.compare_digest(a.username or "", AUTH_USER) and hmac.compare_digest(a.password or "", AUTH_PASS):
        return None
    return Response("Login required", 401, {"WWW-Authenticate": 'Basic realm="Igara ID Studio"'})


@app.errorhandler(Exception)
def on_error(e):
    code = getattr(e, "code", 500)
    return jsonify(error=getattr(e, "description", None) or str(e)), code if isinstance(code, int) else 500


def _folder(name):
    cfg = E.load_config()
    if name not in E.allowed_folders(cfg):
        abort(400, "Unknown folder")
    return E.base_dir(cfg), name, E.base_dir(cfg) / name


def _file(folder, fname):
    base, name, fp = _folder(folder)
    p = fp / Path(fname).name  # basename only: blocks path traversal
    if not p.is_file():
        abort(404, "File not found")
    return base, name, p


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/config")
def get_config():
    return jsonify(E.load_config())


@app.post("/api/config")
def set_config():
    cfg = E.load_config()
    data = request.get_json(force=True)
    if E.LOCKED_BASE:
        data.pop("base", None)
    for k in ("base", "issue_date", "font_pt", "dpi", "font_path", "classes", "layout"):
        if k in data:
            cfg[k] = data[k]
    if "intake" in data:
        cfg["intake"].update(data["intake"])
    cfg["font_pt"], cfg["dpi"] = float(cfg["font_pt"]), int(cfg["dpi"])
    E.save_config(cfg)
    return jsonify(ok=True)


@app.post("/api/template")
def upload_template():
    from PIL import Image
    f = request.files["file"]
    im = Image.open(f.stream).convert("RGB")
    dst = E.base_dir() / "new" / "empty template.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, "PNG")
    return jsonify(width=im.width, height=im.height, ok=(im.size == (1011, 638)))


@app.get("/api/scan")
def scan():
    return jsonify(E.scan())


@app.post("/api/clean")
def clean():
    d = request.get_json(force=True)
    res = {}
    for f in d["folders"]:
        _, _, fp = _folder(f)
        res[f] = E.clean_names(fp, bool(d.get("apply")))
    return jsonify(res)


@app.get("/api/images")
def images():
    base, name, fp = _folder(request.args["folder"])
    return jsonify([{"name": p.name, "v": int(p.stat().st_mtime)} for p in E.list_images(fp)])


@app.get("/api/thumb")
def thumb():
    _, _, p = _file(request.args["folder"], request.args["name"])
    return Response(E.thumbnail_bytes(p), mimetype="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@app.post("/api/rotate")
def rotate():
    d = request.get_json(force=True)
    deg = int(d["degrees"])
    if deg not in (90, -90, 180):
        abort(400, "degrees must be 90, -90 or 180")
    base, name, fp = _folder(d["folder"])
    targets = [fp / Path(n).name for n in d["names"]] if d.get("names") else E.list_images(fp)
    n = 0
    for p in targets:
        if p.is_file():
            E.rotate_file(base, name, p, deg)
            n += 1
    return jsonify(rotated=n)


@app.post("/api/resize-signs")
def resize_signs():
    base, name, fp = _folder("SIGNS")
    n = 0
    for p in E.list_images(fp):
        E.resize_signature(base, name, p)
        n += 1
    return jsonify(resized=n)


@app.post("/api/restore")
def restore():
    base, name, _ = _folder(request.get_json(force=True)["folder"])
    return jsonify(restored=E.restore_originals(base, name))


@app.get("/api/preview")
def preview():
    cls = request.args["cls"]
    _folder(cls)
    return Response(E.render_preview(cls, request.args.get("name")), mimetype="image/png",
                    headers={"Cache-Control": "no-store"})


@app.post("/api/generate")
def generate():
    d = request.get_json(force=True)
    cfg = E.load_config()
    classes = [c for c in d["classes"] if c in cfg["classes"]]
    return jsonify(E.start_generation(classes, bool(d.get("pdf")), int(d["limit"]) if d.get("limit") else None))


@app.get("/api/job/<jid>")
def job(jid):
    j = E.JOBS.get(jid)
    if not j:
        abort(404, "Unknown job")
    return jsonify(j)


@app.get("/api/zip")
def zip_it():
    cls = request.args["cls"]
    base, _, _ = _folder(cls)
    if not (base / "new" / cls).is_dir():
        abort(404, "Nothing generated for this class yet")
    return send_file(E.zip_class(cls), mimetype="application/zip", as_attachment=True,
                     download_name=f"{cls}_IDs.zip")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")  # keep local: this app edits files on your disk
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    if a.host not in ("127.0.0.1", "localhost") and not AUTH_PASS:
        sys.exit("Refusing to listen on a network address without a login. Set IDSTUDIO_PASSWORD first.")
    if not a.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{a.host}:{a.port}")).start()
    app.run(host=a.host, port=a.port, threaded=True)
