"""Web routes for the Intake workflow (registered as a Flask blueprint by app.py)."""
import shutil
from pathlib import Path

from flask import Blueprint, Response, abort, jsonify, request, send_file

import engine as E
import intake as I

bp = Blueprint("intake", __name__, url_prefix="/api/intake")


def _cls(v):
    if v not in E.load_config()["classes"]:
        abort(400, "Unknown class")
    return v


def _kind(v):
    if v not in ("photo", "sign"):
        abort(400, "kind must be photo or sign")
    return v


@bp.get("/caps")
def caps():
    return jsonify(I.capabilities())


@bp.get("/state")
def state():
    return jsonify(I.state(_cls(request.args["cls"])))


@bp.post("/upload")
def upload():
    cls, kind = _cls(request.form["cls"]), _kind(request.form["kind"])
    n = 0
    for f in request.files.getlist("files"):
        if I.add_raw(cls, kind, f.filename or "x.jpg", lambda dst, f=f: f.save(dst)):
            n += 1
    return jsonify(added=n)


@bp.post("/import-folder")
def import_folder():
    d = request.get_json(force=True)
    cls, kind = _cls(d["cls"]), _kind(d["kind"])
    src = Path(d["path"]).expanduser()
    if not src.is_dir():
        abort(400, f"Folder not found: {src}")
    if E.LOCKED_BASE and E.base_dir().resolve() not in src.resolve().parents and src.resolve() != E.base_dir().resolve():
        abort(400, "On the server, import folders must be inside the data folder")
    n = 0
    for p in E.list_images(src):
        if I.add_raw(cls, kind, p.name, lambda dst, p=p: shutil.copy2(p, dst)):
            n += 1
    return jsonify(added=n)


@bp.post("/rename")
def rename():
    d = request.get_json(force=True)
    I.rename(_cls(d["cls"]), set(d["ids"]), d["name"])
    return jsonify(ok=True)


@bp.post("/rename-many")
def rename_many():
    d = request.get_json(force=True)
    cls = _cls(d["cls"])
    for ch in d["changes"]:
        I.rename(cls, set(ch["ids"]), ch["name"])
    return jsonify(ok=True, changed=len(d["changes"]))


@bp.post("/adjust")
def adjust():
    d = request.get_json(force=True)
    I.adjust(_cls(d["cls"]), d["id"], int(d.get("rot", 0)), float(d.get("zoom", 1.0)))
    return jsonify(ok=True)


@bp.post("/replace")
def replace():
    cls, kind = _cls(request.form["cls"]), _kind(request.form["kind"])
    f = request.files["file"]
    iid = I.replace_raw(cls, kind, f.filename or "x.jpg", lambda dst: f.save(dst),
                        iid=request.form.get("id") or None, name=request.form.get("name", ""))
    if not iid:
        abort(400, "Could not use that file")
    return jsonify(id=iid)


@bp.post("/delete")
def delete():
    d = request.get_json(force=True)
    I.delete(_cls(d["cls"]), set(d["ids"]))
    return jsonify(ok=True)


@bp.post("/roster")
def roster():
    d = request.get_json(force=True)
    n = I.set_roster(_cls(d["cls"]), d.get("text", ""), bool(d.get("rematch")))
    return jsonify(rematched=n)


@bp.post("/recrop-all")
def recrop_all():
    return jsonify(queued=I.recrop_all(_cls(request.get_json(force=True)["cls"])))


@bp.post("/commit")
def commit():
    d = request.get_json(force=True)
    return jsonify(I.commit(_cls(d["cls"]), bool(d.get("overwrite"))))


@bp.get("/img")
def img():
    cls, iid = _cls(request.args["cls"]), request.args["id"]
    which = request.args.get("which", "out")
    d = I.load(cls)
    it = next((i for i in d["items"] if i["id"] == iid), None)
    if not it:
        abort(404, "No such item")
    p = I.cdir(cls) / (it["raw"] if which == "raw" else f"out/{iid}.jpg")
    if not p.is_file():
        abort(404, "Not ready")
    if request.args.get("full"):
        return send_file(p)
    size = (360, 140) if it["kind"] == "sign" and which != "raw" else (300, 360)
    return Response(E.thumbnail_bytes(p, size), mimetype="image/jpeg", headers={"Cache-Control": "max-age=86400"})
