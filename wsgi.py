"""Production entry point:  gunicorn -c gunicorn.conf.py wsgi:app"""
import os

if not os.environ.get("IDSTUDIO_PASSWORD"):
    raise RuntimeError("Set IDSTUDIO_PASSWORD before starting the server - this app holds student photos.")

from app import app  # noqa: E402,F401
