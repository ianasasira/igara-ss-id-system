import os

bind = os.environ.get("IDSTUDIO_BIND", "127.0.0.1:8000")
# IMPORTANT: keep ONE worker process. The OCR queue and progress state live in memory;
# concurrency comes from threads instead.
workers = 1
threads = 8
timeout = 300
accesslog = "-"
errorlog = "-"
