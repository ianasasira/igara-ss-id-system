FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-montserrat libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .
# Pre-download nothing at runtime: RapidOCR's models ship inside the pip package.
ENV IDSTUDIO_BASE=/data \
    IDSTUDIO_CONFIG=/data/config.json \
    IDSTUDIO_BIND=0.0.0.0:8000 \
    PYTHONUNBUFFERED=1

VOLUME /data
EXPOSE 8000
CMD ["gunicorn", "-c", "gunicorn.conf.py", "wsgi:app"]
