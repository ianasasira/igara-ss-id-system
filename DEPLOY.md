# Putting Igara ID Studio on a server

You need a Linux server (Ubuntu 22.04/24.04), **≥ 2 GB RAM**, and ideally a domain name.
This app handles photos of students, so it always runs behind a **login** and should be served over **HTTPS**.

## Option A – Docker (easiest)
```bash
# on the server
sudo apt install -y docker.io docker-compose-v2 nginx certbot python3-certbot-nginx
unzip igara-id-studio.zip && cd igara-id-studio
cp .env.example .env && nano .env            # set a long random IDSTUDIO_PASSWORD
sudo docker compose up -d --build            # app now listens on 127.0.0.1:8000
```
All data lives in `./data` (back it up).

## Option B – no Docker
```bash
sudo apt install -y python3-venv fonts-montserrat nginx certbot python3-certbot-nginx
sudo useradd -r -m idstudio
sudo mkdir /opt/igara-id-studio && sudo unzip igara-id-studio.zip -d /opt && sudo chown -R idstudio /opt/igara-id-studio
cd /opt/igara-id-studio
sudo -u idstudio python3 -m venv venv
sudo -u idstudio venv/bin/pip install -r requirements.txt gunicorn
sudo mkdir -p /srv/idstudio-data && sudo chown idstudio /srv/idstudio-data
sudo tee /etc/igara-id-studio.env >/dev/null <<ENV
IDSTUDIO_USER=admin
IDSTUDIO_PASSWORD=change-this-to-a-long-random-password
IDSTUDIO_BASE=/srv/idstudio-data
IDSTUDIO_CONFIG=/srv/idstudio-data/config.json
ENV
sudo chmod 600 /etc/igara-id-studio.env
sudo cp deploy/idstudio.service /etc/systemd/system/ && sudo systemctl enable --now idstudio
```

## HTTPS reverse proxy (both options)
```bash
sudo cp deploy/nginx.conf /etc/nginx/sites-available/idstudio      # edit server_name first
sudo ln -s /etc/nginx/sites-available/idstudio /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d ids.example.com                             # free HTTPS certificate
```
Open `https://ids.example.com`, sign in (user `admin` + your password).

## First run
1. **Setup → ID template**: upload your blank `1011 × 638` card PNG.
2. **Intake**: pick a class, paste the class list (names), drop in the photos and signature sheets, review, **Save to project**.
3. **Generate**: preview, generate, download the per-class zip.

The data folder is fixed on the server (`/data` in Docker). To bring existing folders (S.1 … S.6, SIGNS) onto the server, copy them into that folder with `scp`/SFTP.

## Notes
- **Keep one worker.** `gunicorn.conf.py` uses 1 process + 8 threads on purpose (the OCR queue is in memory). Don't raise `workers`.
- Basic-auth credentials travel with every request, so **use HTTPS**; never expose port 8000 directly.
- When a class is finished, delete its raw pictures (`data/_intake/<class>/raw_*`) – they are full-size photos of students.
- First OCR after a restart takes a few extra seconds (models load once).
