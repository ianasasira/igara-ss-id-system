# Igara ID Studio

Local web app that automates the whole student-ID pipeline:
clean file names → rotate → resize signatures (360×135) → preview → generate IDs (+ optional A4 print sheets).

## Run (Linux)
```bash
cd igara-id-studio
pip install -r requirements.txt   # includes OpenCV + RapidOCR
sudo apt install fonts-montserrat      # or drop Montserrat-SemiBold.ttf into ./fonts/
python3 app.py                         # opens http://127.0.0.1:5000
```
Then in **Setup** enter your `gara id` folder (the one with S.1 … S.6, SIGNS and `new/empty template.png`).

## Workflow
1. **Setup** – set the folder, see counts and missing signatures.
2. **Prepare images** – clean names, rotate sideways photos/signatures, resize signatures. Originals are copied to `_originals/` before the first change; "Restore originals" brings them back.
3. **Classes & layout** – ID prefixes, class text, validity, issue date, font size (7.8 pt), pixel layout.
4. **Generate** – preview any student, then generate. Output: `new/<class>/NAME.png` (+ `<class>_print_sheet.pdf`, 10 cards per A4 page, if ticked). Download per-class zips.

## Server deployment
See **DEPLOY.md** (Docker, systemd, nginx + HTTPS, login). Local use needs no password.

## Intake (step 1)
Drop raw photos + signature sheets per class; the app crops passports (head/shoulders/upper chest), cuts out signatures, reads the name from the paper (offline OCR, matched to your class list) and lets you review/rename/replace before saving into the project folders.

## Files
- `engine.py` – all image logic, no web code (reuse from CLI/Electron)
- `app.py` – Flask API (binds to 127.0.0.1 only)
- `static/index.html` – the UI
- `config.json` – created on first save

## Electron later
Spawn `python3 app.py --no-browser --port <free port>` from the main process (bundle with PyInstaller for Windows),
then `win.loadURL('http://127.0.0.1:<port>')`. Nothing in the UI needs to change.
