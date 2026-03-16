# ALTRU OCR

**Free, private, offline text recognition that runs entirely in your browser.**

No servers. No API keys. No data leaves your device. Ever.

[**→ Launch App**](https://altrudev.github.io/ocr/)

---

## What It Does

Upload an image or snap a photo with your camera. ALTRU OCR extracts the text using PaddleOCR — one of the most accurate open-source OCR engines in the world — running 100% locally in your browser via WebAssembly.

Your documents, receipts, screenshots, and photos never touch a server.

## Features

- **Fully offline** after first visit — works on airplane mode
- **Camera input** — point your phone at text and capture
- **Smart preprocessing** — auto-fixes contrast, skew, noise, and borders before OCR
- **5 presets** — Auto, Clean Scan, Phone Photo, Old/Degraded, Receipt
- **Layout analysis** — detects headings, paragraphs, captions with reading order
- **Confidence scoring** — calibrated per-region scores so you know what to trust
- **Installable PWA** — add to home screen, use like a native app
- **Zero cost** — no accounts, no limits, no ads

## How It Works

1. **You open the app** — static HTML/JS/CSS loads instantly from GitHub Pages
2. **First use** — the engine downloads two small AI models (~10MB) from GitHub and caches them via Service Worker
3. **You upload or photograph a document** — the image goes through a preprocessing pipeline (contrast enhancement, deskew, noise removal, binarization)
4. **PaddleOCR runs locally** — DB algorithm detects text regions, CRNN network reads each one, CTC decoder produces text
5. **Layout analysis + confidence calibration** — regions are classified (heading, paragraph, caption), sorted into reading order, and scored for reliability
6. **You get structured results** — plain text, per-region breakdowns, and processing metadata

All of this happens on your device in your browser tab. The AI models run via ONNX Runtime WebAssembly — no GPU required, works on any modern browser.

## Preprocessing Pipeline

Real-world images need cleanup before OCR can work well. The engine applies (based on your selected preset):

| Step | What It Does | Why It Matters |
|---|---|---|
| Scale normalization | Resizes to the OCR sweet spot | Too small = missed text, too large = slow |
| CLAHE | Adaptive contrast enhancement | Fixes shadows, uneven lighting, flash glare |
| Grayscale | Luminance conversion | Reduces noise from color channels |
| Deskew | Detects and corrects rotation | Skewed text = garbled output |
| Median filter | Removes salt-and-pepper noise | Scanner artifacts, compression noise |
| Sauvola binarization | Adaptive black/white threshold | Handles colored paper, gradient backgrounds |

## Tech Stack

- **PaddleOCR** (Baidu, Apache 2.0) — PP-OCRv3 detection + PP-OCRv4 recognition models
- **ONNX Runtime Web** (Microsoft, MIT) — browser-based AI inference via WebAssembly
- **Vanilla JS** — no framework, no build step, no dependencies beyond ONNX Runtime
- **Service Worker** — caches everything for true offline operation

## Privacy

This is not a privacy policy with asterisks. The architecture makes surveillance impossible:

- The app is static files on GitHub Pages — there is no server to send data to
- OCR processing runs in your browser's JavaScript engine
- AI models are downloaded once and cached locally
- No analytics, no tracking, no cookies, no accounts
- The source code is right here — read every line

## Self-Hosting

These are just static files. Host them anywhere:

- **GitHub Pages** — push to a repo, enable Pages
- **Any web server** — upload the files, done
- **Local** — open `index.html` in a browser (camera requires HTTPS)
- **cPanel / Apache / Nginx** — drop into public_html

## Files

| File | What It Is |
|---|---|
| `index.html` | Entry point — loads ONNX Runtime from CDN |
| `app.js` | Complete application — preprocessing + OCR inference + UI |
| `app.css` | All styles |
| `sw.js` | Service Worker for offline caching |
| `manifest.json` | PWA manifest for install-to-home-screen |
| `icon-192.png` | App icon (192×192) |
| `icon-512.png` | App icon (512×512) |
| `favicon.ico` | Browser tab icon |

Total: 8 files, 61KB. The AI models (~10MB) download on first use and cache automatically.

## Credits

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) by Baidu — OCR models (Apache 2.0)
- [ppu-paddle-ocr](https://github.com/PT-Perkasa-Pilar-Utama/ppu-paddle-ocr) — ONNX inference reference (MIT)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) by Microsoft — browser inference (MIT)
- [RapidOCR](https://github.com/RapidAI/RapidOCR) — community model conversions

## License

MIT

---

**[ALTRU.dev](https://altru.dev)** — Code for Humanity
