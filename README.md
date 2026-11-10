# Chopify (Colab & Multi-LLM Fork)

> [!NOTE]
> **This repository is a fork of the original [Chopify](https://github.com/mehbul/chopify) created by [@mehbul](https://github.com/mehbul).**  
> For upstream features, original installation guides, architecture docs, and workflows, please visit the original repository:  
> 👉 **[https://github.com/mehbul/chopify](https://github.com/mehbul/chopify)** (or the [Live Tour](https://mehbul.github.io/chopify/))

---

## About This Fork

This fork adapts **Chopify** for cloud environments such as **Google Colab** and headless GPU servers, introduces multi-provider LLM clip selection (Google Gemini & NVIDIA NIM), provides automated remote tunneling via Cloudflare, and integrates YouTube cookie management.

---

## Modifications in This Fork

### 1. Multi-LLM Provider Architecture (Gemini, NVIDIA NIM, Ollama)
- **Google Gemini API**: Integrated via the official `google-genai` SDK using JSON-schema structured output for viral clip ranking.
- **NVIDIA NIM API**: Integrated using the official `openai` SDK pointed at `https://integrate.api.nvidia.com/v1` (defaulting to `meta/llama-3.3-70b-instruct`).
- **Ollama Compatibility**: Retained local model support.
- **Automatic Heuristic Fallback**: If an API key is missing or an API call fails/times out, Chopify automatically falls back to deterministic heuristic scoring without crashing.
- **CLI Options**:
  ```bash
  # Google Gemini
  python -m chopify video.mp4 --llm-provider gemini --llm gemini-3.8-flash

  # NVIDIA NIM
  python -m chopify video.mp4 --llm-provider nvidia --llm meta/llama-3.3-70b-instruct

  # Local Ollama
  python -m chopify video.mp4 --llm-provider ollama --llm llama3.1:8b

  # Pure Heuristic
  python -m chopify video.mp4 --llm-provider heuristic
  ```

### 2. In-Dashboard LLM & API Key Manager
- Added a dedicated **"⚙️ LLM & Colab Settings"** modal in the Chopify Studio GUI.
- Configure Gemini API keys, NVIDIA NIM credentials, Ollama endpoints, and model selections directly from the browser.
- Saves and persists directly to a local `.env` file.

### 3. Remote Dashboard Access & Auto Cloudflare Tunnel
- **Any-IP Binding**: `dashboard.py` now binds to `0.0.0.0` and accepts requests from remote/tunnel hosts (removing the local `127.0.0.1` restriction while keeping CSRF token and origin protection).
- **Auto Cloudflare Tunnel**: Launch Studio with `--tunnel` (or `--cloudflare`) to automatically generate a public `https://*.trycloudflare.com` URL:
  ```bash
  python dashboard.py --tunnel
  ```
  *(If `cloudflared` is not installed, it outputs a clear error with installation instructions.)*

### 4. Google Colab YouTube Cookies Support (`cookies.txt`)
- Colab environments often encounter YouTube bot/sign-in blocks with `yt-dlp`.
- **UI Cookie Management**: Upload or paste `cookies.txt` directly from the dashboard; it is automatically saved to `/content/cookies.txt` or repository root.
- **`yt-dlp` Integration**: All video downloads and metadata probing automatically detect and pass `--cookies`.

### 5. Persistent Colab Video Storage (`/content/videos`) with 1-2-3 Numbering
- Video downloads are automatically stored in `/content/videos` (with fallback to `./videos`).
- **1-2-3 Numbering Clause**: Prevents overwriting previously downloaded files (e.g., `video.mp4` $\rightarrow$ `video_1.mp4` $\rightarrow$ `video_2.mp4`).
- **Available Videos Explorer**: Studio GUI displays all videos present in `/content/videos` with their file sizes, a **"📋 Copy Path"** button for quick path copying, and a **"＋ Add"** button to immediately queue them.

### 6. Environment & Secret Management
- Template `.env` provided with variable names only (no keys committed).
- `.gitignore` updated to ensure `.env`, `.env.*`, `cookies.txt`, and downloaded videos are never committed to version control.

---

## Quick Start (Google Colab / Headless Server)

You can launch Chopify in Google Colab in one click using the included notebook:
👉 **[`chopify_colab.ipynb`](chopify_colab.ipynb)**

Or run manually:

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install google-genai openai python-dotenv

# 2. Launch Studio with Cloudflare Tunnel
python dashboard.py --tunnel
```

Open the printed `https://*.trycloudflare.com` URL, open the **"LLM & Colab Settings"** modal, paste your API keys or upload your `cookies.txt`, and start clipping!

---

## Original Project & Upstream Documentation

For the full feature matrix, caption customization guide, local workflows, and troubleshooting, please refer to the upstream repository:
- **Original Repository**: [https://github.com/mehbul/chopify](https://github.com/mehbul/chopify)
- **Live Tour**: [https://mehbul.github.io/chopify/](https://mehbul.github.io/chopify/)
- **License**: MIT (same as original)
