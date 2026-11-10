"""Chopify Studio: browser UI with a serial, resumable project queue and remote access."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import unquote, urlsplit
import webbrowser

import branding
import colab_utils
import llm_provider
import projects
import render_clips

HERE = Path(__file__).resolve().parent
STATIC = HERE / "assets" / "studio"


def start_cloudflare_tunnel(port):
    """Launch a Cloudflare Tunnel pointing to the local port and extract its public URL."""
    binary = shutil.which("cloudflared") or shutil.which("cloudflare")
    if not binary:
        raise RuntimeError(
            "Cloudflare Tunnel (cloudflared) is not installed.\n"
            "Please install it from: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/\n"
            "or run Chopify Studio without --tunnel."
        )
    cmd = [binary, "tunnel", "--url", f"http://127.0.0.1:{port}"]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    tunnel_url = None
    pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    start = time.time()
    while time.time() - start < 15:
        if proc.poll() is not None:
            err = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"Cloudflare Tunnel exited unexpectedly:\n{err}")
        line = proc.stderr.readline()
        if line:
            m = pattern.search(line)
            if m:
                tunnel_url = m.group(0)
                break
        time.sleep(0.1)

    if not tunnel_url:
        proc.terminate()
        raise RuntimeError("Cloudflare Tunnel started but could not detect a trycloudflare.com URL within 15 seconds.")

    return proc, tunnel_url


def project_detail(root, project_id):
    """Build the JSON payload for one project, including selection.json."""
    path = projects.project_path(root, project_id)
    state = projects.read_project(path)
    data = {"project": state, "busy": False, "candidates": [],
            "transcript": None, "selection": {}, "files": [], "log": ""}
    for filename, key in [("segments.json", "candidates"),
                          ("transcript.json", "transcript"),
                          ("selection.json", "selection")]:
        if (path / filename).exists():
            data[key] = projects.read_json(path / filename)
    for directory in ("previews", "clips"):
        for file in sorted((path / directory).glob("*")):
            if file.suffix in {".mp4", ".srt", ".png", ".json"}:
                data["files"].append({"name": file.name, "kind": directory,
                                      "url": f"/media/{project_id}/{directory}/{file.name}"})
    if (path / "job.log").exists():
        with (path / "job.log").open("rb") as log:
            log.seek(max(0, (path / "job.log").stat().st_size - 12000))
            data["log"] = log.read().decode("utf-8", errors="replace")
    transcript = data.get("transcript") or {}
    source_meta = state.get("source_meta") or {}
    render_from_sections = bool(
        source_meta.get("render_from_sections")
        or transcript.get("render_from_sections"))
    video = transcript.get("video") or state.get("video")
    data.update({
        "source_meta": source_meta,
        "progress": state.get("progress") or {},
        "render_errors": state.get("render_errors") or [],
        "playable_source": bool(
            not render_from_sections and video
            and Path(video).is_file()),
    })
    return data


class Studio:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "brands").mkdir(exist_ok=True)
        self.token = secrets.token_urlsafe(32)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chopify")
        self.lock = threading.RLock()
        self.futures = {}
        self.processes = set()
        self.closing = False
        for path in self.root.glob("*/project.json"):
            try:
                state = projects.read_project(path.parent)
                if state["status"] in {"queued", "running"}:
                    state.update(status="interrupted", error="Studio stopped. Resume this project to continue.")
                    projects.save_project(path.parent, state)
            except (ValueError, OSError, KeyError):
                continue

    def busy(self, project_id):
        future = self.futures.get(project_id)
        return future is not None and not future.done()

    def list_projects(self):
        result = []
        for path in self.root.glob("*/project.json"):
            try:
                result.append(projects.read_project(path.parent))
            except (ValueError, OSError):
                continue
        return sorted(result, key=lambda p: p["created"], reverse=True)

    def enqueue(self, project_id, action, preview=False, pick=None):
        if action not in {"prepare", "render", "all"}:
            raise ValueError("Unknown action.")
        if not isinstance(preview, bool):
            raise ValueError("Preview must be true or false.")
        if pick is not None and (not isinstance(pick, str) or not re.fullmatch(r"\d+(,\d+)*", pick)):
            raise ValueError("Select numbered clips, for example 1,3,5.")
        with self.lock:
            path = projects.project_path(self.root, project_id)
            if self.busy(project_id):
                raise ValueError("This project is already queued or running.")
            state = projects.read_project(path)
            state.update(status="queued", stage=action, error=None)
            projects.save_project(path, state)
            self.futures[project_id] = self.executor.submit(self._run, path, action, preview, pick)

    def _run(self, path, action, preview, pick):
        cmd = [sys.executable, str(HERE / "chopify.py"), "--project", str(path),
               "--action", action, "--no-input", "--out", str(path / ("previews" if preview else "clips"))]
        if preview:
            cmd.append("--preview")
        if pick:
            cmd += ["--clips", pick]
        try:
            with (path / "job.log").open("w", encoding="utf-8") as log:
                with self.lock:
                    if self.closing:
                        return
                    child = subprocess.Popen(
                        cmd, cwd=HERE, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT
                    )
                    self.processes.add(child)
                code = child.wait()
                with self.lock:
                    self.processes.discard(child)
            if code:
                state = projects.read_project(path)
                state.update(status="failed", error=state.get("error") or "Job failed. Open the job log for setup or input details.")
                projects.save_project(path, state)
        except Exception as exc:
            state = projects.read_project(path)
            state.update(status="failed", error=str(exc))
            projects.save_project(path, state)

    def close(self):
        with self.lock:
            self.closing = True
            for child in list(self.processes):
                child.terminate()
            self.executor.shutdown(wait=False, cancel_futures=True)


class Handler(BaseHTTPRequestHandler):
    server_version = "ChopifyStudio/1.4.0"

    @property
    def app(self):
        return self.server.app

    def log_message(self, fmt, *args):
        pass

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; media-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def valid_host(self):
        # Allow connections from any IP, LAN address, or Cloudflare Tunnel host
        return True

    def do_GET(self):
        try:
            parts = unquote(urlsplit(self.path).path).strip("/").split("/")
            if parts == ["api", "available-videos"]:
                return self.send_json(colab_utils.list_available_videos())
            if parts == ["api", "cookies"]:
                cpath = colab_utils.find_cookies_path()
                return self.send_json({
                    "exists": bool(cpath),
                    "path": str(cpath) if cpath else None,
                    "size_bytes": cpath.stat().st_size if cpath else 0,
                })
            if parts == ["api", "llm-config"]:
                return self.send_json(llm_provider.get_llm_config())
            if parts == ["api", "config"]:
                cpath = colab_utils.find_cookies_path()
                brands = sorted(p.stem for p in (self.app.root / "brands").glob("*.json"))
                caption_brands = {}
                for brand_path in (self.app.root / "brands").glob("*.json"):
                    try:
                        brand = branding.validate_brand(
                            projects.read_json(brand_path), brand_path.parent)
                    except (ValueError, OSError, KeyError):
                        continue
                    caption_brands[brand_path.stem] = {
                        key: brand[key] for key in branding.CAPTION_FIELDS
                        if brand[key] is not None
                    }
                return self.send_json({
                    "token": self.app.token,
                    "defaults": projects.DEFAULT_OPTIONS,
                    "llm_config": llm_provider.get_llm_config(),
                    "cookies": {
                        "exists": bool(cpath),
                        "path": str(cpath) if cpath else None,
                        "size_bytes": cpath.stat().st_size if cpath else 0,
                    },
                    "videos_dir": str(colab_utils.get_videos_dir()),
                    "brands": brands,
                    "caption_brands": caption_brands,
                    "caption_presets": render_clips.CAPTION_STYLES,
                    "caption_defaults": branding.CAPTION_DEFAULTS,
                    "caption_aspects": {
                        name: {"font_size": size, "margin": margin}
                        for name, (_, _, size, margin) in render_clips.ASPECTS.items()
                    },
                    "caption_power_words": sorted(render_clips.POWER_WORDS),
                    "caption_emoji_rules": [
                        {"words": list(stems), "emoji": emoji}
                        for stems, emoji in render_clips.EMOJI_RULES
                    ],
                })
            if parts == ["api", "projects"]:
                return self.send_json({"projects": self.app.list_projects()})
            if len(parts) == 3 and parts[:2] == ["api", "projects"]:
                data = project_detail(self.app.root, parts[2])
                data["busy"] = self.app.busy(parts[2])
                return self.send_json(data)
            if parts and parts[0] == "media" and len(parts) >= 3:
                path = projects.project_path(self.app.root, parts[1]).resolve()
                if parts[2:] == ["source"]:
                    source = projects.read_json(path / "transcript.json")["video"]
                    return self.send_file(Path(source))
                if len(parts) != 4 or parts[2] not in {"clips", "previews"}:
                    raise ValueError("Unknown media path.")
                file = (path / parts[2] / parts[3]).resolve()
                if file.parent != path / parts[2] or file.suffix not in {".mp4", ".srt", ".png", ".json"}:
                    raise ValueError("Invalid media path.")
                return self.send_file(file)
            static = {"": "index.html", "app.js": "app.js", "style.css": "style.css"}
            filename = "/".join(parts)
            if filename in static:
                return self.send_file(STATIC / static[filename])
            self.send_json({"error": "Not found."}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (OSError, KeyError):
            self.send_json({"error": "Project or file not found."}, 404)
        except (ValueError, TypeError) as exc:
            self.send_json({"error": str(exc)}, 400)

    def do_POST(self):
        origin = self.headers.get("Origin")
        token = self.headers.get("X-Chopify-Token", "")
        if not secrets.compare_digest(token, self.app.token):
            return self.send_json({"error": "Request rejected. Reload Studio and retry."}, 403)
        if origin is not None:
            origin_netloc = urlsplit(origin).netloc
            host = self.headers.get("Host")
            if origin_netloc and host and origin_netloc != host:
                return self.send_json({"error": "Cross-origin request rejected."}, 403)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 10_000_000:
                raise ValueError("Request must be JSON smaller than 10 MB.")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object.")
            parts = unquote(urlsplit(self.path).path).strip("/").split("/")
            with self.app.lock:
                if parts == ["api", "cookies"]:
                    content = body.get("content", "")
                    saved_path = colab_utils.save_cookies_file(content)
                    return self.send_json({"ok": True, "path": saved_path})
                if parts == ["api", "llm-config"]:
                    updated = llm_provider.save_llm_config(body)
                    return self.send_json({"ok": True, "config": updated})
                if parts == ["api", "projects"]:
                    return self.send_json(projects.create_project(self.app.root, body["source"], body.get("options", {})), 201)
                if parts == ["api", "batch"]:
                    sources = body["sources"]
                    if not isinstance(sources, list) or not 1 <= len(sources) <= 100 or any(not isinstance(s, str) or not s.strip() for s in sources):
                        raise ValueError("Provide 1 to 100 sources.")
                    options = projects.validate_options(body.get("options", {}))
                    result = []
                    for source in sources:
                        state = projects.create_project(self.app.root, source, options)
                        self.app.enqueue(state["id"], "prepare")
                        result.append(state)
                    return self.send_json({"projects": result}, 201)
                if parts == ["api", "brands"]:
                    name = body["name"]
                    if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,60}", name):
                        raise ValueError("Preset name must use letters, numbers, - or _.")
                    brand = branding.validate_brand(body["preset"])
                    path = self.app.root / "brands" / f"{name}.json"
                    projects.atomic_json(path, brand)
                    return self.send_json({"name": name, "path": str(path)})
                if len(parts) == 4 and parts[:2] == ["api", "projects"]:
                    project_id, action = parts[2:]
                    path = projects.project_path(self.app.root, project_id)
                    state = projects.read_project(path)
                    if self.app.busy(project_id):
                        raise ValueError("Wait for this project's current job to finish before editing.")
                    if action == "settings":
                        settings = dict(body)
                        if "brand_name" in settings:
                            name = settings.pop("brand_name")
                            if name and (not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,60}", name)):
                                raise ValueError("Invalid brand preset.")
                            settings["brand"] = str(self.app.root / "brands" / f"{name}.json") if name else None
                        state["options"] = projects.validate_options({**state["options"], **settings})
                        projects.save_project(path, state)
                    elif action == "transcript":
                        projects.edit_transcript(path, body["words"])
                    elif action == "candidates":
                        duration = projects.read_json(path / "transcript.json")["duration"]
                        segments = projects.validate_segments(body["candidates"], duration)
                        projects.atomic_json(path / "segments.json", segments)
                    elif action == "run":
                        self.app.enqueue(project_id, body.get("action", "prepare"), body.get("preview", False), body.get("pick"))
                    else:
                        return self.send_json({"error": "Not found."}, 404)
                    return self.send_json({"ok": True})
            self.send_json({"error": "Not found."}, 404)
        except FileNotFoundError:
            self.send_json({"error": "Project or transcript not found. Prepare the project first."}, 404)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            self.send_json({"error": str(exc)}, 400)

    def send_file(self, path):
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        requested = self.headers.get("Range")
        if requested:
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", requested)
            if not match:
                return self.send_json({"error": "Unsupported byte range."}, 416)
            start = int(match[1])
            end = min(int(match[2]) if match[2] else end, end)
            if start > end or start >= size:
                return self.send_json({"error": "Range outside file."}, 416)
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(max(0, end - start + 1)))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        try:
            with path.open("rb") as file:
                file.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = file.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


def make_server(root="projects", port=8765, host="0.0.0.0"):
    server = ThreadingHTTPServer((host, port), Handler)
    server.app = Studio(root)
    return server


def main():
    parser = argparse.ArgumentParser(description="Open Chopify Studio.")
    parser.add_argument("--projects-dir", default="projects")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--tunnel", "--cloudflare", action="store_true",
                        help="Auto-create a public Cloudflare Tunnel and print the remote URL")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = make_server(args.projects_dir, args.port, host=args.host)
    local_url = f"http://127.0.0.1:{server.server_port}"
    print(f"Chopify Studio: {local_url} (accessible from any IP on port {server.server_port})\nProjects: {server.app.root}\nPress Ctrl+C to stop.", flush=True)

    tunnel_proc = None
    if args.tunnel:
        try:
            tunnel_proc, tunnel_url = start_cloudflare_tunnel(server.server_port)
            print(f"\n=======================================================", flush=True)
            print(f"🚀 Cloudflare Tunnel Active!", flush=True)
            print(f"🌐 Remote Studio URL: {tunnel_url}", flush=True)
            print(f"=======================================================\n", flush=True)
        except RuntimeError as exc:
            print(f"\n❌ Error: {exc}\n", file=sys.stderr, flush=True)
            server.app.close()
            server.server_close()
            sys.exit(1)

    if not args.no_browser:
        webbrowser.open(local_url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if tunnel_proc:
            tunnel_proc.terminate()
        server.app.close()
        server.server_close()


if __name__ == "__main__":
    main()
