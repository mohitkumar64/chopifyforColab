"""Colab utilities: cookies.txt management, /content/videos storage with 1-2-3 numbering, and video discovery."""
import os
from pathlib import Path
import shutil

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi"}


def find_cookies_path():
    """Locate cookies.txt in Colab /content, chopifyforColab root, or cwd."""
    candidates = [
        Path("/content/cookies.txt"),
        Path(__file__).resolve().parent / "cookies.txt",
        Path.cwd() / "cookies.txt",
    ]
    env_cookie = os.environ.get("CHOPIFY_COOKIES") or os.environ.get("COOKIES_FILE")
    if env_cookie:
        candidates.insert(0, Path(env_cookie))

    for p in candidates:
        try:
            if p.is_file() and p.stat().st_size > 0:
                return p.resolve()
        except (OSError, PermissionError):
            continue
    return None


def yt_dlp_cookie_args():
    """Return ['--cookies', <path>] if cookies.txt exists, else empty list."""
    cookie = find_cookies_path()
    if cookie:
        return ["--cookies", str(cookie)]
    return []


def save_cookies_file(content):
    """Save cookies.txt to /content (if writable) and chopifyforColab repo root."""
    if not isinstance(content, str):
        content = str(content)
    cleaned = content.strip() + "\n"

    saved_paths = []
    # If in Colab or /content is writable
    colab_content = Path("/content")
    if colab_content.is_dir():
        try:
            target = colab_content / "cookies.txt"
            target.write_text(cleaned, encoding="utf-8")
            saved_paths.append(target.resolve())
        except (OSError, PermissionError):
            pass

    # Save to repo root (chopifyforColab)
    for folder in [Path(__file__).resolve().parent, Path.cwd()]:
        try:
            target = folder / "cookies.txt"
            target.write_text(cleaned, encoding="utf-8")
            saved_paths.append(target.resolve())
        except (OSError, PermissionError):
            pass

    return str(saved_paths[0]) if saved_paths else None


def get_videos_dir():
    """Return the videos directory, prioritizing /content/videos in Colab."""
    env_dir = os.environ.get("CHOPIFY_VIDEOS_DIR")
    if env_dir:
        p = Path(env_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p.resolve()

    colab_dir = Path("/content/videos")
    try:
        colab_dir.mkdir(parents=True, exist_ok=True)
        return colab_dir.resolve()
    except (OSError, PermissionError):
        pass

    local_dir = Path.cwd() / "videos"
    local_dir.mkdir(parents=True, exist_ok=True)
    return local_dir.resolve()


def unique_video_path(directory, filename):
    """Return a unique file path in directory using a 1-2-3 numbering clause to prevent overwriting."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    orig_path = Path(filename)
    stem = orig_path.stem
    suffix = orig_path.suffix or ".mp4"

    target = directory / f"{stem}{suffix}"
    if not target.exists():
        return target.resolve()

    counter = 1
    while True:
        candidate = directory / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate.resolve()
        counter += 1


def list_available_videos():
    """List all video files available in /content/videos (or fallback videos dir)."""
    vdir = get_videos_dir()
    results = []
    if not vdir.is_dir():
        return {"directory": str(vdir), "videos": []}

    try:
        for item in sorted(vdir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if item.is_file() and item.suffix.lower() in VIDEO_EXTS:
                size_mb = round(item.stat().st_size / (1024 * 1024), 2)
                results.append({
                    "name": item.name,
                    "path": str(item.resolve()),
                    "size_mb": size_mb,
                    "mtime": item.stat().st_mtime,
                })
    except (OSError, PermissionError):
        pass

    return {
        "directory": str(vdir),
        "videos": results,
    }

