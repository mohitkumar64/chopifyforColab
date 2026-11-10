"""LLM Provider Interface for Chopify: Gemini API, NVIDIA NIM, and Ollama.

Provides structured clip virality reranking across multiple LLM backends with
automatic fallback to deterministic virality scoring if an API fails or is unavailable.
"""
import json
import math
import os
from pathlib import Path
import re
import urllib.error
import urllib.request

# Default environment and endpoints
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NVIDIA_MODEL = "meta/llama-3.3-70b-instruct"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"

RERANK_SYSTEM = (
    "You are a short-form video editor reviewing candidate clips. "
    "For each candidate, return a score adjustment between -1.0 and 1.0. "
    "Positive means stronger viral potential; negative means weaker. "
    "Never change timestamps. Respond with JSON only: "
    '[{"index": <int>, "adjustment": <-1..1>, "reason": "<short reason>"}]'
)


def find_env_path():
    """Locate .env in chopifyforColab repo root or cwd."""
    candidates = [
        Path(__file__).resolve().parent / ".env",
        Path.cwd() / ".env",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return Path(__file__).resolve().parent / ".env"


def load_env(env_path=None):
    """Load environment variables from .env file (using python-dotenv or fallback)."""
    target = Path(env_path) if env_path else find_env_path()
    try:
        from dotenv import load_dotenv
        if target.is_file():
            load_dotenv(target)
            return
        load_dotenv()
        return
    except ImportError:
        pass

    # Standard-library fallback parser
    if target.is_file():
        try:
            content = target.read_text(encoding="utf-8")
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v
        except OSError:
            pass


# Ensure .env is loaded on import
load_env()


def get_llm_config():
    """Return the active LLM configuration and key presence."""
    return {
        "llm_provider": os.environ.get("LLM_PROVIDER", "gemini"),
        "gemini_api_key_set": bool(os.environ.get("GEMINI_API_KEY")),
        "gemini_model": os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        "nvidia_api_key_set": bool(os.environ.get("NVIDIA_API_KEY")),
        "nvidia_base_url": os.environ.get("NVIDIA_BASE_URL", DEFAULT_NVIDIA_BASE_URL),
        "nvidia_model": os.environ.get("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL),
        "ollama_host": os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
        "ollama_model": os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
    }


def save_llm_config(updates, env_path=None):
    """Save updated LLM configuration into the .env file and update os.environ."""
    target = Path(env_path) if env_path else find_env_path()
    existing_lines = []
    if target.is_file():
        try:
            existing_lines = target.read_text(encoding="utf-8").splitlines()
        except OSError:
            existing_lines = []

    # Map keys to env variable names
    key_map = {
        "llm_provider": "LLM_PROVIDER",
        "gemini_api_key": "GEMINI_API_KEY",
        "gemini_model": "GEMINI_MODEL",
        "nvidia_api_key": "NVIDIA_API_KEY",
        "nvidia_base_url": "NVIDIA_BASE_URL",
        "nvidia_model": "NVIDIA_MODEL",
        "ollama_host": "OLLAMA_HOST",
        "ollama_model": "OLLAMA_MODEL",
    }

    env_updates = {}
    for k, v in updates.items():
        env_key = key_map.get(k, k.upper())
        if v is not None:
            val_str = str(v).strip()
            env_updates[env_key] = val_str
            os.environ[env_key] = val_str

    # Rewrite or append to .env
    updated_keys = set()
    new_lines = []
    for line in existing_lines:
        trimmed = line.strip()
        if trimmed and not trimmed.startswith("#") and "=" in trimmed:
            k, _ = trimmed.split("=", 1)
            k = k.strip()
            if k in env_updates:
                new_lines.append(f"{k}={env_updates[k]}")
                updated_keys.add(k)
                continue
        new_lines.append(line)

    for k, v in env_updates.items():
        if k not in updated_keys:
            new_lines.append(f"{k}={v}")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(new_lines).strip() + "\n", encoding="utf-8")
    return get_llm_config()


def _candidate_key(candidate):
    """Stable identity for merging rerank adjustments into candidate records."""
    return (round(float(candidate["start"]), 2), round(float(candidate["end"]), 2),
            candidate.get("seed_index"))


def _parse_llm_json(content):
    """Parse JSON array of adjustments from raw model text output."""
    if not content or not isinstance(content, str):
        return []
    cleaned = content.strip()
    # Strip markdown code blocks
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned).strip()

    parsed = None
    try:
        parsed = json.loads(cleaned)
    except Exception:
        match = re.search(r"\[.*\]", cleaned, re.S)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except Exception:
                pass
        if parsed is None:
            match_obj = re.search(r"\{.*\}", cleaned, re.S)
            if match_obj:
                try:
                    parsed = json.loads(match_obj.group(0))
                except Exception:
                    pass

    if isinstance(parsed, dict):
        for v in parsed.values():
            if isinstance(v, list):
                parsed = v
                break

    if not isinstance(parsed, list):
        return []

    valid = []
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        idx = entry.get("index")
        adj = entry.get("adjustment")
        if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0:
            continue
        if isinstance(adj, bool) or not isinstance(adj, (int, float)) or not math.isfinite(adj):
            continue
        valid.append({
            "index": idx,
            "adjustment": max(-1.0, min(1.0, float(adj))),
            "reason": str(entry.get("reason", ""))[:120],
        })
    return valid


def _gemini_chat(sent, model, api_key, timeout=120):
    """Call Google Gemini API via official google-genai SDK with structured output."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "google-genai SDK not installed. Install with: pip install google-genai"
        ) from exc

    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    # Use JSON schema / structured output
    schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "index": {"type": "INTEGER"},
                "adjustment": {"type": "NUMBER"},
                "reason": {"type": "STRING"},
            },
            "required": ["index", "adjustment"],
        },
    }

    config = types.GenerateContentConfig(
        system_instruction=RERANK_SYSTEM,
        response_mime_type="application/json",
        response_schema=schema,
        temperature=0.2,
    )

    prompt = json.dumps(sent, ensure_ascii=False)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    raw_text = getattr(response, "text", "") or ""
    return _parse_llm_json(raw_text)


def _nvidia_chat(sent, model, api_key, base_url, timeout=120):
    """Call NVIDIA NIM API via OpenAI Python SDK pointed at NVIDIA endpoint."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai SDK not installed. Install with: pip install openai"
        ) from exc

    if not api_key:
        raise ValueError("NVIDIA_API_KEY is not set.")

    client = OpenAI(
        base_url=base_url or DEFAULT_NVIDIA_BASE_URL,
        api_key=api_key,
        timeout=timeout,
    )

    messages = [
        {"role": "system", "content": RERANK_SYSTEM},
        {"role": "user", "content": json.dumps(sent, ensure_ascii=False)},
    ]

    response = client.chat.completions.create(
        model=model or DEFAULT_NVIDIA_MODEL,
        messages=messages,
        temperature=0.2,
    )

    content = ""
    if response.choices and response.choices[0].message:
        content = response.choices[0].message.content or ""
    return _parse_llm_json(content)


def _ollama_chat(model, host, messages, timeout=120):
    """POST to local Ollama chat endpoint and return parsed adjustments."""
    payload = {
        "model": model, "stream": False, "format": "json",
        "options": {"temperature": 0.2},
        "messages": messages,
    }
    request = urllib.request.Request(
        host.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content = json.loads(response.read().decode("utf-8")).get(
            "message", {}).get("content", "")
    return _parse_llm_json(content)


def gemini_rerank(candidates, model=None, api_key=None, timeout=120):
    """Rerank candidates using Gemini API."""
    if not candidates:
        return None
    model = model or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    ranked = sorted(candidates, key=lambda item: (
        -item["overall"], item["start"], item["duration"]))[:50]
    sent = [{"index": i, "start": item["start"], "end": item["end"],
             "text": str(item.get("text", ""))[:400]}
            for i, item in enumerate(ranked)]
    try:
        raw = _gemini_chat(sent, model, api_key, timeout=timeout)
    except Exception as exc:
        print(f"Gemini unavailable ({exc}) -> deterministic scores", flush=True)
        return None
    return apply_adjustments(candidates, ranked, raw, model)


def nvidia_rerank(candidates, model=None, api_key=None, base_url=None, timeout=120):
    """Rerank candidates using NVIDIA NIM API."""
    if not candidates:
        return None
    model = model or os.environ.get("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL)
    api_key = api_key or os.environ.get("NVIDIA_API_KEY")
    base_url = base_url or os.environ.get("NVIDIA_BASE_URL", DEFAULT_NVIDIA_BASE_URL)
    ranked = sorted(candidates, key=lambda item: (
        -item["overall"], item["start"], item["duration"]))[:50]
    sent = [{"index": i, "start": item["start"], "end": item["end"],
             "text": str(item.get("text", ""))[:400]}
            for i, item in enumerate(ranked)]
    try:
        raw = _nvidia_chat(sent, model, api_key, base_url, timeout=timeout)
    except Exception as exc:
        print(f"NVIDIA NIM unavailable ({exc}) -> deterministic scores", flush=True)
        return None
    return apply_adjustments(candidates, ranked, raw, model)


def ollama_rerank(candidates, model=None, host=DEFAULT_OLLAMA_HOST, timeout=120):
    """Rerank candidates using local Ollama endpoint."""
    if not candidates:
        return None
    model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    ranked = sorted(candidates, key=lambda item: (
        -item["overall"], item["start"], item["duration"]))[:50]
    sent = [{"index": i, "start": item["start"], "end": item["end"],
             "text": str(item.get("text", ""))[:400]}
            for i, item in enumerate(ranked)]
    messages = [
        {"role": "system", "content": RERANK_SYSTEM},
        {"role": "user", "content": json.dumps(sent, ensure_ascii=False)},
    ]
    try:
        raw = _ollama_chat(model, host, messages, timeout)
    except Exception as exc:
        print(f"Ollama unavailable ({exc}) -> deterministic scores", flush=True)
        return None
    return apply_adjustments(candidates, ranked, raw, model)


def apply_adjustments(candidates, ranked, raw_adjustments, model):
    """Merge score adjustments into candidate records without altering timestamps."""
    if not raw_adjustments:
        return None

    adjustments = {}
    for entry in raw_adjustments:
        idx = entry.get("index")
        adj = entry.get("adjustment")
        if isinstance(idx, int) and 0 <= idx < len(ranked) and isinstance(adj, (int, float)):
            adjustments[idx] = (max(-1.0, min(1.0, float(adj))),
                                str(entry.get("reason", ""))[:120])

    if not adjustments:
        return None

    updated = [dict(item) for item in candidates]
    positions = {_candidate_key(item): i for i, item in enumerate(updated)}
    for index, (adjustment, reason) in adjustments.items():
        position = positions.get(_candidate_key(ranked[index]))
        if position is None:
            continue
        record = updated[position]
        base = float(record.get("overall", 0.0))
        previous = record.get("llm_adjustment")
        if isinstance(previous, (int, float)) and not isinstance(previous, bool):
            base -= float(previous)
        record["overall"] = round(max(0.0, min(10.0, base + adjustment)), 2)
        record["llm_adjustment"] = adjustment
        record["llm_model"] = model
        if reason and reason not in str(record.get("reason", "")):
            base_reason = str(record.get("reason", ""))[:160]
            record["reason"] = (base_reason + " | " + reason)[:160]
    return updated


def rerank(candidates, model=None, provider=None, host=DEFAULT_OLLAMA_HOST, timeout=120):
    """Rerank candidates using the selected LLM provider.

    Returns (updated_candidates, mode). If provider fails or returns no adjustments,
    returns (None, 'heuristic').
    """
    if not candidates:
        return None, "heuristic"

    # Resolve provider
    chosen_provider = (provider or os.environ.get("LLM_PROVIDER") or "ollama").strip().lower()

    if chosen_provider == "heuristic":
        return None, "heuristic"

    if chosen_provider == "gemini":
        adjusted = gemini_rerank(candidates, model=model, timeout=timeout)
        if adjusted is not None:
            return adjusted, "gemini"
        return None, "heuristic"

    if chosen_provider in {"nvidia", "nvidia_nim", "nim"}:
        adjusted = nvidia_rerank(candidates, model=model, timeout=timeout)
        if adjusted is not None:
            return adjusted, "nvidia"
        return None, "heuristic"

    if chosen_provider == "ollama":
        adjusted = ollama_rerank(candidates, model=model, host=host, timeout=timeout)
        if adjusted is not None:
            return adjusted, "ollama"
        return None, "heuristic"

    print(f"Unknown LLM provider '{chosen_provider}' -> deterministic scores", flush=True)
    return None, "heuristic"
