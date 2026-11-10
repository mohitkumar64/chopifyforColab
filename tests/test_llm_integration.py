"""Comprehensive unit and integration tests for LLM providers, Cloudflare Tunnel, and Dashboard host check."""
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

# Mock numpy if not installed so dashboard & projects can be imported cleanly in tests
if "numpy" not in sys.modules:
    class MockArray(list):
        def astype(self, dtype):
            return self
        def __ge__(self, other):
            return MockArray([x >= other for x in self])

    mock_np = MagicMock()
    mock_np.hanning.return_value = MockArray([0.0] * 800)
    mock_np.fft.rfftfreq.return_value = MockArray([i * 20.0 for i in range(401)])
    mock_np.float32 = float
    mock_np.nan = float("nan")
    mock_np.isfinite = lambda x: True
    sys.modules["numpy"] = mock_np

if "cv2" not in sys.modules:
    sys.modules["cv2"] = MagicMock()

# Ensure chopifyforColab is in path
HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dashboard
import llm_provider
import colab_utils


class TestEnvAndConfig(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.env_file = Path(self.temp_dir.name) / ".env"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_and_load_config(self):
        updates = {
            "llm_provider": "nvidia",
            "gemini_api_key": "test-gemini-key",
            "gemini_model": "gemini-3.8-flash",
            "nvidia_api_key": "test-nv-key",
            "nvidia_base_url": "https://integrate.api.nvidia.com/v1",
            "nvidia_model": "meta/llama-3.3-70b-instruct",
            "ollama_host": "http://127.0.0.1:11434",
            "ollama_model": "llama3.1:8b",
        }
        cfg = llm_provider.save_llm_config(updates, env_path=self.env_file)
        self.assertEqual(cfg["llm_provider"], "nvidia")
        self.assertTrue(cfg["gemini_api_key_set"])
        self.assertTrue(cfg["nvidia_api_key_set"])
        self.assertEqual(cfg["gemini_model"], "gemini-3.8-flash")
        self.assertEqual(cfg["nvidia_model"], "meta/llama-3.3-70b-instruct")

        # Verify .env contents
        content = self.env_file.read_text(encoding="utf-8")
        self.assertIn("LLM_PROVIDER=nvidia", content)
        self.assertIn("GEMINI_API_KEY=test-gemini-key", content)
        self.assertIn("NVIDIA_API_KEY=test-nv-key", content)


class TestJsonParsing(unittest.TestCase):
    def test_parse_pure_json(self):
        raw = '[{"index": 0, "adjustment": 0.5, "reason": "strong hook"}]'
        parsed = llm_provider._parse_llm_json(raw)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["index"], 0)
        self.assertAlmostEqual(parsed[0]["adjustment"], 0.5)
        self.assertEqual(parsed[0]["reason"], "strong hook")

    def test_parse_markdown_fenced(self):
        raw = '```json\n[{"index": 2, "adjustment": -0.4, "reason": "boring"}]\n```'
        parsed = llm_provider._parse_llm_json(raw)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["index"], 2)
        self.assertAlmostEqual(parsed[0]["adjustment"], -0.4)

    def test_parse_dict_with_nested_list(self):
        raw = '{"clips": [{"index": 1, "adjustment": 0.9, "reason": "viral"}]}'
        parsed = llm_provider._parse_llm_json(raw)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["index"], 1)
        self.assertAlmostEqual(parsed[0]["adjustment"], 0.9)

    def test_clamping_and_validation(self):
        raw = '[{"index": 0, "adjustment": 99.0, "reason": "huge"}, {"index": "bad"}, {"index": 1, "adjustment": -5.0}]'
        parsed = llm_provider._parse_llm_json(raw)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["adjustment"], 1.0)
        self.assertEqual(parsed[1]["adjustment"], -1.0)


class TestGeminiProvider(unittest.TestCase):
    def test_missing_key_fallback(self):
        candidates = [{"start": 0.0, "end": 10.0, "duration": 10.0, "overall": 5.0, "seed_index": 0}]
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            result = llm_provider.gemini_rerank(candidates, api_key="")
            self.assertIsNone(result)

    def test_gemini_sdk_mock(self):
        candidates = [{"start": 0.0, "end": 10.0, "duration": 10.0, "overall": 7.0, "seed_index": 0}]
        mock_response = MagicMock()
        mock_response.text = '[{"index": 0, "adjustment": 0.4, "reason": "great punchline"}]'

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_types = MagicMock()

        with patch.dict(sys.modules, {"google": MagicMock(genai=mock_genai), "google.genai": mock_genai, "google.genai.types": mock_types}):
            updated = llm_provider._gemini_chat([{"index": 0}], "gemini-3.8-flash", "fake-key")
            self.assertEqual(len(updated), 1)
            self.assertAlmostEqual(updated[0]["adjustment"], 0.4)

            # Test end to end rerank
            res = llm_provider.gemini_rerank(candidates, api_key="fake-key")
            self.assertIsNotNone(res)
            self.assertAlmostEqual(res[0]["overall"], 7.4)
            self.assertIn("great punchline", res[0]["reason"])


class TestNvidiaProvider(unittest.TestCase):
    def test_missing_key_fallback(self):
        candidates = [{"start": 0.0, "end": 10.0, "duration": 10.0, "overall": 5.0, "seed_index": 0}]
        with patch.dict(os.environ, {"NVIDIA_API_KEY": ""}):
            result = llm_provider.nvidia_rerank(candidates, api_key="")
            self.assertIsNone(result)

    def test_nvidia_sdk_mock(self):
        candidates = [{"start": 0.0, "end": 10.0, "duration": 10.0, "overall": 6.5, "seed_index": 0}]
        mock_choice = MagicMock()
        mock_choice.message.content = '[{"index": 0, "adjustment": 0.5, "reason": "exciting intro"}]'
        mock_response = MagicMock(choices=[mock_choice])

        mock_openai_client = MagicMock()
        mock_openai_client.chat.completions.create.return_value = mock_response
        mock_openai_class = MagicMock(return_value=mock_openai_client)

        with patch.dict(sys.modules, {"openai": MagicMock(OpenAI=mock_openai_class)}):
            updated = llm_provider._nvidia_chat([{"index": 0}], "meta/llama-3.3-70b-instruct", "fake-key", "https://integrate.api.nvidia.com/v1")
            self.assertEqual(len(updated), 1)
            self.assertAlmostEqual(updated[0]["adjustment"], 0.5)

            # Test end to end rerank
            res = llm_provider.nvidia_rerank(candidates, api_key="fake-key")
            self.assertIsNotNone(res)
            self.assertAlmostEqual(res[0]["overall"], 7.0)
            self.assertIn("exciting intro", res[0]["reason"])


class TestUnifiedRerank(unittest.TestCase):
    def test_heuristic_provider_returns_none(self):
        candidates = [{"start": 0.0, "end": 10.0, "duration": 10.0, "overall": 5.0, "seed_index": 0}]
        res, mode = llm_provider.rerank(candidates, provider="heuristic")
        self.assertIsNone(res)
        self.assertEqual(mode, "heuristic")

    def test_unknown_provider_fallback(self):
        candidates = [{"start": 0.0, "end": 10.0, "duration": 10.0, "overall": 5.0, "seed_index": 0}]
        res, mode = llm_provider.rerank(candidates, provider="unknown_ai_xyz")
        self.assertIsNone(res)
        self.assertEqual(mode, "heuristic")


class TestCloudflareTunnel(unittest.TestCase):
    def test_cloudflare_not_installed_error(self):
        with patch("shutil.which", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                dashboard.start_cloudflare_tunnel(8765)
            self.assertIn("cloudflared) is not installed", str(ctx.exception))

    def test_cloudflare_tunnel_success(self):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stderr.readline.side_effect = [
            "2026-11-10 INF Initializing...\n",
            "2026-11-10 INF Your quick Tunnel has been created! Visit it at: https://orange-cat-jump.trycloudflare.com\n",
        ]
        with patch("shutil.which", return_value="/usr/local/bin/cloudflared"), \
             patch("subprocess.Popen", return_value=mock_proc):
            proc, url = dashboard.start_cloudflare_tunnel(8765)
            self.assertEqual(url, "https://orange-cat-jump.trycloudflare.com")
            self.assertEqual(proc, mock_proc)


class TestDashboardRemoteAccess(unittest.TestCase):
    """Test dashboard HTTP server allowing any IP, remote hostnames, and LLM routes."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.server = dashboard.make_server(cls.temp_dir.name, port=0, host="127.0.0.1")
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_port
        cls.base_url = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.server.app.close()
        cls.thread.join()
        cls.temp_dir.cleanup()

    def test_get_with_external_host_header_is_allowed(self):
        # Formerly rejected with 403 "Use the 127.0.0.1 Studio address."
        req = urllib.request.Request(
            f"{self.base_url}/api/config",
            headers={"Host": "192.168.1.150:8765"},
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertIn("token", data)
            self.assertIn("llm_config", data)

    def test_get_with_tunnel_domain_is_allowed(self):
        req = urllib.request.Request(
            f"{self.base_url}/api/llm-config",
            headers={"Host": "orange-cat-jump.trycloudflare.com"},
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertIn("llm_provider", data)

    def test_post_llm_config_with_tunnel_origin_is_accepted(self):
        tunnel_host = "orange-cat-jump.trycloudflare.com"
        tunnel_origin = f"https://{tunnel_host}"
        body = json.dumps({"llm_provider": "gemini", "gemini_model": "gemini-3.8-flash"}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/llm-config",
            data=body,
            headers={
                "Host": tunnel_host,
                "Origin": tunnel_origin,
                "Content-Type": "application/json",
                "X-Chopify-Token": self.server.app.token,
            },
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            res = json.loads(resp.read().decode())
            self.assertTrue(res.get("ok"))
            self.assertEqual(res["config"]["gemini_model"], "gemini-3.8-flash")

    def test_cross_origin_attack_rejected(self):
        # Origin does not match Host -> 403
        body = json.dumps({"llm_provider": "gemini"}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/llm-config",
            data=body,
            headers={
                "Host": f"127.0.0.1:{self.port}",
                "Origin": "https://evil.attacker.com",
                "Content-Type": "application/json",
                "X-Chopify-Token": self.server.app.token,
            },
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 403)

    def test_get_available_videos(self):
        req = urllib.request.Request(f"{self.base_url}/api/available-videos")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertIn("directory", data)
            self.assertIn("videos", data)
            self.assertIsInstance(data["videos"], list)

    def test_post_and_get_cookies(self):
        cookie_text = "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tFALSE\t1735689600\tSID\tsecret123\n"
        body = json.dumps({"content": cookie_text}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/cookies",
            data=body,
            headers={
                "Host": f"127.0.0.1:{self.port}",
                "Origin": f"http://127.0.0.1:{self.port}",
                "Content-Type": "application/json",
                "X-Chopify-Token": self.server.app.token,
            },
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            res = json.loads(resp.read().decode())
            self.assertTrue(res.get("ok"))

        # Now GET /api/cookies
        get_req = urllib.request.Request(f"{self.base_url}/api/cookies")
        with urllib.request.urlopen(get_req) as resp:
            self.assertEqual(resp.status, 200)
            cdata = json.loads(resp.read().decode())
            self.assertTrue(cdata["exists"])
            self.assertGreater(cdata["size_bytes"], 0)


class TestColabUtils(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.vdir = Path(self.temp_dir.name) / "videos"
        self.vdir.mkdir(parents=True, exist_ok=True)
        self.orig_vdir_env = os.environ.get("CHOPIFY_VIDEOS_DIR")
        os.environ["CHOPIFY_VIDEOS_DIR"] = str(self.vdir)

    def tearDown(self):
        if self.orig_vdir_env is not None:
            os.environ["CHOPIFY_VIDEOS_DIR"] = self.orig_vdir_env
        else:
            os.environ.pop("CHOPIFY_VIDEOS_DIR", None)
        self.temp_dir.cleanup()

    def test_unique_video_path_numbering(self):
        # 1. No file exists yet
        p1 = colab_utils.unique_video_path(self.vdir, "my_video.mp4")
        self.assertEqual(p1.name, "my_video.mp4")
        p1.write_text("dummy 1")

        # 2. First collision -> _1 clause
        p2 = colab_utils.unique_video_path(self.vdir, "my_video.mp4")
        self.assertEqual(p2.name, "my_video_1.mp4")
        p2.write_text("dummy 2")

        # 3. Second collision -> _2 clause
        p3 = colab_utils.unique_video_path(self.vdir, "my_video.mp4")
        self.assertEqual(p3.name, "my_video_2.mp4")

    def test_list_available_videos(self):
        (self.vdir / "vid1.mp4").write_bytes(b"x" * 1024 * 1024)
        (self.vdir / "vid2.mov").write_bytes(b"x" * 2048 * 1024)
        (self.vdir / "readme.txt").write_text("not a video")

        res = colab_utils.list_available_videos()
        self.assertEqual(res["directory"], str(self.vdir))
        names = [v["name"] for v in res["videos"]]
        self.assertIn("vid1.mp4", names)
        self.assertIn("vid2.mov", names)
        self.assertNotIn("readme.txt", names)

    def test_cookies_management(self):
        cookie_path = Path(self.temp_dir.name) / "test_cookies.txt"
        cookie_content = "# Netscape HTTP Cookie File\n.google.com TRUE / FALSE 1234 SID abc"
        cookie_path.write_text(cookie_content)

        with patch.dict(os.environ, {"CHOPIFY_COOKIES": str(cookie_path)}):
            found = colab_utils.find_cookies_path()
            self.assertEqual(str(found), str(cookie_path.resolve()))
            args = colab_utils.yt_dlp_cookie_args()
            self.assertEqual(args, ["--cookies", str(cookie_path.resolve())])


class TestScoreClipsLLMIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workdir = Path(self.temp_dir.name)
        # Create minimal candidates.json
        candidates = [
            {"start": 1.0, "end": 20.0, "duration": 19.0, "overall": 7.0,
             "raw_score": 7.0, "text": "Clip one text", "seed_index": 0},
            {"start": 30.0, "end": 50.0, "duration": 20.0, "overall": 6.5,
             "raw_score": 6.5, "text": "Clip two text", "seed_index": 1},
        ]
        discovery = {"content_profile": "mixed", "evidence_available": False}
        (self.workdir / "candidates.json").write_text(
            json.dumps({"candidates": candidates, "stats": discovery}), encoding="utf-8"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_select_with_gemini_mock(self):
        import score_clips

        def mock_gemini_chat(sent, model, api_key, timeout=120):
            return [{"index": 0, "adjustment": 0.5, "reason": "strong hook"}]

        with patch("llm_provider._gemini_chat", side_effect=mock_gemini_chat), \
             patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            segments, mode = score_clips.select(self.workdir, min_score=0.0, llm_provider="gemini")
            self.assertEqual(mode, "gemini")
            self.assertAlmostEqual(segments[0]["overall"], 7.5)
            candidates = json.loads((self.workdir / "candidates.json").read_text())["candidates"]
            self.assertEqual(candidates[0]["llm_model"], "gemini-3.8-flash")

    def test_select_with_nvidia_mock(self):
        import score_clips

        def mock_nvidia_chat(sent, model, api_key, base_url, timeout=120):
            return [{"index": 0, "adjustment": 0.8, "reason": "viral moment"}]

        with patch("llm_provider._nvidia_chat", side_effect=mock_nvidia_chat), \
             patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
            segments, mode = score_clips.select(self.workdir, min_score=0.0, llm_provider="nvidia")
            self.assertEqual(mode, "nvidia")
            self.assertAlmostEqual(segments[0]["overall"], 7.8)
            candidates = json.loads((self.workdir / "candidates.json").read_text())["candidates"]
            self.assertEqual(candidates[0]["llm_model"], "meta/llama-3.3-70b-instruct")

    def test_select_with_ollama_backward_compat(self):
        import score_clips

        def mock_ollama_chat(model, host, messages, timeout=120):
            return [{"index": 0, "adjustment": 0.3, "reason": "ollama reaction"}]

        with patch.object(score_clips, "_ollama_chat", side_effect=mock_ollama_chat):
            segments, mode = score_clips.select(self.workdir, min_score=0.0, llm="qwen2.5:7b")
            self.assertEqual(mode, "ollama")
            self.assertAlmostEqual(segments[0]["overall"], 7.3)

    def test_select_fallback_on_failure(self):
        import score_clips
        with patch("llm_provider._gemini_chat", side_effect=RuntimeError("Connection refused")), \
             patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            segments, mode = score_clips.select(self.workdir, min_score=0.0, llm_provider="gemini")
            self.assertEqual(mode, "heuristic")
            self.assertAlmostEqual(segments[0]["overall"], 7.0)


if __name__ == "__main__":
    unittest.main()
