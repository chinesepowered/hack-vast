"""Cosmos3-Reason clip verification.

For each (scenario, clip): download the ~5 s segment from VSS, shrink it to 480p / 8 fps H.264
(imageio-ffmpeg, optional), and ask Cosmos3-Reason to *watch* it and answer a strict JSON rubric.
When Cosmos is unconfigured, fails, or the clip is too big, the ingest caption is judged by the
W&B LLM instead (method "caption"). Verdicts are cached on disk keyed by sha1(source + scenario).
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

import llm
from vss import VSSError, redact

log = logging.getLogger("ecm.cosmos")

RAW_SEND_LIMIT = 15 * 1024 * 1024  # send an untranscoded clip only below this size
FFMPEG_TIMEOUT = 60
VERDICT_FIELDS = ("match", "confidence", "why", "method", "model", "note", "clip")

RUBRIC = (
    "You are verifying training data for autonomous-vehicle and robotics engineers.\n"
    'Watch this ~5 second video clip. Does it clearly show the scenario: "{scenario}"?\n'
    "Judge only what is visible in the clip itself. If the key actors or the key action are missing, "
    "or you are unsure, answer false. Keep any reasoning brief.\n"
    'Reply ONLY with JSON and no other text: {{"match": true|false, "confidence": 0..1, '
    '"why": "<one sentence citing what is visible>"}}'
)
# Token budgets: a short answer first; if a reasoning model runs out of tokens while still thinking
# (finish_reason "length", no verdict yet), retry once with room to finish.
TOKEN_BUDGETS = (400, 1500)


class CosmosError(Exception):
    """Cosmos call failed; message is safe to show."""


def _clean_base(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    if url and "://" not in url:
        url = "http://" + url
    return url


class CosmosClient:
    """OpenAI-style chat client for the shared Cosmos3-Reason endpoint."""

    def __init__(self, url: str = "", token: str = "", model: str = "", timeout: float = 150.0):
        self.base = _clean_base(url)
        self._token = token or ""
        self._model = model or None
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._lock: asyncio.Lock | None = None
        self._retry_at = 0.0
        self.last_error: str | None = None

    def __repr__(self) -> str:  # never show the bearer token
        return f"CosmosClient(configured={self.configured}, model={self._model!r})"

    @property
    def configured(self) -> bool:
        return bool(self.base)

    @property
    def model_name(self) -> str | None:
        return self._model

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self._timeout, connect=15.0),
                                             headers={"User-Agent": llm.USER_AGENT})
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"} if self._token else {}

    async def model(self) -> str | None:
        """COSMOS3_REASON_MODEL, or GET {url}/v1/models → data[0].id (retried at most once a minute)."""
        if self._model or not self.configured:
            return self._model
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._model or time.monotonic() < self._retry_at:
                return self._model
            self._retry_at = time.monotonic() + 60
            try:
                resp = await self._http().get(f"{self.base}/v1/models", headers=self._headers())
                if resp.status_code != 200:
                    self.last_error = f"model list failed (HTTP {resp.status_code})"
                    return None
                data = resp.json().get("data") or []
                self._model = str(data[0]["id"]) if data and isinstance(data[0], dict) else None
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"model list failed ({type(exc).__name__})"
                return None
            self.last_error = None if self._model else "no model listed"
            return self._model

    async def judge_video(self, scenario: str, clip: bytes) -> dict:
        """Ask Cosmos3-Reason whether the clip shows the scenario. Raises CosmosError on failure."""
        model = await self.model()
        if not model:
            raise CosmosError(self.last_error or "model unknown")
        payload: dict[str, Any] = {
            "model": model,
            "temperature": 0,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": RUBRIC.format(scenario=scenario.replace('"', "'"))},
                    {"type": "video_url",
                     "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(clip).decode("ascii")}},
                ],
            }],
        }
        t0 = time.perf_counter()
        verdict: dict = {"match": None, "confidence": None, "why": "empty reply"}
        for budget in TOKEN_BUDGETS:
            payload["max_tokens"] = budget
            try:
                resp = await self._http().post(f"{self.base}/v1/chat/completions", json=payload,
                                               headers=self._headers())
            except httpx.TimeoutException:
                raise CosmosError("Cosmos3-Reason timed out") from None
            except httpx.HTTPError as exc:
                raise CosmosError(f"Cosmos3-Reason request failed ({type(exc).__name__})") from None
            if resp.status_code != 200:
                raise CosmosError(f"Cosmos3-Reason HTTP {resp.status_code}: {redact(resp.text)[:160]}")
            try:
                choice = resp.json()["choices"][0]
                message = choice["message"]
            except Exception:  # noqa: BLE001
                raise CosmosError("Cosmos3-Reason returned an unexpected response shape") from None
            text = message.get("content") or ""
            truncated = choice.get("finish_reason") == "length"
            if not text.strip() and not truncated:  # some servers put the whole answer in reasoning_content
                text = message.get("reasoning_content") or ""
            verdict = llm.parse_verdict(text)
            if verdict["match"] is not None or not truncated:
                break
        verdict.update(method="cosmos-video", model=model, ms=int((time.perf_counter() - t0) * 1000))
        return verdict


# --------------------------------------------------------------------------- clip shrinking

_FFMPEG: Any = False  # False = not looked up yet; None = unavailable


def ffmpeg_exe() -> str | None:
    """ffmpeg bundled with imageio-ffmpeg (optional package), else a system ffmpeg, else None."""
    global _FFMPEG
    if _FFMPEG is False:
        try:
            import imageio_ffmpeg  # noqa: PLC0415 - optional

            _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001
            _FFMPEG = shutil.which("ffmpeg")
    return _FFMPEG


async def shrink_clip(data: bytes, workdir: Path) -> tuple[bytes, str]:
    """Transcode to small H.264 (≤480p, 8 fps, no audio). Returns (bytes, description)."""
    exe = ffmpeg_exe()
    if not exe:
        return data, "original"
    workdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        src, dst = Path(tmp, "in.mp4"), Path(tmp, "out.mp4")
        src.write_bytes(data)
        proc = await asyncio.create_subprocess_exec(
            exe, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-t", "12", "-an",
            "-vf", "scale=-2:trunc(min(480\\,ih)/2)*2,fps=8", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "30", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst),
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE)
        try:
            _, err = await asyncio.wait_for(proc.communicate(), FFMPEG_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            log.warning("ffmpeg timed out; sending the original clip")
            return data, "original"
        if proc.returncode != 0 or not dst.is_file() or dst.stat().st_size == 0:
            log.warning("ffmpeg failed (rc=%s): %s", proc.returncode, (err or b"").decode(errors="ignore")[-200:])
            return data, "original"
        return dst.read_bytes(), "h264 480p 8fps"


# --------------------------------------------------------------------------- verdict cache

def norm_scenario(text: str) -> str:
    return " ".join((text or "").lower().split())


class VerdictCache:
    """One JSON file per (source, scenario) under ``directory``; mirrored in memory.

    Picks up files written by other processes (e.g. ``python main.py --warm``) lazily.
    """

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, dict] = {}
        self._dir_mtime = -1
        self._sync()

    @staticmethod
    def key(source: str, scenario: str) -> str:
        return hashlib.sha1(f"{source}\n{norm_scenario(scenario)}".encode()).hexdigest()

    def _sync(self) -> None:
        try:
            mtime = self.dir.stat().st_mtime_ns
        except OSError:
            return
        if mtime == self._dir_mtime:
            return
        self._dir_mtime = mtime
        for path in self.dir.glob("*.json"):
            if path.stem in self._mem:
                continue
            try:
                entry = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(entry, dict) and entry.get("source"):
                self._mem[path.stem] = entry

    def get(self, source: str, scenario: str) -> dict | None:
        self._sync()
        return self._mem.get(self.key(source, scenario))

    def put(self, source: str, scenario: str, verdict: dict, extra: dict | None = None) -> dict:
        entry = {"source": source, "scenario": norm_scenario(scenario),
                 **{k: verdict.get(k) for k in VERDICT_FIELDS if verdict.get(k) is not None},
                 "match": verdict.get("match"), **(extra or {}), "ts": int(time.time())}
        key = self.key(source, scenario)
        tmp = self.dir / f".{key}.{os.getpid()}.tmp"
        try:
            tmp.write_text(json.dumps(entry))
            os.replace(tmp, self.dir / f"{key}.json")
        except OSError as exc:
            log.warning("could not persist verdict (%s)", type(exc).__name__)
        self._mem[key] = entry
        return entry

    def for_scenario(self, scenario: str) -> list[dict]:
        self._sync()
        wanted = norm_scenario(scenario)
        return [e for e in self._mem.values() if e.get("scenario") == wanted]

    def __len__(self) -> int:
        return len(self._mem)


def public_verdict(v: dict) -> dict:
    out = {k: v.get(k) for k in ("match", "confidence", "why", "method")}
    for k in ("model", "note", "clip", "ms", "cached"):
        if v.get(k) is not None:
            out[k] = v[k]
    return out


# --------------------------------------------------------------------------- orchestration

class Verifier:
    """cache → Cosmos3-Reason on the video → caption judge (W&B LLM) → "no verifier"."""

    def __init__(self, cache: VerdictCache, cosmos: CosmosClient, llm_client: llm.LLMClient, vss_client: Any,
                 lookup: Callable[[str], Awaitable[dict | None]], concurrency: int = 6,
                 workdir: Path = Path(tempfile.gettempdir()),
                 mock_judge: Callable[[str, str], Awaitable[dict]] | None = None):
        self.cache = cache
        self.cosmos = cosmos
        self.llm = llm_client
        self.vss = vss_client
        self._lookup = lookup
        self._sem = asyncio.Semaphore(max(1, concurrency))
        self.workdir = workdir
        self._mock_judge = mock_judge
        self._inflight: dict[str, asyncio.Task] = {}

    async def verify(self, scenario: str, source: str) -> dict:
        cached = self.cache.get(source, scenario)
        if cached:
            return public_verdict({**cached, "cached": True})
        key = VerdictCache.key(source, scenario)
        task = self._inflight.get(key)
        if task is None:  # de-duplicate concurrent requests for the same clip + scenario
            task = asyncio.ensure_future(self._verify_uncached(scenario, source))
            self._inflight[key] = task
            task.add_done_callback(lambda _t, k=key: self._inflight.pop(k, None))
        return await asyncio.shield(task)

    async def _verify_uncached(self, scenario: str, source: str) -> dict:
        async with self._sem:
            t0 = time.perf_counter()
            hit = await self._lookup(source) or {}
            try:
                verdict = await self._judge(scenario, source, hit)
            except Exception as exc:  # noqa: BLE001 - a verdict request must never 500
                log.warning("verification error for a clip (%s)", type(exc).__name__)
                verdict = {"match": None, "confidence": None, "why": f"verification error ({type(exc).__name__})",
                           "method": "error"}
            verdict.setdefault("ms", int((time.perf_counter() - t0) * 1000))
        # Cache real verdicts only: a caption-judge fallback caused by a transient Cosmos/clip failure
        # must not stick, so the next request gets Cosmos to watch the clip after all.
        if verdict.get("match") is not None and "Cosmos3-Reason unavailable" not in (verdict.get("note") or ""):
            self.cache.put(source, scenario, verdict,
                           {"location": hit.get("location"), "camera_id": hit.get("camera_id")})
        return public_verdict(verdict)

    async def _judge(self, scenario: str, source: str, hit: dict) -> dict:
        if self._mock_judge is not None:
            return await self._mock_judge(scenario, source)
        caption = (hit.get("caption") or "").strip()
        notes: list[str] = []
        cosmos_reply: dict | None = None
        if self.cosmos.configured:
            try:
                raw = await self.vss.download_clip(source)
                clip, how = await shrink_clip(raw, self.workdir)
                if len(clip) > RAW_SEND_LIMIT:
                    raise CosmosError(f"clip too large to send ({len(clip) // 1048576} MB)")
                verdict = await self.cosmos.judge_video(scenario, clip)
                verdict["clip"] = f"{how}, {max(1, len(clip) // 1024)} KB"
                if verdict.get("match") is not None:
                    return verdict
                cosmos_reply = verdict
                notes.append("Cosmos3-Reason reply was not valid JSON")
            except (CosmosError, VSSError) as exc:
                notes.append(f"Cosmos3-Reason unavailable: {getattr(exc, 'message', None) or exc}")
        if self.llm.configured and caption:
            try:
                verdict = await self.llm.judge_caption(scenario, caption)
                if notes:
                    verdict["note"] = "; ".join(notes)
                if verdict.get("match") is not None or cosmos_reply is None:
                    return verdict
            except llm.LLMError as exc:
                notes.append(f"caption judge failed: {exc}")
        if cosmos_reply is not None:
            cosmos_reply["note"] = "; ".join(notes)
            return cosmos_reply
        if notes:
            reason = "; ".join(notes)
        elif self.llm.configured:
            reason = "no caption available to judge"
        else:
            reason = "no verifier configured (set COSMOS3_REASON_URL + GPU_BEARER_TOKEN, or WANDB_API_KEY)"
        return {"match": None, "confidence": None, "why": reason, "method": "none"}
