"""LLM side of the pipeline: W&B Inference (OpenAI-compatible) client, robust JSON parsing,
query expansion and caption judging — plus optional Weave tracing and W&B Artifacts.

weave / wandb are optional: every import and call is guarded, so the app runs without them.
"""
from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
import logging
import os
import re
import threading
import time
from typing import Any, Callable, Iterator

import httpx

log = logging.getLogger("ecm.llm")

DEFAULT_INFERENCE_URL = "https://api.inference.wandb.ai/v1"
# W&B Inference sits behind Cloudflare, which rejects some default client User-Agents.
USER_AGENT = "edge-case-miner/1.0"

# The weave/wandb SDKs ship Sentry error telemetry; keep it off unless explicitly enabled.
os.environ.setdefault("WANDB_ERROR_REPORTING", "false")
# MOCK=1 promises no network calls at all, so don't even import the tracing SDK then.
_OFFLINE = os.environ.get("MOCK", "").strip().lower() in ("1", "true", "yes", "on")

_weave = None
if not _OFFLINE:
    try:  # optional tracing
        import weave as _weave
    except Exception:  # noqa: BLE001 - any import problem just disables tracing
        _weave = None


def _identity(fn: Callable) -> Callable:
    return fn


#: Decorator for pipeline steps: ``weave.op`` when Weave is installed, otherwise a no-op.
op: Callable = _weave.op if _weave is not None else _identity
WEAVE_ENABLED = False


class LLMError(Exception):
    """LLM call failed; message is safe to show (no keys)."""


# --------------------------------------------------------------------------- Weave / W&B

def _scrub(value: Any) -> Any:
    """Keep playback URLs (they embed a JWT) and anything key-like out of Weave traces."""
    if isinstance(value, dict):
        return {k: ("<redacted>" if str(k).lower() in {"stream_url", "token", "access_token", "password",
                                                         "api_key", "authorization"} else _scrub(v))
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(v) for v in value]
    if isinstance(value, str) and "token=" in value:
        return re.sub(r"token=[^&\s\"']+", "token=<redacted>", value)
    return value


def init_weave(entity: str, project: str, api_key: str) -> bool:
    """weave.init("<entity>/<project>") if weave is installed and a key is set. Never raises."""
    global WEAVE_ENABLED
    if _weave is None or not api_key:
        return False
    name = f"{entity}/{project}" if entity and project else (project or "edge-case-miner")
    hooks = ({"postprocess_inputs": _scrub, "postprocess_output": _scrub},
             {"global_postprocess_inputs": _scrub, "global_postprocess_output": _scrub},
             {})
    for kwargs in hooks:
        try:
            _weave.init(name, **kwargs)
        except TypeError:  # older/newer weave without these hook names
            continue
        except Exception as exc:  # noqa: BLE001
            log.warning("Weave init failed (%s); tracing disabled", type(exc).__name__)
            return False
        WEAVE_ENABLED = True
        log.info("Weave tracing enabled for project %s", name)
        return True
    return False


_WANDB_LOCK = threading.Lock()


def wandb_installed() -> bool:
    try:
        return importlib.util.find_spec("wandb") is not None
    except Exception:  # noqa: BLE001
        return False


def log_dataset_artifact(manifest_path: str, name: str, metadata: dict, entity: str, project: str,
                         api_key: str) -> str | None:
    """Version a manifest as a W&B Artifact (type "dataset"). Returns the run URL, or None. Never raises."""
    if not api_key or not wandb_installed():
        return None
    try:
        os.environ.setdefault("WANDB_SILENT", "true")
        import wandb  # noqa: PLC0415 - optional, slow import

        with _WANDB_LOCK:  # one run at a time; wandb runs are process-global
            kwargs: dict[str, Any] = {
                "entity": entity or None,
                "project": project or "edge-case-miner",
                "job_type": "edge-case-export",
                "name": f"export-{name}"[:120],
                "config": {"scenario": metadata.get("scenario"), "counts": metadata.get("counts")},
                "reinit": True,
            }
            try:
                kwargs["settings"] = wandb.Settings(silent=True, console="off", init_timeout=90)
            except Exception:  # noqa: BLE001 - settings fields vary between versions
                pass
            run = wandb.init(**kwargs)
            try:
                artifact = wandb.Artifact(name=name, type="dataset", metadata=metadata)
                artifact.add_file(manifest_path, name="manifest.json")
                run.log_artifact(artifact)
                url = getattr(run, "url", None)
            finally:
                run.finish()
            return url
    except Exception as exc:  # noqa: BLE001
        log.warning("W&B artifact export failed (%s)", type(exc).__name__)
        return None


# --------------------------------------------------------------------------- robust parsing

_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)


def strip_think(text: str) -> str:
    """Drop <think>…</think> reasoning and ``` fences."""
    text = _THINK_RE.sub(" ", text or "")
    low = text.lower()
    if "</think>" in low:  # reasoning without an opening tag
        text = text[low.rfind("</think>") + len("</think>"):]
    elif "<think>" in low:  # truncated reasoning: keep what follows the tag
        text = text[low.find("<think>") + len("<think>"):]
    text = re.sub(r"```[a-zA-Z]*", " ", text)
    return text.strip()


def _balanced_objects(text: str) -> Iterator[str]:
    """Yield balanced {...} substrings, ignoring braces inside double-quoted strings."""
    depth, start, in_str, escaped = 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"' and depth > 0:
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                yield text[start:i + 1]


def _loose_load(block: str) -> Any:
    """Python-literal style objects ({'match': True}) and JSON with true/false/null."""
    fixed = re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", re.sub(r"\bnull\b", "None", block)))
    return ast.literal_eval(fixed)


def parse_json_object(text: str) -> dict | None:
    """First JSON object in a model reply (after stripping think blocks and fences)."""
    cleaned = strip_think(text)
    if len(cleaned) > 20000:
        cleaned = cleaned[:20000]
    for block in _balanced_objects(cleaned):
        for loader in (json.loads, _loose_load):
            try:
                obj = loader(block)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(obj, dict):
                return obj
    return None


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "yes", "y", "match", "matches", "1"}:
            return True
        if s in {"false", "no", "n", "no match", "no-match", "nomatch", "0", "none"}:
            return False
    return None


def _to_conf(value: Any) -> float | None:
    try:
        f = float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    if f > 1.0:
        f /= 100.0
    return round(max(0.0, min(1.0, f)), 3)


def _clip(text: Any, n: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


_MATCH_RE = re.compile(r"[\"']?\bmatch\b[\"']?\s*[:=]\s*[\"']?(true|false|yes|no)\b", re.I)


def parse_verdict(text: str) -> dict:
    """Parse the {"match","confidence","why"} contract. Never raises; match=None if unparseable."""
    obj = parse_json_object(text)
    if obj is not None:
        match = _to_bool(obj.get("match", obj.get("is_match", obj.get("answer"))))
        if match is not None:
            why = obj.get("why") or obj.get("reason") or obj.get("explanation") or ""
            return {"match": match, "confidence": _to_conf(obj.get("confidence")), "why": _clip(why, 300)}
    cleaned = strip_think(text)
    found = _MATCH_RE.search(cleaned) or re.match(r"\s*(yes|no)\b", cleaned, re.I)
    if found:
        return {"match": found.group(1).lower() in ("true", "yes"), "confidence": None,
                "why": _clip(cleaned, 300)}
    return {"match": None, "confidence": None, "why": _clip(cleaned or "empty reply", 300)}


# --------------------------------------------------------------------------- query expansion

_STOP = {"a", "an", "the", "of", "in", "on", "at", "or", "and", "to", "with", "by", "for", "from",
         "into", "is", "are", "be", "someone", "something", "while", "next", "close"}
_REPHRASE = {"pedestrian": "person walking", "pedestrians": "people walking", "vehicle": "car",
             "vehicles": "cars", "car": "vehicle", "cars": "vehicles", "cyclist": "person on a bicycle",
             "truck": "semi truck", "road": "street", "walkway": "path", "aisle": "corridor",
             "corridor": "hallway", "doorway": "entrance", "stopping": "halting", "people": "persons",
             "crowding": "gathering in", "moving": "driving", "blocked": "obstructed"}


def _dedupe(items: Iterator[str] | list[str]) -> list[str]:
    seen, out = set(), []
    for item in items:
        key = " ".join(item.lower().split())
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def fallback_queries(scenario: str) -> list[str]:
    """Template rephrasings used when the LLM is unavailable: [scenario] + 2–3 variants."""
    s = " ".join(scenario.split()).strip(" .")
    swapped = " ".join(_REPHRASE.get(w.lower(), w) for w in s.split())
    keywords = " ".join(w for w in re.findall(r"[A-Za-z][A-Za-z\-]+", s) if w.lower() not in _STOP)
    return _dedupe([s, swapped, f"video clip showing {s}", keywords])[:4]


def _queries_from_text(text: str) -> list[str]:
    obj = parse_json_object(text)
    if obj is not None:
        queries = obj.get("queries") or obj.get("search_queries") or []
        return [str(q) for q in queries] if isinstance(queries, list) else []
    cleaned = strip_think(text)
    match = re.search(r"\[[\s\S]*?\]", cleaned)
    if match:
        try:
            arr = json.loads(match.group(0))
            if isinstance(arr, list):
                return [str(q) for q in arr]
        except ValueError:
            pass
    return [ln for ln in cleaned.splitlines() if ln.strip()]


def _clean_queries(raw: list[str], n: int) -> list[str]:
    out = []
    for q in raw:
        q = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", str(q)).strip().strip("\"'").strip()
        q = " ".join(q.split()).rstrip(".")
        if 3 <= len(q) <= 160:
            out.append(q)
    return _dedupe(out)[:n]


EXPAND_SYSTEM = (
    "You write search queries for a video archive used by autonomous-vehicle and robotics engineers. "
    "Every ~5 second clip from highway cameras, dashcams, street cameras, warehouse ceiling cameras and "
    "indoor cameras was captioned by a vision-language model in plain descriptive prose. Write queries the "
    "way that captioner describes scenes: concrete, visual, present tense, 5-12 words, naming the actors, "
    "objects and the action. No camera jargon, no hashtags, no explanations."
)

JUDGE_SYSTEM = (
    "You check whether a video clip matches a requested scenario using only the caption a video model "
    "wrote for the clip. Be strict: the scenario's key actors and key action must be explicitly described. "
    "If the caption does not mention them, answer false."
)


# Preferred W&B Inference model per job, matched case-insensitively against the ids GET /models returns:
# Qwen 3.8 writes the search queries (fast, reliable JSON), DeepSeek V4 judges captions when Cosmos can't.
MODEL_PREFS: dict[str, tuple[str, ...]] = {
    "expand": ("qwen3.8", "deepseek-v4", "qwen3", "nemotron"),
    "judge": ("deepseek-v4-pro", "deepseek-v4", "qwen3.8", "qwen3", "nemotron"),
}


def pick_model(ids: list[str], prefs: tuple[str, ...] = MODEL_PREFS["expand"]) -> str | None:
    """First chat-capable id matching the earliest preference, else the first chat-capable id."""
    usable = [i for i in ids if not re.search(r"embed|rerank|reward|guard|safety|whisper|tts", i, re.I)] or ids
    for pref in prefs:
        for model_id in usable:
            if pref in model_id.lower():
                return model_id
    return usable[0] if usable else None


# Retried once when the preferred expansion model answers with nothing usable: models that don't think.
BACKUP_PREFS = ("qwen3-30b-a3b-instruct", "deepseek-v4-flash", "deepseek-v4.1-flash", "llama-3.3-70b")


def _hybrid_thinker(model_id: str) -> bool:
    """Models that think by default and honour a "/no_think" system-prompt switch."""
    return bool(re.search(r"nemotron|qwen3", model_id, re.I))


class LLMClient:
    """Minimal async client for W&B Inference's OpenAI-compatible API."""

    def __init__(self, api_key: str = "", base_url: str = DEFAULT_INFERENCE_URL, entity: str = "",
                 project: str = "", model: str = "", timeout: float = 60.0,
                 expand_model: str = "", judge_model: str = ""):
        self._key = api_key or ""
        self.base = (base_url or DEFAULT_INFERENCE_URL).strip().rstrip("/")
        self.entity = entity or ""
        self.project = project or ""
        # Per-job pins (LLM_EXPAND_MODEL / LLM_JUDGE_MODEL, or LLM_MODEL for both); unpinned jobs are
        # resolved from the live model list.
        self._models: dict[str, str | None] = {"expand": expand_model or model or None,
                                               "judge": judge_model or model or None}
        self._ids: list[str] = []
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._lock: asyncio.Lock | None = None
        self._retry_at = 0.0
        self.last_error: str | None = None

    def __repr__(self) -> str:  # never show the key
        return f"LLMClient(configured={self.configured}, models={self._models!r})"

    @property
    def configured(self) -> bool:
        return bool(self._key)

    @property
    def model_name(self) -> str | None:
        """The query-expansion model (the one the UI shows)."""
        return self._models["expand"]

    @property
    def judge_model_name(self) -> str | None:
        return self._models["judge"]

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self._timeout, connect=15.0),
                                             headers={"User-Agent": USER_AGENT})
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _headers(self) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self._key}"}
        if self.entity and self.project:
            headers["OpenAI-Project"] = f"{self.entity}/{self.project}"
        return headers

    async def model(self, role: str = "expand") -> str | None:
        """The pinned model for ``role``, else one picked from GET {base}/models (retried once a minute)."""
        if self._models.get(role) or not self.configured:
            return self._models.get(role)
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._models.get(role) or time.monotonic() < self._retry_at:
                return self._models.get(role)
            self._retry_at = time.monotonic() + 60
            try:
                resp = await self._http().get(f"{self.base}/models", headers=self._headers())
                if resp.status_code != 200:
                    self.last_error = f"model list failed (HTTP {resp.status_code})"
                    return None
                data = resp.json().get("data") or []
                ids = [str(m["id"]) for m in data if isinstance(m, dict) and m.get("id")]
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"model list failed ({type(exc).__name__})"
                return None
            self._ids = ids
            for job, prefs in MODEL_PREFS.items():
                self._models[job] = self._models.get(job) or pick_model(ids, prefs)
            self.last_error = None if self._models.get(role) else "no models available"
            return self._models.get(role)

    async def chat(self, messages: list[dict], max_tokens: int = 600, temperature: float = 0.2,
                   role: str = "expand", model: str | None = None) -> str:
        model = model or await self.model(role)
        if not model:
            raise LLMError(self.last_error or "LLM not configured")
        body: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "temperature": temperature}
        if _hybrid_thinker(model):
            # Qwen3 / Nemotron think by default, which can eat the whole budget before any answer:
            # turn it off both ways ("/no_think" switch and the vLLM chat-template flag).
            if messages and messages[0].get("role") == "system":
                messages = [{"role": "system", "content": "/no_think\n" + messages[0]["content"]}] + messages[1:]
            body["chat_template_kwargs"] = {"enable_thinking": False}
        body["messages"] = messages
        resp = await self._post_chat(body)
        if resp.status_code in (400, 422) and "chat_template_kwargs" in body:
            body.pop("chat_template_kwargs")  # a server that rejects the flag: rely on "/no_think"
            resp = await self._post_chat(body)
        if resp.status_code != 200:
            raise LLMError(f"LLM HTTP {resp.status_code}: {_clip(resp.text, 160)}")
        try:
            choice = resp.json()["choices"][0]
            message = choice["message"]
        except Exception:  # noqa: BLE001
            raise LLMError("LLM returned an unexpected response shape") from None
        content = (message.get("content") or "").strip()
        reasoning = (message.get("reasoning_content") or message.get("reasoning") or "").strip()
        log.info("llm %s: finish=%s content=%d chars reasoning=%d chars", model, choice.get("finish_reason"),
                 len(content), len(reasoning))
        if not content:
            # Thinking is not an answer (it would parse as junk queries): let the caller fall back.
            raise LLMError(f"{model} gave no answer (finish={choice.get('finish_reason')}, "
                           f"{len(reasoning)} chars of reasoning)")
        return content

    async def _post_chat(self, body: dict) -> httpx.Response:
        try:
            return await self._http().post(f"{self.base}/chat/completions", json=body, headers=self._headers())
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed ({type(exc).__name__})") from None

    def backup_model(self, role: str) -> str | None:
        """A non-thinking model to retry with when the preferred one returns nothing usable."""
        current = self._models.get(role)
        for pref in BACKUP_PREFS:
            for model_id in self._ids:
                if pref in model_id.lower() and model_id != current:
                    return model_id
        return None

    async def expand_queries(self, scenario: str, n: int = 4) -> dict:
        """Scenario → n caption-style search queries. Falls back to templates on any failure."""
        error = "WANDB_API_KEY not set"
        if self.configured:
            prompt = (f'Edge-case scenario: "{scenario}"\n'
                      f'Return ONLY JSON: {{"queries": [{n} different search queries]}}. '
                      "Cover different phrasings and viewpoints (dashcam, street camera, overhead, indoor) "
                      "where they make sense.")
            messages = [{"role": "system", "content": EXPAND_SYSTEM}, {"role": "user", "content": prompt}]
            await self.model("expand")
            for model in dict.fromkeys(m for m in (self.model_name, self.backup_model("expand")) if m):
                try:
                    text = await self.chat(messages, max_tokens=1500, temperature=0.3, model=model)
                    queries = _clean_queries(_queries_from_text(text), n)
                    if queries:
                        return {"queries": queries, "model": model, "fallback": False}
                    error = f"{model} reply contained no queries"
                except LLMError as exc:
                    error = str(exc)
                log.info("query expansion: %s", error)
        return {"queries": fallback_queries(scenario), "model": None, "fallback": True, "error": error}

    async def judge_caption(self, scenario: str, caption: str) -> dict:
        """Judge the ingest caption against the scenario (same JSON contract as the video judge)."""
        prompt = (f'Scenario: "{scenario}"\n'
                  f'Caption of a ~5 second clip: """{_clip(caption, 1500)}"""\n'
                  'Reply ONLY with JSON: {"match": true|false, "confidence": 0..1, '
                  '"why": "<one sentence citing the caption>"}')
        text = await self.chat([{"role": "system", "content": JUDGE_SYSTEM},
                                {"role": "user", "content": prompt}], max_tokens=1200, temperature=0.0,
                               role="judge")
        verdict = parse_verdict(text)
        verdict.update(method="caption", model=self.judge_model_name)
        return verdict
