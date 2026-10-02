"""Client for the team's VSS retrieval backend: login, hybrid search, playback URLs, clip bytes.

Live /api/v1/search rows are flat: ``source``, ``original_video``, ``similarity_score``,
``reasoning_content``, ``segment_start_sec``/``segment_end_sec``, ``camera_id``/``location``/
``capture_type``, ``object_classes`` ("bench,car,person") and ``object_counts`` (a JSON *string*,
'{"car": 5, "person": 1}'). ``chunk_results[].timeline`` lists every 5 s segment of the parent
chunk, which gives cheap before/after context. Parsing stays defensive anyway: alternative key
names are tried, unknown values stay ``None``, and a trimmed ``raw`` copy of each row is kept for
the UI's debug panel.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Iterable
from urllib.parse import quote

import httpx

log = logging.getLogger("ecm.vss")

MAX_CLIP_BYTES = 40 * 1024 * 1024
USER_AGENT = "edge-case-miner/1.0"

# Candidate key names, most likely first.
SOURCE_KEYS = ("source", "segment_source", "clip_source", "s3_uri", "uri", "preview_source")
ORIGINAL_KEYS = ("original_video", "parent_video", "original_source", "video_uri", "video")
SIM_KEYS = ("similarity_score", "similarity", "score", "hybrid_score", "relevance")
CAPTION_KEYS = ("reasoning_content", "caption", "description", "reasoning", "summary", "text")
START_KEYS = ("segment_start_sec", "start_sec", "start_time", "segment_start", "clip_start",
              "start_seconds", "best_match_start_sec", "start")
END_KEYS = ("segment_end_sec", "end_sec", "end_time", "segment_end", "clip_end",
            "end_seconds", "best_match_end_sec", "end")
NESTED_META_KEYS = ("metadata", "upload_metadata", "extra_metadata", "stream_metadata", "meta")
RAW_OMIT = {"vectors", "vectors_visual", "embedding", "embeddings", "perception_json", "timeline"}

_SECRET_RES = (
    re.compile(r"(?i)(token=|bearer\s+|access_token[\"']?\s*[:=]\s*[\"']?|password[\"']?\s*[:=]\s*[\"']?)"
               r"[^\s&\"',}]+"),
    re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]+"),  # bare JWTs
)


def redact(text: Any) -> str:
    """Strip JWTs, bearer tokens and passwords from text bound for logs or API errors."""
    out = str(text or "")
    out = _SECRET_RES[0].sub(lambda m: m.group(1) + "***", out)
    return _SECRET_RES[1].sub("***", out)


class VSSError(Exception):
    """A backend call failed. ``message`` is safe to show to users (no secrets, no hosts)."""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.message = message
        self.status = status


# --------------------------------------------------------------------------- normalizing

def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _as_dict(value: Any) -> dict | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            parsed = json.loads(value)
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _first(d: dict, keys: Iterable[str]) -> Any:
    for key in keys:
        value = d.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _meta(d: dict, field: str) -> str | None:
    """A metadata field at top level, or nested under metadata/upload_metadata/extra_metadata."""
    value = d.get(field)
    if value not in (None, ""):
        return str(value)
    for key in NESTED_META_KEYS:
        nested = _as_dict(d.get(key))
        if nested and nested.get(field) not in (None, ""):
            return str(nested[field])
    return None


def _objects(value: Any) -> list[str]:
    if isinstance(value, str):
        parsed = _as_dict(value)
        items: Iterable[Any] = parsed.keys() if parsed else value.replace(";", ",").split(",")
    elif isinstance(value, dict):
        items = value.keys()
    elif isinstance(value, list):
        items = value
    else:
        return []
    out: list[str] = []
    for item in items:
        name = str(item.get("label") or item.get("name") or "") if isinstance(item, dict) else str(item)
        name = name.strip().lower()
        if name and name not in out:
            out.append(name)
    return out[:8]


def _counts(value: Any) -> dict[str, int]:
    """YOLO object_counts: a JSON string ('{"car": 5, "person": 1}') or dict → {label: count}."""
    parsed = _as_dict(value) or {}
    out: dict[str, int] = {}
    for label, count in parsed.items():
        n = _num(count)
        if label and n is not None and n > 0:
            out[str(label).strip().lower()] = int(n)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def trim_raw(value: Any, depth: int = 0) -> Any:
    """A small, JSON-safe copy of a backend row for the debug panel."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for i, (key, item) in enumerate(value.items()):
            if i >= 40:
                out["..."] = f"{len(value) - 40} more keys"
                break
            if key in RAW_OMIT:
                out[key] = f"<{type(item).__name__} omitted>"
            else:
                out[key] = trim_raw(item, depth + 1) if depth < 3 else "<nested>"
        return out
    if isinstance(value, list):
        items = [trim_raw(item, depth + 1) for item in value[:8]]
        if len(value) > 8:
            items.append(f"... {len(value) - 8} more")
        return items
    if isinstance(value, str) and len(value) > 400:
        return value[:400] + "..."
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def normalize_hit(row: Any) -> dict | None:
    """Map one backend row (segment hit, chunk hit or metadata row) to our candidate shape."""
    if not isinstance(row, dict):
        return None
    source = _first(row, SOURCE_KEYS)
    if not isinstance(source, str) or not source.strip():
        return None
    sim = _num(_first(row, SIM_KEYS))
    if sim is None and _num(row.get("distance")) is not None:
        sim = 1.0 - float(row["distance"])
    start = _num(_first(row, START_KEYS))
    end = _num(_first(row, END_KEYS))
    if start is None and _num(row.get("start_ms")) is not None:
        start = float(row["start_ms"]) / 1000.0
    if end is None and _num(row.get("end_ms")) is not None:
        end = float(row["end_ms"]) / 1000.0
    seg_no = _num(row.get("segment_number"))
    original = _first(row, ORIGINAL_KEYS)
    caption = _first(row, CAPTION_KEYS)
    counts = _counts(row.get("object_counts"))
    return {
        "source": source.strip(),
        "original_video": str(original) if original is not None else None,
        "filename": str(row["filename"]) if row.get("filename") else None,
        "camera_id": _meta(row, "camera_id"),
        "location": _meta(row, "location"),
        "capture_type": _meta(row, "capture_type"),
        "start_sec": start,
        "end_sec": end,
        "segment_number": int(seg_no) if seg_no is not None else None,
        "similarity": round(sim, 4) if sim is not None else None,
        "caption": str(caption).strip() if caption is not None else "",
        "objects": _objects(row.get("object_classes")) or list(counts)[:8],
        "object_counts": counts,
        "raw": trim_raw(row),
    }


def _context_segment(seg: dict | None) -> dict | None:
    if not seg:
        return None
    return {"source": str(seg["source"]), "start_sec": _num(_first(seg, START_KEYS)),
            "end_sec": _num(_first(seg, END_KEYS)), "caption": str(_first(seg, CAPTION_KEYS) or "")[:600],
            "object_counts": _counts(seg.get("object_counts"))}


def attach_context(hits: list[dict], chunks: Any) -> None:
    """Give each hit its neighbouring 5 s segments (before/after) from chunk_results[].timeline."""
    neighbours: dict[str, tuple[dict | None, dict | None]] = {}
    for chunk in chunks if isinstance(chunks, list) else []:
        timeline = chunk.get("timeline") if isinstance(chunk, dict) else None
        if not isinstance(timeline, list):
            continue
        segs = [s for s in timeline if isinstance(s, dict) and isinstance(s.get("source"), str)]
        segs.sort(key=lambda s: (_num(s.get("segment_number")) or 0, _num(_first(s, START_KEYS)) or 0))
        for i, seg in enumerate(segs):
            neighbours[seg["source"]] = (segs[i - 1] if i > 0 else None, segs[i + 1] if i + 1 < len(segs) else None)
    for hit in hits:
        before, after = neighbours.get(hit["source"], (None, None))
        if before or after:
            hit["context"] = {"prev": _context_segment(before), "next": _context_segment(after)}


def normalize_search(data: Any) -> tuple[list[dict], dict]:
    """Segment hits from a /search response (deduped by source) plus a little response meta."""
    if not isinstance(data, dict):
        return [], {"note": f"unexpected response type {type(data).__name__}"}
    results = data.get("results") or data.get("hits") or data.get("segments") or []
    chunks = data.get("chunk_results") or data.get("chunks") or []
    chunk_by_video = {
        str(c["original_video"]): c
        for c in (chunks if isinstance(chunks, list) else [])
        if isinstance(c, dict) and c.get("original_video")
    }
    hits: list[dict] = []
    seen: set[str] = set()
    for row in results if isinstance(results, list) else []:
        hit = normalize_hit(row)
        if not hit or hit["source"] in seen:
            continue
        parent = chunk_by_video.get(hit["original_video"] or "")
        if parent:  # fill metadata gaps from the parent upload
            for field in ("camera_id", "location", "capture_type"):
                hit[field] = hit[field] or _meta(parent, field)
        seen.add(hit["source"])
        hits.append(hit)
    if not hits:  # a deployment that only returns grouped chunks: use their best-matching segment
        for chunk in chunks if isinstance(chunks, list) else []:
            hit = normalize_hit(chunk)
            if hit and hit["source"] not in seen:
                hit["from_chunk"] = True
                seen.add(hit["source"])
                hits.append(hit)
    attach_context(hits, chunks)
    meta = {k: data.get(k) for k in ("total", "chunk_total", "embedding_time_ms", "search_time_ms",
                                     "permission_filtered") if k in data}
    return hits, meta


def _clean_base(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    if url and not re.match(r"^https?://", url, re.I):
        url = "http://" + url
    if url.endswith("/api/v1"):
        url = url[: -len("/api/v1")]
    return url


def _detail(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        detail = body.get("detail", body) if isinstance(body, dict) else body
    except ValueError:
        detail = resp.text
    if not isinstance(detail, str):
        detail = json.dumps(detail)[:300]
    return redact(detail)[:240]


# --------------------------------------------------------------------------- client

class VSSClient:
    """Async client for /api/v1 on the team's VSS backend. Credentials never leave this object."""

    def __init__(self, base_url: str, username: str, password: str, public_url: str = "",
                 timeout: float = 120.0):
        self.base = _clean_base(base_url)
        self.public = _clean_base(public_url) or self.base
        self.username = username or ""
        self._password = password or ""
        self._timeout = timeout
        self._token: str | None = None
        self._client: httpx.AsyncClient | None = None
        self._login_lock: asyncio.Lock | None = None
        self.ok: bool | None = None
        self.last_error: str | None = None

    def __repr__(self) -> str:  # never show credentials
        return f"VSSClient(configured={self.configured}, ok={self.ok})"

    @property
    def configured(self) -> bool:
        return bool(self.base and self.username and self._password)

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self._timeout, connect=15.0),
                                             follow_redirects=True, headers={"User-Agent": USER_AGENT})
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def login(self, stale: str | None = None) -> str:
        """Return a cached JWT; log in when there is none or when ``stale`` was rejected."""
        if not self.configured:
            raise VSSError("VSS backend not configured", 503)
        if self._login_lock is None:
            self._login_lock = asyncio.Lock()
        async with self._login_lock:
            if self._token and self._token != stale:
                return self._token
            try:
                resp = await self._http().post(
                    f"{self.base}/api/v1/auth/login",
                    json={"username": self.username, "password": self._password})
            except httpx.HTTPError as exc:
                self.ok, self.last_error = False, f"cannot reach VSS backend ({type(exc).__name__})"
                raise VSSError(f"Cannot reach the VSS backend ({type(exc).__name__})", 502) from None
            if resp.status_code != 200:
                self.ok, self.last_error = False, f"login failed (HTTP {resp.status_code})"
                raise VSSError(f"VSS login failed (HTTP {resp.status_code}): {_detail(resp)}", 502)
            try:
                body = resp.json()
            except ValueError:
                body = {}
            token = (body.get("access_token") or body.get("token")) if isinstance(body, dict) else None
            if not token:
                self.ok, self.last_error = False, "login response had no access_token"
                raise VSSError("VSS login response had no access_token", 502)
            self._token = str(token)
            self.ok, self.last_error = True, None
            return self._token

    async def check(self) -> bool:
        try:
            await self.login()
            return True
        except VSSError:
            return False

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Authenticated request; on 401 re-login once and retry."""
        token = await self.login()
        resp: httpx.Response | None = None
        for attempt in (0, 1):
            try:
                resp = await self._http().request(method, f"{self.base}{path}",
                                                  headers={"Authorization": f"Bearer {token}"}, **kwargs)
            except httpx.TimeoutException:
                raise VSSError(f"VSS backend timed out on {path}", 504) from None
            except httpx.HTTPError as exc:
                raise VSSError(f"Cannot reach the VSS backend ({type(exc).__name__})", 502) from None
            if resp.status_code == 401 and attempt == 0:
                token = await self.login(stale=token)
                continue
            break
        assert resp is not None
        return resp

    async def search(self, query: str, top_k: int = 25, min_similarity: float = 0.3,
                     metadata_filters: dict | None = None,
                     hybrid_text_weight: float | None = None) -> tuple[list[dict], dict]:
        """POST /api/v1/search → (normalized hits, response meta)."""
        body: dict[str, Any] = {
            "query": query,
            "top_k": max(1, min(100, int(top_k))),
            "min_similarity": max(0.0, min(1.0, float(min_similarity))),
            "llm_top_n": 1,  # the backend rejects 0 (422, ge=1); 1 keeps its synthesis step minimal
            "include_public": True,
            "metadata_filters": {k: v for k, v in (metadata_filters or {}).items() if v not in (None, "")},
        }
        if hybrid_text_weight is not None:
            body["hybrid_text_weight"] = max(0.0, min(1.0, float(hybrid_text_weight)))
        resp = await self._request("POST", "/api/v1/search", json=body)
        if resp.status_code != 200:
            raise VSSError(f"VSS search failed (HTTP {resp.status_code}): {_detail(resp)}", 502)
        try:
            data = resp.json()
        except ValueError:
            raise VSSError("VSS search returned a non-JSON response", 502) from None
        return normalize_search(data)

    async def location_values(self, limit: int = 50) -> list[str]:
        resp = await self._request("GET", "/api/v1/metadata/values",
                                   params={"field": "location", "limit": limit})
        if resp.status_code != 200:
            raise VSSError(f"metadata values failed (HTTP {resp.status_code})", 502)
        try:
            data = resp.json()
        except ValueError:
            return []
        values = data.get("values") if isinstance(data, dict) else data
        out: list[str] = []
        for value in values if isinstance(values, list) else []:
            if isinstance(value, dict):
                value = value.get("value") or value.get("name") or value.get("label")
            if value not in (None, "") and str(value) not in out:
                out.append(str(value))
        return out

    async def segment_row(self, source: str) -> dict | None:
        """GET /api/v1/videos/metadata?source=… → the segment's row (caption, timing, metadata)."""
        resp = await self._request("GET", "/api/v1/videos/metadata", params={"source": source})
        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            raise VSSError(f"segment metadata failed (HTTP {resp.status_code})", 502)
        try:
            data = resp.json()
        except ValueError:
            return None
        if isinstance(data, dict) and isinstance(data.get("segment"), dict):
            data = data["segment"]
        return data if isinstance(data, dict) else None

    def stream_url(self, source: str, public: bool = True) -> str | None:
        """Browser playback URL (the JWT rides in ?token= because <video> can't send headers)."""
        if not self._token or not source:
            return None
        base = self.public if public else self.base
        return (f"{base}/api/v1/videos/stream?source={quote(source, safe='')}"
                f"&token={quote(self._token, safe='')}")

    async def download_clip(self, source: str, max_bytes: int = MAX_CLIP_BYTES) -> bytes:
        """Fetch a segment's bytes server-side via /videos/stream (capped at ``max_bytes``)."""
        token = await self.login()
        for attempt in (0, 1):
            try:
                async with self._http().stream("GET", f"{self.base}/api/v1/videos/stream",
                                               params={"source": source, "token": token}) as resp:
                    if resp.status_code == 401 and attempt == 0:
                        token = await self.login(stale=token)
                        continue
                    if resp.status_code not in (200, 206):
                        raise VSSError(f"clip download failed (HTTP {resp.status_code})", 502)
                    size = int(_num(resp.headers.get("content-length")) or 0)
                    if size > max_bytes:
                        raise VSSError(f"clip too large ({size // 1048576} MB)", 413)
                    buf = bytearray()
                    async for chunk in resp.aiter_bytes():
                        buf.extend(chunk)
                        if len(buf) > max_bytes:
                            raise VSSError(f"clip larger than {max_bytes // 1048576} MB", 413)
                    return bytes(buf)
            except httpx.TimeoutException:
                raise VSSError("clip download timed out", 504) from None
            except httpx.HTTPError as exc:
                raise VSSError(f"clip download failed ({type(exc).__name__})", 502) from None
        raise VSSError("clip download unauthorized", 502)
