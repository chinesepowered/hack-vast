"""Edge-Case Miner: find, verify and export rare edge-case clips from a VAST VSS video archive.

    python main.py          serve on 0.0.0.0:$PORT (default 8080); routes live at / (Ingress strips /app)
    python main.py --warm   precompute the coverage grid + verifications into the cache, then exit
    MOCK=1 python main.py   offline demo corpus, no network calls at all

Pipeline: scenario → W&B LLM query expansion → VSS hybrid search across every camera →
Cosmos3-Reason watches each candidate clip → coverage grid (scenario × location) →
manifest + versioned W&B Artifact. Steps are traced with W&B Weave when available.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from urllib.parse import quote

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

import cosmos
import llm
import mock
import vss

HERE = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for _noisy in ("httpx", "httpcore"):  # their INFO lines print full URLs; stream URLs carry a JWT
    logging.getLogger(_noisy).setLevel(logging.WARNING)
log = logging.getLogger("ecm")


# ============================================================================ configuration (read once)

def env(*names: str, default: str = "") -> str:
    """First non-empty environment variable among ``names``."""
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return default


def env_int(name: str, default: int) -> int:
    try:
        return int(env(name, default=str(default)))
    except ValueError:
        return default


MOCK = env("MOCK").lower() in ("1", "true", "yes", "on")
PORT = env_int("PORT", 8080)
MAX_VERIFY = max(1, env_int("MAX_VERIFY", 24))
VERIFY_CONCURRENCY = max(1, env_int("VERIFY_CONCURRENCY", 6))
SEARCH_CACHE_TTL = env_int("SEARCH_CACHE_TTL", 6 * 3600)
SEARCH_CONCURRENCY = 4  # the stock VSS backend runs 4 workers; leave room for the team's other users
# Hybrid similarity on the live index is low (the best hit for a clear query scores ~0.3), so search
# wide and let Cosmos verification do the filtering.
DEFAULT_MIN_SIM = 0.12
DEFAULT_TOP_K = 30
COVERAGE_TOP_K = 100
COVERAGE_MIN_SIM = 0.12   # what the coverage search fetches (all of it is eligible for verification)
COVERAGE_HIT_SIM = 0.2    # what a coverage cell counts as a "hit"
UNKNOWN = "(unknown)"
WANDB_KEY = env("WANDB_API_KEY")
WANDB_ENTITY = env("WANDB_TEAM", "WANDB_ENTITY")
WANDB_PROJECT = env("WANDB_PROJECT")
NOT_CONFIGURED = ("VSS backend not configured. Set VSS_URL (or INGRESS_URL), VSS_USERNAME (or USERNAME) and "
                  "VSS_PASSWORD (or PASSWORD), e.g. `set -a; source /config/<team>.config; set +a` - "
                  "or run with MOCK=1 for the offline demo.")


def _cache_root() -> Path:
    sub = "mock" if MOCK else "live"
    for root in (Path(env("CACHE_DIR", default="/tmp/ecm-cache")), Path(tempfile.gettempdir()) / "ecm-cache"):
        try:
            (root / sub).mkdir(parents=True, exist_ok=True)
            return root / sub
        except OSError:
            continue
    raise SystemExit("no writable CACHE_DIR")


CACHE_DIR = _cache_root()
EXPORT_DIR = CACHE_DIR / "exports"
EXPORT_DIR.mkdir(exist_ok=True)
COVERAGE_FILE = CACHE_DIR / "coverage.json"

VSS = vss.VSSClient(env("VSS_URL", "INGRESS_URL"), env("VSS_USERNAME", "USERNAME"),
                    env("VSS_PASSWORD", "PASSWORD"), env("PUBLIC_VSS_URL"))
LLM = llm.LLMClient(WANDB_KEY, env("WANDB_INFERENCE_URL", default=llm.DEFAULT_INFERENCE_URL),
                    WANDB_ENTITY, WANDB_PROJECT, env("LLM_MODEL"),
                    expand_model=env("LLM_EXPAND_MODEL"), judge_model=env("LLM_JUDGE_MODEL"))
COSMOS = cosmos.CosmosClient(env("COSMOS3_REASON_URL"), env("GPU_BEARER_TOKEN"), env("COSMOS3_REASON_MODEL"))
TAXONOMY: list[dict] = json.loads((HERE / "taxonomy.json").read_text())


# ============================================================================ small helpers

class AppError(Exception):
    """An error with an HTTP status and a message that is safe to show."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def err_text(exc: BaseException) -> str:
    if isinstance(exc, (AppError, vss.VSSError)):
        return exc.message
    return f"{type(exc).__name__}: {vss.redact(exc)[:200]}"


def require_vss() -> None:
    if not MOCK and not VSS.configured:
        raise AppError(503, NOT_CONFIGURED)


def _ms(t0: float, t1: float | None = None) -> int:
    return int(((t1 if t1 is not None else time.perf_counter()) - t0) * 1000)


def _dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for item in items:
        key = " ".join(str(item).lower().split())
        if key and key not in seen:
            seen.add(key)
            out.append(" ".join(str(item).split()))
    return out


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _write_json(path: Path, value: Any) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(value))
        os.replace(tmp, path)
    except (OSError, TypeError, ValueError) as exc:
        log.warning("could not write %s (%s)", path.name, type(exc).__name__)


def slugify(text: str, limit: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return (slug[:limit].strip("-") or "edge-cases")


class JsonCache:
    """JSON files keyed by hash, with a TTL. Shared with `--warm` runs through the filesystem."""

    def __init__(self, directory: Path, ttl: int = 0):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl

    @staticmethod
    def key(*parts: Any) -> str:
        return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()

    def get(self, key: str) -> Any:
        path = self.dir / f"{key}.json"
        try:
            if self.ttl and time.time() - path.stat().st_mtime > self.ttl:
                return None
        except OSError:
            return None
        return _read_json(path)

    def put(self, key: str, value: Any) -> None:
        _write_json(self.dir / f"{key}.json", value)


SEARCH_CACHE = JsonCache(CACHE_DIR / "search", SEARCH_CACHE_TTL)
EXPAND_CACHE = JsonCache(CACHE_DIR / "expand")
SEARCH_SEM = asyncio.Semaphore(SEARCH_CONCURRENCY)

# Every clip we've seen in a search (minus stream URLs), so /api/verify can find its caption.
REGISTRY: "OrderedDict[str, dict]" = OrderedDict()


def remember(hit: dict) -> None:
    source = hit.get("source")
    if not source:
        return
    REGISTRY[source] = {k: v for k, v in hit.items() if k not in ("stream_url", "matched_queries")}
    REGISTRY.move_to_end(source)
    while len(REGISTRY) > 20000:
        REGISTRY.popitem(last=False)


async def lookup(source: str) -> dict | None:
    """A clip's metadata + caption: from earlier searches, else from GET /videos/metadata."""
    if source in REGISTRY:
        return REGISTRY[source]
    if MOCK:
        hit = mock.get(source)
    elif VSS.configured:
        try:
            hit = vss.normalize_hit(await VSS.segment_row(source))
        except vss.VSSError:
            hit = None
    else:
        hit = None
    if hit:
        remember(hit)
    return hit


def clip_url(source: str | None) -> str | None:
    """Playback URL relative to the page: /api/clip relays the video, so the JWT stays server-side
    and clips play wherever the app is reachable (Ingress, public tunnel, localhost)."""
    if MOCK or not source:
        return None
    return f"api/clip?source={quote(source, safe='')}"


def with_stream(candidate: dict) -> dict:
    """Add browser playback URLs to a candidate and its before/after context segments."""
    out = dict(candidate)
    out["stream_url"] = clip_url(candidate["source"])
    context = candidate.get("context")
    if isinstance(context, dict):
        out["context"] = {side: ({**seg, "stream_url": clip_url(seg["source"])}
                                 if isinstance(seg, dict) and seg.get("source") else None)
                          for side, seg in context.items()}
    return out


VERDICTS = cosmos.VerdictCache(CACHE_DIR / "verdicts")
VERIFIER = cosmos.Verifier(VERDICTS, COSMOS, LLM, VSS, lookup, concurrency=VERIFY_CONCURRENCY,
                           workdir=CACHE_DIR / "tmp", mock_judge=mock.judge if MOCK else None)


# ============================================================================ pipeline steps (Weave ops)

@llm.op
async def expand(scenario: str) -> dict:
    """Scenario → caption-style search queries (W&B Inference LLM; templates as fallback)."""
    if MOCK:
        return {"queries": mock.expand(scenario), "model": "mock-templates", "fallback": False}
    key = JsonCache.key("expand", scenario.lower(), await LLM.model("expand") or "")
    cached = EXPAND_CACHE.get(key)
    if cached:
        return cached
    result = await LLM.expand_queries(scenario)
    if not result.get("fallback"):
        EXPAND_CACHE.put(key, result)
    return result


@llm.op
async def search(query: str, top_k: int = DEFAULT_TOP_K, min_similarity: float = DEFAULT_MIN_SIM,
                 location: Optional[str] = None, hybrid_text_weight: Optional[float] = None,
                 fresh: bool = False) -> list:
    """One hybrid (caption text + visual embedding) search across every camera."""
    if MOCK:
        hits = mock.search(query, top_k, min_similarity, location, hybrid_text_weight)
    else:
        require_vss()
        key = JsonCache.key("search", query, top_k, round(min_similarity, 3), location or "", hybrid_text_weight)
        hits = None if fresh else SEARCH_CACHE.get(key)
        if hits is None:
            async with SEARCH_SEM:
                hits, _meta = await VSS.search(query, top_k, min_similarity,
                                               {"location": location} if location else None, hybrid_text_weight)
            SEARCH_CACHE.put(key, hits)
    for hit in hits:
        remember(hit)
    return hits


def merge_hits(per_query: list[tuple[str, list[dict]]]) -> list[dict]:
    """Union of hits across queries: keep the max similarity and list which queries matched."""
    merged: dict[str, dict] = {}
    for query, hits in per_query:
        for hit in hits:
            current = merged.get(hit["source"])
            if current is None:
                current = merged[hit["source"]] = {**hit, "matched_queries": []}
            elif (hit.get("similarity") or 0) > (current.get("similarity") or 0):
                current.update({**hit, "matched_queries": current["matched_queries"]})
            if query not in current["matched_queries"]:
                current["matched_queries"].append(query)
    return sorted(merged.values(), key=lambda c: (-(c.get("similarity") or 0), -len(c["matched_queries"])))


async def _search_all(queries: list[str], top_k: int, min_similarity: float, location: Optional[str],
                      hybrid_text_weight: Optional[float], fresh: bool) -> tuple[list, list]:
    results = await asyncio.gather(
        *(search(q, top_k, min_similarity, location, hybrid_text_weight, fresh) for q in queries),
        return_exceptions=True)
    per_query, errors = [], []
    for query, res in zip(queries, results):
        if isinstance(res, BaseException):
            errors.append({"query": query, "error": err_text(res)})
        else:
            per_query.append((query, res))
    if not per_query:
        first = next((r for r in results if isinstance(r, (AppError, vss.VSSError))), None)
        status = first.status if first is not None else 502
        raise AppError(status, f"All {len(queries)} searches failed: {errors[0]['error'] if errors else 'no queries'}")
    return per_query, errors


@llm.op
async def mine(scenario: str, top_k: int = DEFAULT_TOP_K, min_similarity: float = DEFAULT_MIN_SIM,
               hybrid_text_weight: Optional[float] = None, location: Optional[str] = None,
               fresh: bool = False) -> dict:
    """Expand the scenario, search every query concurrently, merge into ranked candidates."""
    t0 = time.perf_counter()
    expansion = await expand(scenario)
    queries = _dedupe([scenario] + list(expansion.get("queries") or []))[:5]
    t1 = time.perf_counter()
    notes = []
    if expansion.get("fallback"):
        notes.append(f"Query expansion used templates ({expansion.get('error') or 'LLM unavailable'}).")
    per_query, errors = await _search_all(queries, top_k, min_similarity, location, hybrid_text_weight, fresh)
    candidates = merge_hits(per_query)
    used_min_sim = min_similarity
    if not candidates and not errors and min_similarity > 0.08:
        used_min_sim = round(max(0.05, min_similarity / 2), 2)
        notes.append(f"No hits at min similarity {min_similarity:.2f}; relaxed to {used_min_sim:.2f}.")
        per_query, errors = await _search_all(queries, top_k, used_min_sim, location, hybrid_text_weight, fresh)
        candidates = merge_hits(per_query)
    t2 = time.perf_counter()
    return {
        "scenario": scenario,
        "queries": queries,
        "candidates": candidates[:top_k],
        "total_unique": len(candidates),
        "expansion": {"model": expansion.get("model"), "fallback": bool(expansion.get("fallback"))},
        "min_similarity": used_min_sim,
        "location": location,
        "timings": {"expand_ms": _ms(t0, t1), "search_ms": _ms(t1, t2), "total_ms": _ms(t0, t2)},
        "notes": notes,
        "errors": errors,
    }


@llm.op
async def verify(scenario: str, source: str) -> dict:
    """Cosmos3-Reason watches the clip and votes match/no-match (cached; caption-judge fallback)."""
    return await VERIFIER.verify(scenario, source)


def models_used() -> dict:
    if MOCK:
        return {"search": "mock concept-overlap search", "verifier": "mock-cosmos",
                "query_expansion": "mock-templates"}
    return {
        "search": "VSS hybrid search (Cosmos-Embed1 text + visual vectors in VastDB)",
        "captions": "Cosmos3-Reason (ingest captions)",
        "verifier": COSMOS.model_name or (f"caption judge ({LLM.judge_model_name})" if LLM.judge_model_name else None),
        "query_expansion": LLM.model_name or "templates",
    }


def _f(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return round(f, 4) if f == f else None


def _s(value: Any, limit: int = 300) -> str | None:
    return None if value in (None, "") else str(value)[:limit]


def manifest_entry(item: Any, default_scenario: str) -> dict | None:
    if isinstance(item, str):
        item = {"source": item}
    if not isinstance(item, dict) or not str(item.get("source") or "").strip():
        return None
    source = str(item["source"])
    known = REGISTRY.get(source, {})
    label = _norm_text(item.get("scenario") or item.get("label") or default_scenario)
    cached = VERDICTS.get(source, label) if label else None
    verdict = cached or (item.get("verdict") if isinstance(item.get("verdict"), dict) else {}) or {}

    def pick(key: str) -> Any:
        value = item.get(key)
        return value if value not in (None, "") else known.get(key)

    match = verdict.get("match")
    return {
        "source": source,
        "original_video": _s(pick("original_video"), 1000),
        "start_sec": _f(pick("start_sec")),
        "end_sec": _f(pick("end_sec")),
        "camera_id": _s(pick("camera_id")),
        "location": _s(pick("location")),
        "label": label or None,
        "match": match if isinstance(match, bool) else None,
        "confidence": _f(verdict.get("confidence")),
        "why": _s(verdict.get("why"), 500),
        "method": _s(verdict.get("method"), 40),
        "caption": _s(pick("caption"), 2000),
        "similarity": _f(pick("similarity")),
    }


def _norm_text(value: Any) -> str:
    return " ".join(str(value or "").split())[:300]


@llm.op
async def export_dataset(scenario: str, items: list) -> dict:
    """Write the verified set as a manifest JSON and version it as a W&B Artifact when possible."""
    entries = [e for e in (manifest_entry(item, scenario) for item in items) if e]
    if not entries:
        raise AppError(400, "No valid items to export (each item needs a source).")
    entries = list({e["source"]: e for e in entries}.values())
    now = time.gmtime()
    slug = slugify(scenario)
    export_id = f"{slug}-{time.strftime('%Y%m%d-%H%M%S', now)}-{uuid.uuid4().hex[:6]}"
    counts = {
        "items": len(entries),
        "matches": sum(1 for e in entries if e["match"] is True),
        "rejected": sum(1 for e in entries if e["match"] is False),
        "unverified": sum(1 for e in entries if e["match"] is None),
    }
    models = models_used()
    manifest = {
        "name": slug,
        "scenario": scenario,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", now),
        "generator": "edge-case-miner",
        "mock": MOCK,
        "counts": counts,
        "models": models,
        "items": entries,
    }
    path = EXPORT_DIR / f"{export_id}.json"
    _write_json(path, manifest)
    wandb_url = None
    if not MOCK and WANDB_KEY:
        wandb_url = await asyncio.to_thread(
            llm.log_dataset_artifact, str(path), slug,
            {"scenario": scenario, "counts": counts, "models": models}, WANDB_ENTITY, WANDB_PROJECT, WANDB_KEY)
    return {"id": export_id, "manifest_url": f"api/export/{export_id}.json", "wandb_url": wandb_url,
            "counts": counts}


# ============================================================================ coverage grid

COVERAGE: dict | None = None
COVERAGE_MTIME = 0
COVERAGE_LOCK = asyncio.Lock()
_LOCATIONS_CACHE: dict[str, Any] = {"at": 0.0, "values": []}


async def known_locations(hits_by_scenario: dict | None = None) -> list[str]:
    """Columns: /metadata/values?field=location (cached 10 min) ∪ locations seen in hits."""
    values: list[str] = []
    if MOCK:
        values = mock.location_values()
    elif VSS.configured:
        if time.time() - _LOCATIONS_CACHE["at"] < 600:
            values = list(_LOCATIONS_CACHE["values"])
        else:
            try:
                values = await VSS.location_values()
                _LOCATIONS_CACHE.update(at=time.time(), values=list(values))
            except vss.VSSError as exc:
                log.info("location values unavailable (%s); using locations seen in results", exc.message)
    for hits in (hits_by_scenario or {}).values():
        for hit in hits:
            loc = hit.get("location") or UNKNOWN
            if loc not in values:
                values.append(loc)
    return sorted(set(values), key=lambda v: (v == UNKNOWN, v.lower()))


def _compact(hit: dict) -> dict:
    keep = ("source", "original_video", "camera_id", "location", "capture_type", "start_sec", "end_sec",
            "similarity", "objects", "object_counts")
    return {**{k: hit.get(k) for k in keep}, "caption": (hit.get("caption") or "")[:1500]}


async def coverage_base(refresh: bool = False) -> dict:
    """One search per taxonomy scenario (no location filter); hits kept per scenario. Cached on disk."""
    global COVERAGE, COVERAGE_MTIME
    async with COVERAGE_LOCK:
        if not refresh:
            try:
                mtime = COVERAGE_FILE.stat().st_mtime_ns
            except OSError:
                mtime = 0
            if mtime and mtime != COVERAGE_MTIME:  # written by us or by `python main.py --warm`
                disk = _read_json(COVERAGE_FILE)
                if isinstance(disk, dict) and isinstance(disk.get("hits"), dict):
                    COVERAGE, COVERAGE_MTIME = disk, mtime
                    for hits in disk["hits"].values():
                        for hit in hits:
                            remember(hit)
            if COVERAGE:
                return COVERAGE
        t0 = time.perf_counter()
        results = await asyncio.gather(
            *(search(t["query"], COVERAGE_TOP_K, COVERAGE_MIN_SIM, None, None, refresh) for t in TAXONOMY),
            return_exceptions=True)
        hits: dict[str, list] = {}
        errors: dict[str, str] = {}
        for entry, res in zip(TAXONOMY, results):
            if isinstance(res, BaseException):
                errors[entry["id"]] = err_text(res)
                hits[entry["id"]] = []
            else:
                hits[entry["id"]] = [_compact(h) for h in res]
        if len(errors) == len(TAXONOMY):
            first = next(iter(errors.values()))
            raise AppError(503 if first == NOT_CONFIGURED else 502, f"Coverage searches failed: {first}")
        COVERAGE = {
            "generated_at": time.time(),
            "elapsed_ms": _ms(t0),
            "params": {"top_k": COVERAGE_TOP_K, "min_similarity": COVERAGE_MIN_SIM, "hit_similarity": COVERAGE_HIT_SIM},
            "locations": await known_locations(hits),
            "hits": hits,
            "errors": errors,
        }
        _write_json(COVERAGE_FILE, COVERAGE)
        try:
            COVERAGE_MTIME = COVERAGE_FILE.stat().st_mtime_ns
        except OSError:
            pass
        return COVERAGE


def _cell(hits: int, matches: int, verified: int) -> dict:
    gap = (matches == 0) if verified else (hits == 0)
    status = "covered" if matches else "gap" if gap else "unverified"
    return {"hits": hits, "verified_matches": matches, "verified_total": verified, "gap": gap, "status": status}


def coverage_view(base: dict) -> dict:
    """Grid cells: search hits (similarity ≥ COVERAGE_HIT_SIM) per location + verified counts from the
    verdict cache. A cell is a GAP when verified_matches == 0 if anything there was verified, else when
    hits == 0; cells with hits but no verdicts yet are "unverified" (Warm cache verifies them).
    """
    locations = list(base.get("locations") or [])
    cells: dict[str, dict] = {}
    totals: dict[str, dict] = {}
    per_scenario_verdicts = {}
    for entry in TAXONOMY:
        verdicts: dict[str, list] = {}
        for v in VERDICTS.for_scenario(entry["query"]):
            loc = v.get("location") or REGISTRY.get(v["source"], {}).get("location") or UNKNOWN
            verdicts.setdefault(loc, []).append(v)
            if loc not in locations:
                locations.append(loc)
        per_scenario_verdicts[entry["id"]] = verdicts
    locations.sort(key=lambda v: (v == UNKNOWN, v.lower()))
    gaps = 0
    for entry in TAXONOMY:
        sid = entry["id"]
        by_loc: dict[str, set] = {}
        for hit in base.get("hits", {}).get(sid, []):
            if (hit.get("similarity") or 0) >= COVERAGE_HIT_SIM:
                by_loc.setdefault(hit.get("location") or UNKNOWN, set()).add(hit["source"])
        row = {}
        for loc in locations:
            vs = per_scenario_verdicts[sid].get(loc, [])
            row[loc] = _cell(len(by_loc.get(loc, ())), sum(1 for v in vs if v.get("match") is True),
                             sum(1 for v in vs if v.get("match") is not None))
            gaps += row[loc]["gap"]
        cells[sid] = row
        totals[sid] = _cell(sum(c["hits"] for c in row.values()), sum(c["verified_matches"] for c in row.values()),
                            sum(c["verified_total"] for c in row.values()))
    return {
        "scenarios": TAXONOMY,
        "locations": locations,
        "cells": cells,
        "totals": totals,
        "summary": {
            "cells": len(TAXONOMY) * len(locations),
            "gaps": gaps,
            "unverified": sum(1 for row in cells.values() for c in row.values() if c["status"] == "unverified"),
            "covered": sum(1 for row in cells.values() for c in row.values() if c["status"] == "covered"),
            "scenario_gaps": sum(1 for t in totals.values() if t["gap"]),
            "verified": sum(t["verified_total"] for t in totals.values()),
            "verified_matches": sum(t["verified_matches"] for t in totals.values()),
        },
        "generated_at": base.get("generated_at"),
        "params": base.get("params"),
        "errors": base.get("errors") or {},
    }


def coverage_picks(base: dict, scenario_id: str, per_location: int = 3) -> list[str]:
    """Top hits per location (so every cell gets some verification), best first."""
    by_loc: dict[str, list] = {}
    for hit in base.get("hits", {}).get(scenario_id, []):
        by_loc.setdefault(hit.get("location") or UNKNOWN, []).append(hit)
    picks = []
    for hits in by_loc.values():
        hits.sort(key=lambda h: -(h.get("similarity") or 0))
        picks.extend(h["source"] for h in hits[:per_location])
    return picks


# ============================================================================ warm-up (background)

WARM: dict[str, Any] = {"running": False, "phase": "idle", "done": 0, "total": 0, "detail": "",
                        "started_at": None, "finished_at": None, "verified": 0, "matches": 0, "errors": []}


async def run_warm() -> dict:
    """Coverage grid + mine every taxonomy scenario + verify its top hits, all into the cache."""
    WARM.update(running=True, phase="coverage", done=0, total=1 + len(TAXONOMY), verified=0, matches=0,
                errors=[], started_at=time.time(), finished_at=None,
                detail="Searching every scenario across all cameras")
    try:
        base = await coverage_base(refresh=True)
        WARM["done"] += 1
        WARM["phase"] = "mine"
        work: list[tuple[dict, list[str]]] = []
        for entry in TAXONOMY:
            WARM["detail"] = f"Mining: {entry['label']}"
            picks: list[str] = []
            try:
                result = await mine(entry["query"])
                picks = [c["source"] for c in result["candidates"][:MAX_VERIFY]]
            except Exception as exc:  # noqa: BLE001 - keep warming the other scenarios
                WARM["errors"].append(f"{entry['label']}: {err_text(exc)}")
            work.append((entry, list(dict.fromkeys(picks + coverage_picks(base, entry["id"])))))
            WARM["done"] += 1
        WARM.update(phase="verify", total=WARM["done"] + sum(len(p) for _, p in work),
                    detail="Cosmos3-Reason is watching the top clips")

        async def one(entry: dict, source: str) -> None:
            verdict = await verify(entry["query"], source)
            WARM["done"] += 1
            WARM["verified"] += verdict.get("match") is not None
            WARM["matches"] += verdict.get("match") is True
            WARM["detail"] = f"Verified a clip for: {entry['label']}"

        await asyncio.gather(*(one(entry, src) for entry, picks in work for src in picks))
        WARM.update(phase="done", detail=f"Verified {WARM['verified']} clips, {WARM['matches']} matches")
    except Exception as exc:  # noqa: BLE001
        WARM["errors"].append(err_text(exc))
        WARM.update(phase="error", detail=err_text(exc))
    finally:
        WARM["running"] = False
        WARM["finished_at"] = time.time()
    return WARM


# ============================================================================ app + routes

PROBE: asyncio.Task | None = None
WARM_TASK: asyncio.Task | None = None


async def probe_integrations() -> None:
    """Background start-up checks: Weave init, VSS login, Cosmos/LLM model discovery."""
    jobs = []
    if WANDB_KEY:
        jobs.append(asyncio.to_thread(llm.init_weave, WANDB_ENTITY, WANDB_PROJECT, WANDB_KEY))
    if VSS.configured:
        jobs.append(VSS.check())
    if COSMOS.configured:
        jobs.append(COSMOS.model())
    if LLM.configured:
        jobs.append(LLM.model())
    await asyncio.gather(*jobs, return_exceptions=True)
    log.info("integrations: vss=%s cosmos=%s llm=%s weave=%s", VSS.ok, COSMOS.model_name, LLM.model_name,
             llm.WEAVE_ENABLED)


async def close_clients() -> None:
    for client in (VSS, LLM, COSMOS):
        try:
            await client.aclose()
        except Exception:  # noqa: BLE001
            pass


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global PROBE
    if not MOCK:
        PROBE = asyncio.create_task(probe_integrations())
    log.info("Edge-Case Miner up (mock=%s, cache=%s, verified clips cached=%d)", MOCK, CACHE_DIR, len(VERDICTS))
    yield
    await close_clients()


app = FastAPI(title="Edge-Case Miner", version="1.0", lifespan=lifespan)


@app.exception_handler(AppError)
async def _app_error(_request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse({"error": exc.message}, status_code=exc.status)


@app.exception_handler(vss.VSSError)
async def _vss_error(_request: Request, exc: vss.VSSError) -> JSONResponse:
    return JSONResponse({"error": exc.message}, status_code=exc.status)


@app.exception_handler(Exception)
async def _unexpected(_request: Request, exc: Exception) -> JSONResponse:
    log.error("unhandled %s: %s", type(exc).__name__, vss.redact(exc)[:300])
    return JSONResponse({"error": f"Internal error ({type(exc).__name__})"}, status_code=500)


class MineRequest(BaseModel):
    scenario: str = Field(..., min_length=2, max_length=300)
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=100)
    min_similarity: float = Field(DEFAULT_MIN_SIM, ge=0.0, le=1.0)
    hybrid_text_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    location: Optional[str] = Field(None, max_length=200)
    fresh: bool = False


class VerifyRequest(BaseModel):
    scenario: str = Field(..., min_length=2, max_length=300)
    sources: list[str] = Field(..., min_length=1)


class SimilarRequest(BaseModel):
    source: str = Field(..., min_length=3)
    scenario: str = Field("", max_length=300)
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=100)
    min_similarity: float = Field(DEFAULT_MIN_SIM, ge=0.0, le=1.0)
    hybrid_text_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    location: Optional[str] = Field(None, max_length=200)


class ExportRequest(BaseModel):
    scenario: str = Field("", max_length=300)
    items: list[Any] = Field(default_factory=list)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(HERE / "index.html", media_type="text/html", headers={"Cache-Control": "no-cache"})


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "mock": MOCK}


@app.get("/api/config")
async def api_config() -> dict:
    """Which integrations are configured / working. Never returns secrets or hosts."""
    common = {"max_verify": MAX_VERIFY, "verify_concurrency": VERIFY_CONCURRENCY}
    if MOCK:
        return {"mock": True, "vss": {"configured": True, "ok": True},
                "cosmos": {"configured": True, "model": "mock-cosmos"},
                "llm": {"configured": True, "model": "mock-templates"},
                "weave": False, "wandb": False, **common}
    if PROBE is not None and not PROBE.done():
        try:
            await asyncio.wait_for(asyncio.shield(PROBE), timeout=4)
        except asyncio.TimeoutError:
            pass
    return {
        "mock": False,
        "vss": {"configured": VSS.configured, "ok": VSS.ok, "error": VSS.last_error},
        "cosmos": {"configured": COSMOS.configured, "model": COSMOS.model_name, "error": COSMOS.last_error,
                   "transcode": cosmos.ffmpeg_exe() is not None},
        "llm": {"configured": LLM.configured, "model": LLM.model_name, "judge_model": LLM.judge_model_name,
                "error": LLM.last_error},
        "weave": llm.WEAVE_ENABLED,
        "wandb": bool(WANDB_KEY) and llm.wandb_installed(),
        "probing": PROBE is not None and not PROBE.done(),
        **common,
    }


@app.get("/api/taxonomy")
async def api_taxonomy() -> dict:
    return {"scenarios": TAXONOMY}


@app.get("/api/locations")
async def api_locations() -> dict:
    return {"locations": [loc for loc in await known_locations() if loc != UNKNOWN]}


@app.get("/api/clip")
async def api_clip(source: str, request: Request) -> StreamingResponse:
    """Relay one segment's video from VSS, passing Range through so the player can seek."""
    if MOCK:
        raise AppError(404, "No footage in mock mode")
    require_vss()
    # Only clips this app has surfaced (or VSS knows): not an open proxy into the bucket.
    if not source.startswith("s3://") or not await lookup(source):
        raise AppError(404, "Unknown clip")
    upstream = await VSS.open_stream(source, request.headers.get("range"))
    if upstream.status_code not in (200, 206):
        await upstream.aclose()
        raise AppError(502, f"Clip unavailable (HTTP {upstream.status_code})")
    headers = {k: upstream.headers[k] for k in ("content-length", "content-range", "accept-ranges")
               if k in upstream.headers}
    if "content-encoding" in upstream.headers:
        headers.pop("content-length", None)
    headers.setdefault("accept-ranges", "bytes")
    headers["cache-control"] = "private, max-age=3600"
    # VSS labels clips binary/octet-stream; say what they are so every browser plays them inline.
    return StreamingResponse(upstream.aiter_bytes(), status_code=upstream.status_code,
                             media_type="video/mp4", headers=headers,
                             background=BackgroundTask(upstream.aclose))


@app.post("/api/mine")
async def api_mine(req: MineRequest) -> dict:
    require_vss()
    result = await mine(_norm_text(req.scenario), req.top_k, req.min_similarity, req.hybrid_text_weight,
                        (req.location or "").strip() or None, req.fresh)
    return {**result, "candidates": [with_stream(c) for c in result["candidates"]]}


@app.post("/api/verify")
async def api_verify(req: VerifyRequest) -> dict:
    sources = list(dict.fromkeys(s for s in req.sources if s and s.strip()))  # exact: S3 keys are case-sensitive
    if len(sources) > 8:
        raise AppError(400, "At most 8 sources per /api/verify call.")
    scenario = _norm_text(req.scenario)
    verdicts = await asyncio.gather(*(verify(scenario, s) for s in sources))
    return {"scenario": scenario, "verdicts": dict(zip(sources, verdicts))}


@app.post("/api/similar")
async def api_similar(req: SimilarRequest) -> dict:
    """More like this: search with the seed clip's caption, excluding the seed."""
    require_vss()
    t0 = time.perf_counter()
    seed = await lookup(req.source)
    caption = (seed or {}).get("caption") or ""
    if not caption:
        raise AppError(404, "No caption known for that clip - run a search first.")
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(caption.split()))
    query = ""
    for sentence in sentences:  # first sentence or two: focused enough for the embedder
        if query and len(query) + len(sentence) > 280:
            break
        query = f"{query} {sentence}".strip()
        if len(query) >= 140:
            break
    query = query[:300]
    location = (req.location or "").strip() or None
    min_sim, notes = req.min_similarity, []
    for attempt in (0, 1):
        hits = await search(query, min(100, req.top_k + 1), min_sim, location, req.hybrid_text_weight)
        candidates = [{**h, "matched_queries": [query]} for h in hits if h["source"] != seed["source"]]
        if candidates or attempt == 1 or min_sim <= 0.08:
            break
        relaxed = round(max(0.05, min_sim / 2), 2)
        notes.append(f"No similar clips at min similarity {min_sim:.2f}; relaxed to {relaxed:.2f}.")
        min_sim = relaxed
    return {"scenario": _norm_text(req.scenario), "seed": seed["source"], "query": query, "queries": [query],
            "candidates": [with_stream(c) for c in candidates[: req.top_k]], "notes": notes, "errors": [],
            "min_similarity": min_sim, "timings": {"search_ms": _ms(t0), "total_ms": _ms(t0)}}


@app.get("/api/coverage")
async def api_coverage(refresh: bool = False) -> dict:
    require_vss()
    return coverage_view(await coverage_base(refresh))


@app.post("/api/warm")
async def api_warm() -> dict:
    global WARM_TASK
    require_vss()
    if not WARM["running"]:
        WARM.update(running=True, phase="starting", done=0, total=1, detail="Starting")
        WARM_TASK = asyncio.create_task(run_warm())
    return WARM


@app.get("/api/warm")
async def api_warm_status() -> dict:
    return WARM


@app.post("/api/export")
async def api_export(req: ExportRequest) -> dict:
    if not req.items:
        raise AppError(400, "Nothing to export - add clips to the dataset first.")
    if len(req.items) > 2000:
        raise AppError(400, "Too many items (max 2000).")
    return await export_dataset(_norm_text(req.scenario), req.items)


@app.get("/api/export/{export_id}.json")
async def api_export_file(export_id: str) -> FileResponse:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,90}", export_id):
        raise AppError(404, "Export not found.")
    path = EXPORT_DIR / f"{export_id}.json"
    if not path.is_file():
        raise AppError(404, "Export not found.")
    return FileResponse(path, media_type="application/json", filename=f"{export_id}.json")


# ============================================================================ CLI

def cli_warm() -> int:
    if not MOCK and not VSS.configured:
        print(NOT_CONFIGURED, file=sys.stderr)
        return 2

    async def runner() -> int:
        if not MOCK and WANDB_KEY:
            await asyncio.to_thread(llm.init_weave, WANDB_ENTITY, WANDB_PROJECT, WANDB_KEY)
        task = asyncio.create_task(run_warm())
        last = ""
        while not task.done():
            line = f"[warm] {WARM['phase']:<8} {WARM['done']}/{WARM['total']}  {WARM['detail']}"
            if line != last:
                print(line, flush=True)
                last = line
            await asyncio.sleep(1.0)
        state = await task
        print(f"[warm] {state['phase']}: {state['detail']}")
        for error in state["errors"][:8]:
            print(f"[warm] error: {error}")
        if COVERAGE:
            summary = coverage_view(COVERAGE)["summary"]
            print(f"[warm] coverage: {summary['gaps']} gap cells of {summary['cells']}, "
                  f"{summary['scenario_gaps']} scenarios with no verified clip anywhere; cache in {CACHE_DIR}")
        await close_clients()
        return 0 if state["phase"] == "done" else 1

    return asyncio.run(runner())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Edge-Case Miner")
    parser.add_argument("--warm", action="store_true",
                        help="precompute the coverage grid + verifications into the cache, then exit")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args(argv)
    if args.warm:
        return cli_warm()
    import uvicorn  # noqa: PLC0415

    uvicorn.run(app, host=args.host, port=args.port, proxy_headers=True, forwarded_allow_ips="*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
