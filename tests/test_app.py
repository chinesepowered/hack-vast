"""Tests for Edge-Case Miner: response normalizing (live VSS row shapes), verdict parsing,
redaction, and an offline API smoke test (MOCK=1).

    python -m pytest tests/test_app.py      # or, without pytest:
    python tests/test_app.py
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import llm  # noqa: E402
import vss  # noqa: E402

# One results[] row as the live backend returns it (flat; object_counts is a JSON *string*).
LIVE_ROW = {
    "source": "s3://team-x-vss-chunks-segments/team-x/20260901_pie_set03_chunk_0007_seg_003.mp4",
    "original_video": "s3://team-x-vss-chunks/team-x/20260901_pie_set03_chunk_0007.mp4",
    "filename": "20260901_pie_set03_chunk_0007.mp4",
    "similarity_score": 0.3121,
    "reasoning_content": "A pedestrian in a red coat steps off the curb as a white sedan approaches.",
    "camera_id": "pie_cam-3", "location": "toronto", "capture_type": "streets",
    "segment_start_sec": 10.0, "segment_end_sec": 15.0, "segment_number": 3, "total_segments": 6,
    "duration": 5.0,
    "object_classes": "bench,car,donut,person",
    "object_counts": '{"car": 5, "person": 1, "bench": 1, "donut": 1}',
    "perception_json": '{"source": "yolo11_coco", "frames": [1, 2, 3]}',
    "perception_ok": True, "max_detection_conf": 0.91,
    "detection_count": None, "detection_frame_count": None, "detection_sidecar_uri": None,
    "cosmos_model": "nvidia/cosmos3-nano-reasoner", "upload_timestamp": "2026-09-01T12:00:00Z",
    "tags": [], "is_public": True, "extra_metadata": None, "tokens_used": 812, "cached_prompt_tokens": 0,
}


def _timeline(prefix, n=6):
    return [{"segment_number": i, "segment_start_sec": 5.0 * (i - 1), "segment_end_sec": 5.0 * i,
             "source": f"{prefix}_seg_{i:03d}.mp4", "reasoning_content": f"segment {i}",
             "object_classes": "car", "object_counts": '{"car": %d}' % i, "perception_ok": True,
             "similarity_score": 0.1, "is_search_match": i == 3, "is_best_match": i == 3} for i in range(1, n + 1)]


def test_normalize_live_flat_row():
    hit = vss.normalize_hit(LIVE_ROW)
    assert hit["source"] == LIVE_ROW["source"]
    assert hit["original_video"] == LIVE_ROW["original_video"]
    assert (hit["start_sec"], hit["end_sec"], hit["segment_number"]) == (10.0, 15.0, 3)
    assert (hit["camera_id"], hit["location"], hit["capture_type"]) == ("pie_cam-3", "toronto", "streets")
    assert hit["similarity"] == 0.3121
    assert hit["caption"].startswith("A pedestrian in a red coat")
    assert hit["objects"] == ["bench", "car", "donut", "person"]
    assert hit["object_counts"] == {"car": 5, "person": 1, "bench": 1, "donut": 1}
    assert list(hit["object_counts"])[0] == "car"  # sorted by count
    assert hit["raw"]["perception_json"].startswith("<")  # large blobs are omitted from the debug copy


def test_normalize_search_response_with_timeline_context():
    prefix = LIVE_ROW["source"].rsplit("_seg_", 1)[0]
    data = {
        "query": "pedestrian", "total": 2, "chunk_total": 1, "embedding_time_ms": 12.0, "search_time_ms": 40.0,
        "permission_filtered": 0, "llm_synthesis": {"response": "..."}, "sql_query": "SELECT ...",
        "results": [LIVE_ROW, dict(LIVE_ROW)],  # duplicate source is dropped
        "chunk_results": [{
            "original_video": LIVE_ROW["original_video"], "filename": LIVE_ROW["filename"],
            "best_match_start_sec": 10.0, "best_match_end_sec": 15.0, "best_segment_number": 3,
            "preview_source": LIVE_ROW["source"], "matched_segment_count": 1, "chunk_duration_sec": 30.0,
            "camera_id": "pie_cam-3", "location": "toronto", "capture_type": "streets",
            "similarity_score": 0.3121, "reasoning_content": "...", "stream_id": None, "total_segments": 6,
            "timeline": _timeline(prefix),
        }],
    }
    hits, meta = vss.normalize_search(data)
    assert len(hits) == 1 and meta["total"] == 2
    ctx = hits[0]["context"]
    assert ctx["prev"]["source"].endswith("_seg_002.mp4") and ctx["prev"]["start_sec"] == 5.0
    assert ctx["next"]["source"].endswith("_seg_004.mp4") and ctx["next"]["object_counts"] == {"car": 4}


def test_normalize_alternative_shapes():
    nested = {"source": "s3://b/k.mp4", "similarity": "0.25", "caption": "x", "start_time": 5, "end_time": 10,
              "metadata": {"camera_id": "cam", "location": "nashville"},
              "extra_metadata": '{"capture_type": "traffic"}'}
    hit = vss.normalize_hit(nested)
    assert (hit["start_sec"], hit["end_sec"], hit["similarity"]) == (5.0, 10.0, 0.25)
    assert (hit["camera_id"], hit["location"], hit["capture_type"]) == ("cam", "nashville", "traffic")
    assert vss.normalize_hit({"similarity_score": 0.4}) is None  # no source → dropped
    hits, _ = vss.normalize_search({"results": [], "chunk_results": [
        {"original_video": "s3://b/p.mp4", "preview_source": "s3://b/p_seg_2.mp4", "similarity_score": 0.2,
         "best_match_start_sec": 5.0, "best_match_end_sec": 10.0, "location": "indoor"}]})
    assert hits[0]["source"] == "s3://b/p_seg_2.mp4" and hits[0]["start_sec"] == 5.0 and hits[0]["from_chunk"]
    assert vss.normalize_search(["not", "a", "dict"])[0] == []


def test_parse_verdict_variants():
    v = llm.parse_verdict('<think>looking</think>\n```json\n{"match": true, "confidence": 0.81, "why": "Car brakes."}\n```')
    assert v == {"match": True, "confidence": 0.81, "why": "Car brakes."}
    assert llm.parse_verdict("{'match': False, 'confidence': 90, 'why': 'empty road'}")["confidence"] == 0.9
    assert llm.parse_verdict('Sure! {"match": "no", "why": "nobody crosses"} done')["match"] is False
    assert llm.parse_verdict("Yes, a cyclist rides beside the cars.")["match"] is True
    bad = llm.parse_verdict("<think>unfinished reasoning about the clip")
    assert bad["match"] is None and bad["why"]
    assert llm.parse_verdict("")["match"] is None


def test_query_parsing_and_model_pick():
    assert llm._clean_queries(llm._queries_from_text('{"queries": ["a b c", "1. d e f", "A B C"]}'), 4) == ["a b c", "d e f"]
    assert llm._clean_queries(llm._queries_from_text("1. car brakes hard\n2) truck merges"), 4) == ["car brakes hard", "truck merges"]
    assert llm.pick_model(["meta/llama", "nvidia/NVIDIA-Nemotron-X", "nvidia/nemotron-embed"]) == "nvidia/NVIDIA-Nemotron-X"
    assert llm.fallback_queries("pedestrian crossing at night")[0] == "pedestrian crossing at night"


def test_cosmos_retries_truncated_reasoning():
    import asyncio

    import httpx

    import cosmos

    budgets = []

    def handler(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "nvidia/cosmos3-nano-reasoner"}]})
        body = json.loads(request.content)
        budgets.append(body["max_tokens"])
        assert body["messages"][0]["content"][1]["video_url"]["url"].startswith("data:video/mp4;base64,")
        if len(budgets) == 1:  # a reasoning model that ran out of tokens mid-thought
            return httpx.Response(200, json={"choices": [{"finish_reason": "length",
                                                          "message": {"content": "<think>The cyclist is"}}]})
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": '<think>ok</think>{"match": true, "confidence": 0.7, "why": "A cyclist rides beside cars."}'}}]})

    client = cosmos.CosmosClient("http://cosmos.test:8001/v1", "t", "")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    verdict = asyncio.run(client.judge_video("cyclist riding next to moving cars", b"\x00fake-mp4"))
    assert verdict["match"] is True and verdict["method"] == "cosmos-video"
    assert verdict["model"] == "nvidia/cosmos3-nano-reasoner" and budgets == list(cosmos.TOKEN_BUDGETS)


def test_redact():
    fake_jwt = ".".join(["eyJ" + "x" * 12, "eyJ" + "y" * 12, "z" * 10])  # built at runtime: no token literal
    text = f"GET /stream?source=s3%3A%2F%2Fb&token={fake_jwt} Bearer abc.def; also bare {fake_jwt}"
    out = vss.redact(text)
    assert "eyJ" not in out and "abc.def" not in out and "token=***" in out


def test_mock_api_smoke():
    cache = tempfile.mkdtemp(prefix="ecm-test-")
    os.environ["MOCK"] = "1"
    os.environ["CACHE_DIR"] = cache
    try:
        _smoke()
    finally:
        shutil.rmtree(cache, ignore_errors=True)


def _smoke():
    from fastapi.testclient import TestClient

    import main  # noqa: PLC0415 - reads env at import

    with TestClient(main.app) as client:
        assert client.get("/health").json() == {"ok": True, "mock": True}
        assert client.get("/api/config").json()["mock"] is True
        assert "Edge-Case Miner" in client.get("/").text
        mine = client.post("/api/mine", json={"scenario": "pedestrian crossing at night"}).json()
        assert mine["queries"][0] == "pedestrian crossing at night" and mine["candidates"]
        cand = mine["candidates"][0]
        assert cand["stream_url"] is None and cand["matched_queries"] and cand["object_counts"]
        sources = [c["source"] for c in mine["candidates"][:4]]
        verdicts = client.post("/api/verify", json={"scenario": mine["scenario"], "sources": sources}).json()["verdicts"]
        assert set(verdicts) == set(sources) and all(v["match"] is not None for v in verdicts.values())
        nine = [f"s3://b/{i}.mp4" for i in range(9)]
        assert client.post("/api/verify", json={"scenario": "x y", "sources": nine}).status_code == 400
        similar = client.post("/api/similar", json={"scenario": mine["scenario"], "source": sources[0]}).json()
        assert all(c["source"] != sources[0] for c in similar["candidates"])
        coverage = client.get("/api/coverage").json()
        assert len(coverage["scenarios"]) == 10 and len(coverage["locations"]) == 6
        cell = coverage["cells"]["night-crossing"]["toronto"]
        assert set(cell) >= {"hits", "verified_matches", "verified_total", "gap", "status"}
        export = client.post("/api/export", json={"scenario": "night", "items": [{"source": s} for s in sources]}).json()
        manifest = client.get("/" + export["manifest_url"]).json()
        assert manifest["counts"]["items"] == 4 and "token=" not in json.dumps(manifest)


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        try:
            fn()
            print(f"ok   {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failed else 0)
