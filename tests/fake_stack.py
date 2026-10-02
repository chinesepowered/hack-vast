#!/usr/bin/env python3
"""Fake VSS backend + Cosmos3-Reason + W&B Inference, for testing the app's real
(non-mock) code paths without the workshop stack.

Response shapes follow the skill docs; where the docs are vague (timing and
metadata keys) the fake deliberately alternates between two plausible shapes so
the app's normalizer gets exercised.

  python3 tests/fake_stack.py            # VSS :9001, Cosmos :9002, W&B :9003
  VSS_URL=http://127.0.0.1:9001 VSS_USERNAME=u VSS_PASSWORD=p \
  COSMOS3_REASON_URL=http://127.0.0.1:9002 GPU_BEARER_TOKEN=t \
  WANDB_API_KEY=k WANDB_INFERENCE_URL=http://127.0.0.1:9003 python3 app/main.py
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

CLIP = os.environ.get("FAKE_CLIP", os.path.join(tempfile.gettempdir(), "ecm_fake_clip.mp4"))
TOKEN = "fake-jwt"

SEGMENTS = [
    ("pie_cam-3", "toronto", "live_driving", "A pedestrian steps off the curb into the road while the ego car approaches slowly."),
    ("pie_cam-3", "toronto", "live_driving", "A cyclist rides in the right lane next to moving cars on a busy street."),
    ("pie_cam-3", "toronto", "live_driving", "The car ahead brakes hard at an intersection; brake lights are on."),
    ("i24_cam-1", "nashville", "traffic", "Dense highway traffic; a white box truck changes from lane 2 to lane 3."),
    ("i24_cam-1", "nashville", "traffic", "Vehicles slow to a stop in heavy congestion on the interstate."),
    ("neighborhood_cam-1", "neighborhood", "surveillance", "A sedan stops at the curb in front of a house and a person gets out."),
    ("neighborhood_cam-1", "neighborhood", "surveillance", "A car passes houses on a quiet residential street."),
    ("sdg_warehouse_cam-2", "warehouse3", "warehouse", "A worker walks in an aisle while a forklift moves toward them."),
    ("sdg_warehouse_cam-2", "warehouse3", "warehouse", "A pallet blocks the walkway between two shelving racks."),
    ("smartspace_cam-1", "indoor", "surveillance", "A group of people crowd a corridor near a doorway."),
]


def make_clip():
    if not os.path.exists(CLIP):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        "testsrc=duration=5:size=640x360:rate=30", "-pix_fmt", "yuv420p", CLIP], check=True)


def score(query, text):
    q = set(re.findall(r"[a-z]+", query.lower()))
    t = set(re.findall(r"[a-z]+", text.lower()))
    return round(0.2 + 0.7 * len(q & t) / max(1, len(q)), 3)


def row(i, seg, sim):
    cam, loc, cap, text = seg
    src = f"s3://team-x-vss-chunks-segments/{cam}/clip_{i:03d}_seg_{i % 6 + 1}.mp4"
    base = {"source": src, "original_video": f"s3://team-x-vss-chunks/{cam}/clip_{i:03d}.mp4",
            "similarity_score": sim, "reasoning_content": text, "tags": ["demo"], "is_public": True}
    if i % 2:  # shape A: flat metadata, *_sec timing
        base.update({"camera_id": cam, "location": loc, "capture_type": cap,
                     "segment_start_sec": 5.0 * (i % 6), "segment_end_sec": 5.0 * (i % 6) + 5})
    else:      # shape B: nested metadata, *_time timing
        base.update({"metadata": {"camera_id": cam, "location": loc, "capture_type": cap},
                     "start_time": 5.0 * (i % 6), "end_time": 5.0 * (i % 6) + 5})
    return base


class Handler(BaseHTTPRequestHandler):
    service = "vss"

    def log_message(self, *args):
        pass

    def send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def authed(self):
        return self.headers.get("Authorization", "").startswith("Bearer ")

    def do_GET(self):
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        if self.service == "vss":
            if url.path == "/api/v1/videos/stream":
                if qs.get("token", [""])[0] != TOKEN:
                    return self.send_json({"detail": "unauthorized"}, 401)
                data = open(CLIP, "rb").read()
                status, first, last = 200, 0, len(data) - 1
                ranged = re.match(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
                if ranged:
                    status, first = 206, int(ranged.group(1))
                    last = min(int(ranged.group(2) or last), last)
                self.send_response(status)
                self.send_header("Content-Type", "binary/octet-stream")  # what the live backend sends
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(last - first + 1))
                if ranged:
                    self.send_header("Content-Range", f"bytes {first}-{last}/{len(data)}")
                self.end_headers()
                return self.wfile.write(data[first:last + 1])
            if not self.authed():
                return self.send_json({"detail": "unauthorized"}, 401)
            if url.path == "/api/v1/metadata/values":
                field = qs.get("field", ["location"])[0]
                idx = {"camera_id": 0, "location": 1, "capture_type": 2}.get(field, 1)
                vals = sorted({s[idx] for s in SEGMENTS})
                return self.send_json({"field": field, "values": vals, "count": len(vals)})
            if url.path == "/api/v1/metadata/schema":
                return self.send_json({"schema": [{"name": "location"}, {"name": "camera_id"}, {"name": "capture_type"}]})
            if url.path == "/api/v1/dashboard/stats":
                return self.send_json({"overview": {"unique_videos": 10, "indexed_clips": 60},
                                       "metadata": {"location": {s[1]: 6 for s in SEGMENTS}},
                                       "objects": [{"label": "person", "segment_count": 30}, {"label": "car", "segment_count": 40}]})
            if url.path == "/api/v1/videos/detections":
                return self.send_json({"frames": [{"t": 0.0, "boxes": [{"label": "person", "bbox": [10, 10, 50, 120]}]}]})
            return self.send_json({"detail": "not found"}, 404)
        if self.service == "cosmos" and url.path == "/v1/models":
            return self.send_json({"data": [{"id": "nvidia/cosmos3-reason"}]})
        if self.service == "wandb" and url.path in ("/models", "/v1/models"):
            return self.send_json({"data": [{"id": "meta-llama/Llama-3.1-8B-Instruct"},
                                            {"id": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B"},
                                            {"id": "deepseek-ai/DeepSeek-V4-Pro-0813"},
                                            {"id": "Qwen/Qwen3.8-27B"},
                                            {"id": "Qwen/Qwen3-30B-A3B-Instruct-2507"}]})
        return self.send_json({"detail": "not found"}, 404)

    def do_POST(self):
        url = urlparse(self.path)
        req = self.body()
        if self.service == "vss":
            if url.path == "/api/v1/auth/login":
                if req.get("username") and req.get("password"):
                    return self.send_json({"access_token": TOKEN, "token_type": "bearer", "username": req["username"]})
                return self.send_json({"detail": "bad credentials"}, 401)
            if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                return self.send_json({"detail": "unauthorized"}, 401)
            if url.path == "/api/v1/search":
                if req.get("llm_top_n") == 0:
                    return self.send_json({"detail": [{"msg": "llm_top_n must be >= 1"}]}, 422)
                q = req.get("query", "")
                rows = [row(i, s, score(q, s[3])) for i, s in enumerate(SEGMENTS)]
                loc = (req.get("metadata_filters") or {}).get("location")
                rows = [r for r in rows if not loc or loc in (r.get("location"), r.get("metadata", {}).get("location"))]
                rows = [r for r in rows if r["similarity_score"] >= req.get("min_similarity", 0.1)]
                rows.sort(key=lambda r: -r["similarity_score"])
                rows = rows[: req.get("top_k", 15)]
                chunks = [{"original_video": r["original_video"], "best_match_start_sec": 0.0,
                           "best_match_end_sec": 5.0, "preview_source": r["source"], "matched_segment_count": 1}
                          for r in rows]
                return self.send_json({"results": rows, "chunk_results": chunks, "sql_query": "SELECT …",
                                       "llm_synthesis": {"response": "fake synthesis"}})
            return self.send_json({"detail": "not found"}, 404)
        if self.service == "cosmos" and url.path == "/v1/chat/completions":
            parts = req["messages"][0]["content"]
            text = next(p["text"] for p in parts if p.get("type") == "text")
            has_video = any(p.get("type") == "video_url" for p in parts)
            scenario = re.findall(r'"([^"]+)"', text)
            match = has_video and bool(scenario) and len(scenario[0]) % 2 == 0
            answer = ('<think>The clip shows a test pattern.</think>\n```json\n'
                      + json.dumps({"match": match, "confidence": 0.81, "why": "Fake Cosmos verdict on a test pattern."})
                      + "\n```")
            return self.send_json({"choices": [{"message": {"role": "assistant", "content": answer}}]})
        if self.service == "wandb" and url.path in ("/chat/completions", "/v1/chat/completions"):
            # Like the live Qwen 3.8: thinks until the budget runs out unless thinking is disabled
            # (FAKE_QWEN_BROKEN=1: never answers, to exercise the app's backup model).
            if "qwen3.8" in req.get("model", "").lower() and (
                    os.environ.get("FAKE_QWEN_BROKEN") == "1" or "chat_template_kwargs" not in req):
                return self.send_json({"choices": [{"finish_reason": "length", "message": {
                    "role": "assistant", "content": None, "reasoning_content": "Okay, let me think..."}}]})
            prompt = json.dumps(req.get("messages", [])).lower()
            if "queries" in prompt:
                content = json.dumps({"queries": ["a pedestrian walks into the street in front of a car",
                                                  "person crossing the road near moving traffic",
                                                  "pedestrian steps off the curb",
                                                  "car approaching a person in the road"]})
            else:
                content = json.dumps({"match": True, "confidence": 0.6, "why": "Caption mentions the scenario."})
            return self.send_json({"choices": [{"message": {"role": "assistant", "content": content}}]})
        return self.send_json({"detail": "not found"}, 404)


def serve(service, port):
    handler = type(f"{service}Handler", (Handler,), {"service": service})
    ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()


if __name__ == "__main__":
    make_clip()
    ports = [int(p) for p in os.environ.get("FAKE_PORTS", "9001,9002,9003").split(",")]
    for name, port in zip(("vss", "cosmos", "wandb"), ports):
        threading.Thread(target=serve, args=(name, port), daemon=True).start()
    print(f"fake stack: VSS :{ports[0]}  Cosmos :{ports[1]}  W&B :{ports[2]}", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        sys.exit(0)
