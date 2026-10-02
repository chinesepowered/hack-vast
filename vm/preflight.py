#!/usr/bin/env python3
"""Preflight check for Edge-Case Miner on the workshop VM.

Checks every service the app needs and prints a short report that is safe to
share: secret values are never printed, only whether they are set. Standard
library only, so it runs on a fresh VM before anything is installed.

Usage:  python3 vm/preflight.py
"""

import base64
import glob
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

WANDB_INFERENCE_URL = os.environ.get("WANDB_INFERENCE_URL", "https://api.inference.wandb.ai/v1")
APP_NAME = os.environ.get("APP_NAME", "edge-case-miner")
TEST_QUERY = "pedestrian near a moving car"
MAX_DIRECT_CLIP_MB = 15

SECRET_KEYS = ("PASSWORD", "GPU_BEARER_TOKEN", "WANDB_API_KEY", "SECRET_KEY", "ACCESS_KEY")
CHECK_KEYS = (
    "USERNAME", "PASSWORD", "INGRESS_URL", "PIPELINE", "GPU_BEARER_TOKEN",
    "COSMOS3_REASON_URL", "COSMOS3_REASON_MODEL", "YOLO_URL", "COSMOS_EMBED1_URL",
    "CANARY_1B_URL", "WANDB_API_KEY", "WANDB_TEAM", "WANDB_ENTITY", "WANDB_PROJECT",
)

lines = []


def out(msg=""):
    print(msg, flush=True)
    lines.append(msg)


def redact(text):
    text = re.sub(r"(?i)(token|signature|x-amz-[a-z-]+|access_token|api_key|key)=([^&\s\"']+)",
                  r"\1=<redacted>", str(text))
    return re.sub(r"(?i)bearer\s+[a-z0-9._\-]+", "Bearer <redacted>", text)


def summarize(value, maxlen=110):
    if isinstance(value, str):
        s = redact(value).replace("\n", " ")
        return repr(s[:maxlen] + ("…" if len(s) > maxlen else ""))
    if value is None or isinstance(value, (bool, int, float)):
        return repr(value)
    if isinstance(value, list):
        if len(value) > 16 and all(isinstance(x, (int, float)) for x in value[:8]):
            return f"<list[{len(value)}] of numbers>"
        head = summarize(value[0], 70) if value else ""
        return f"<list[{len(value)}]> {head}"
    if isinstance(value, dict):
        items = list(value.items())
        body = ", ".join(f"{k}: {summarize(v, 50)}" for k, v in items[:14])
        return "{" + body + (", …" if len(items) > 14 else "") + "}"
    return repr(type(value).__name__)


def load_config():
    """Env vars win; fall back to the single /config/*.config team file."""
    cfg = {}
    files = sorted(glob.glob("/config/*.config"))
    if len(files) == 1:
        with open(files[0]) as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export "):]
                key, _, val = line.partition("=")
                cfg[key.strip()] = val.strip().strip('"').strip("'")
    for key in CHECK_KEYS + ("VSS_URL", "VSS_USERNAME", "VSS_PASSWORD"):
        if os.environ.get(key):
            cfg[key] = os.environ[key]
    cfg.setdefault("INGRESS_URL", cfg.get("VSS_URL", ""))
    cfg.setdefault("USERNAME", cfg.get("VSS_USERNAME", ""))
    cfg.setdefault("PASSWORD", cfg.get("VSS_PASSWORD", ""))
    return cfg, files


def http(method, url, headers=None, body=None, timeout=60, raw=False):
    data = None
    # api.inference.wandb.ai sits behind Cloudflare, which rejects urllib's default
    # "Python-urllib" User-Agent with error 1010.
    headers = {"User-Agent": "edge-case-miner-preflight/1.0", **(headers or {})}
    if body is not None:
        data = json.dumps(body).encode()
        headers.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
            status, ctype = resp.status, resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as err:
        payload, status, ctype = err.read(), err.code, err.headers.get("Content-Type", "")
    elapsed = time.time() - t0
    if raw:
        return status, payload, ctype, elapsed
    try:
        parsed = json.loads(payload.decode() or "null")
    except ValueError:
        parsed = payload.decode(errors="replace")[:300]
    return status, parsed, ctype, elapsed


def step(title, fn):
    out(f"\n== {title}")
    try:
        return fn()
    except Exception as exc:  # report and keep going: every check is independent
        out(f"   ❌ {type(exc).__name__}: {redact(exc)[:300]}")
        return None


def main():
    cfg, files = load_config()
    out("EDGE-CASE MINER PREFLIGHT  (secrets are never printed)")
    out(f"time: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    out(f"config files: {files or 'none found in /config'}")
    for key in CHECK_KEYS:
        val = cfg.get(key, "")
        if not val:
            shown = "— unset"
        elif key in SECRET_KEYS:
            shown = "set"
        else:
            shown = redact(val)
        out(f"   {key:<22} {shown}")

    backend = cfg.get("INGRESS_URL", "").rstrip("/")
    state = {}

    def login():
        status, data, _, dt = http("POST", f"{backend}/api/v1/auth/login",
                                   body={"username": cfg["USERNAME"], "password": cfg["PASSWORD"]}, timeout=30)
        if status != 200 or not isinstance(data, dict) or "access_token" not in data:
            out(f"   ❌ login HTTP {status}: {summarize(data)}")
            return
        state["token"] = data["access_token"]
        out(f"   ✅ logged in ({dt:.1f}s)")

    def stats():
        status, data, _, dt = http("GET", f"{backend}/api/v1/dashboard/stats?scope=all",
                                   headers=auth(), timeout=60)
        if status != 200 or not isinstance(data, dict):
            out(f"   ❌ HTTP {status}: {summarize(data)}")
            return
        ov = data.get("overview", {})
        out(f"   ✅ ({dt:.1f}s) videos={ov.get('unique_videos')} indexed_clips={ov.get('indexed_clips')} "
            f"segment_rows={ov.get('segment_rows')} re_ingest_rows={ov.get('re_ingest_rows')}")
        for field, dist in (data.get("metadata") or {}).items():
            out(f"   metadata.{field}: {summarize(dist, 400)}")
        objs = data.get("objects") or []
        top = ", ".join(f"{o.get('label')}:{o.get('segment_count')}" for o in objs[:12] if isinstance(o, dict))
        out(f"   objects (label:segments): {top or summarize(objs)}")

    def locations():
        status, data, _, _ = http("GET", f"{backend}/api/v1/metadata/values?field=location&limit=50",
                                  headers=auth(), timeout=30)
        out(f"   location values (HTTP {status}): {summarize(data, 300)}")
        status, data, _, _ = http("GET", f"{backend}/api/v1/metadata/schema", headers=auth(), timeout=30)
        if isinstance(data, dict):
            names = [s.get("name") for s in data.get("schema", []) if isinstance(s, dict)]
            out(f"   filterable fields (HTTP {status}): {names}")

    def search():
        body = {"query": TEST_QUERY, "top_k": 3, "min_similarity": 0.1, "llm_top_n": 0, "include_public": True}
        status, data, _, dt = http("POST", f"{backend}/api/v1/search", headers=auth(), body=body, timeout=120)
        if status >= 400:
            out(f"   llm_top_n=0 rejected (HTTP {status}): {summarize(data, 200)} — retrying with 1")
            body["llm_top_n"] = 1
            status, data, _, dt = http("POST", f"{backend}/api/v1/search", headers=auth(), body=body, timeout=120)
        if status != 200 or not isinstance(data, dict):
            out(f"   ❌ HTTP {status}: {summarize(data)}")
            return
        out(f"   ✅ HTTP 200 in {dt:.1f}s, top-level keys: {sorted(data.keys())}")
        results = data.get("results") or []
        chunks = data.get("chunk_results") or []
        out(f"   results={len(results)} chunk_results={len(chunks)}")
        if results:
            state["source"] = results[0].get("source")
            out("   results[0]:")
            for key in sorted(results[0]):
                out(f"      {key}: {summarize(results[0][key])}")
        if chunks:
            out("   chunk_results[0]:")
            for key in sorted(chunks[0]):
                out(f"      {key}: {summarize(chunks[0][key])}")

    def download():
        if not state.get("source"):
            out("   skipped: no search hit to download")
            return
        qs = urllib.parse.urlencode({"source": state["source"], "token": state["token"]})
        status, payload, ctype, dt = http("GET", f"{backend}/api/v1/videos/stream?{qs}", timeout=120, raw=True)
        if status not in (200, 206):
            out(f"   ❌ HTTP {status} {ctype}: {redact(payload[:200])}")
            return
        state["clip"] = payload
        out(f"   ✅ {len(payload) / 1e6:.2f} MB {ctype} in {dt:.1f}s")

    def cosmos():
        url = cfg.get("COSMOS3_REASON_URL", "").rstrip("/")
        if not url:
            out("   ❌ COSMOS3_REASON_URL unset — the app will verify from captions instead")
            return
        headers = {"Authorization": f"Bearer {cfg.get('GPU_BEARER_TOKEN', '')}"}
        status, data, _, dt = http("GET", f"{url}/v1/models", headers=headers, timeout=20)
        ids = [m.get("id") for m in (data.get("data", []) if isinstance(data, dict) else [])]
        out(f"   /v1/models HTTP {status} ({dt:.1f}s): {ids}")
        model = cfg.get("COSMOS3_REASON_MODEL") or (ids[0] if ids else "nvidia/cosmos3-reason")
        clip = state.get("clip")
        if not clip:
            out("   skipped clip test: no clip downloaded")
            return
        if len(clip) > MAX_DIRECT_CLIP_MB * 1e6:
            out(f"   skipped clip test: clip is {len(clip) / 1e6:.1f} MB (the app transcodes before sending)")
            return
        content = [
            {"type": "text", "text": f'Does this clip show: "{TEST_QUERY}"? Reply ONLY with JSON '
                                     '{"match": true|false, "confidence": 0-1, "why": "<one sentence>"}'},
            {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(clip).decode()}},
        ]
        body = {"model": model, "messages": [{"role": "user", "content": content}],
                "max_tokens": 400, "temperature": 0}
        status, data, _, dt = http("POST", f"{url}/v1/chat/completions", headers=headers, body=body, timeout=180)
        if status != 200 or not isinstance(data, dict):
            out(f"   ❌ clip verify HTTP {status} ({dt:.1f}s): {summarize(data, 300)}")
            return
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        out(f"   ✅ clip verify in {dt:.1f}s: {summarize(text, 300)}")

    def wandb():
        key = cfg.get("WANDB_API_KEY", "")
        if not key:
            out("   ❌ WANDB_API_KEY unset — query expansion falls back to templates")
            return
        headers = {"Authorization": f"Bearer {key}"}
        team, project = cfg.get("WANDB_TEAM") or cfg.get("WANDB_ENTITY"), cfg.get("WANDB_PROJECT")
        if team and project:
            headers["OpenAI-Project"] = f"{team}/{project}"
        status, data, _, dt = http("GET", f"{WANDB_INFERENCE_URL}/models", headers=headers, timeout=30)
        ids = [m.get("id") for m in (data.get("data", []) if isinstance(data, dict) else [])]
        out(f"   /models HTTP {status} ({dt:.1f}s): {len(ids)} models")
        if status != 200:
            out(f"   ❌ {summarize(data, 200)}")
            return
        out(f"   ids: {', '.join(sorted(ids))}")
        # Same preference order as app/llm.py MODEL_PREFS["expand"].
        model = os.environ.get("LLM_MODEL") or next(
            (i for pref in ("qwen3.8", "deepseek-v4", "qwen3", "nemotron") for i in ids if pref in i.lower()),
            ids[0] if ids else "")
        # Same call shape as the app: Qwen3 / Nemotron think by default, "/no_think" turns that off.
        body = {"model": model, "max_tokens": 300, "temperature": 0,
                "messages": [{"role": "system", "content": "/no_think"},
                             {"role": "user", "content": 'Reply with JSON {"ok": true} and nothing else.'}]}
        status, data, _, dt = http("POST", f"{WANDB_INFERENCE_URL}/chat/completions",
                                   headers=headers, body=body, timeout=60)
        if status == 200 and isinstance(data, dict):
            msg = data.get("choices", [{}])[0].get("message", {})
            out(f"   {'✅' if msg.get('content') else '⚠️ '} chat with {model} in {dt:.1f}s: "
                f"content={summarize(msg.get('content'), 80)} "
                f"reasoning={'yes' if msg.get('reasoning_content') or msg.get('reasoning') else 'no'} "
                f"finish={data.get('choices', [{}])[0].get('finish_reason')}")
        else:
            out(f"   ❌ chat with {model} HTTP {status}: {summarize(data, 200)}")
        # The app's real query-expansion call (thinking disabled both ways, as app/llm.py does it).
        body = {"model": model, "max_tokens": 1500, "temperature": 0.3,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "system", "content": "/no_think\nYou write search queries for a video archive."},
                             {"role": "user", "content": 'Edge-case scenario: "pedestrian stepping into the road in '
                                                         'front of a moving vehicle"\nReturn ONLY JSON: '
                                                         '{"queries": [4 different search queries]}.'}]}
        status, data, _, dt = http("POST", f"{WANDB_INFERENCE_URL}/chat/completions",
                                   headers=headers, body=body, timeout=90)
        if status in (400, 422):
            out(f"   (chat_template_kwargs rejected: HTTP {status}) retrying without it")
            body.pop("chat_template_kwargs")
            status, data, _, dt = http("POST", f"{WANDB_INFERENCE_URL}/chat/completions",
                                       headers=headers, body=body, timeout=90)
        if status == 200 and isinstance(data, dict):
            choice = data.get("choices", [{}])[0]
            msg = choice.get("message", {})
            out(f"   {'✅' if msg.get('content') else '❌'} expansion with {model} in {dt:.1f}s: "
                f"finish={choice.get('finish_reason')} content={summarize(msg.get('content'), 120)} "
                f"reasoning={len(msg.get('reasoning_content') or msg.get('reasoning') or '')} chars")
        else:
            out(f"   ❌ expansion with {model} HTTP {status}: {summarize(data, 200)}")

    def kube():
        out(f"   /config contains: {sorted(os.listdir('/config')) if os.path.isdir('/config') else 'no /config'}")
        kubeconfig = os.environ.get("KUBECONFIG") or ""
        if not os.path.isfile(kubeconfig):
            kubeconfig = "/config/kubeconfig" if os.path.exists("/config/kubeconfig") else \
                next(iter(sorted(glob.glob("/config/*-k8s.yaml"))), "")
        kubectl = os.environ.get("KUBECTL") or shutil.which("kubectl")
        ns = cfg.get("USERNAME", "")
        out(f"   kubectl: {kubectl or 'not found'}  kubeconfig: {kubeconfig or 'none'}  namespace: {ns}")
        if not kubectl or not kubeconfig:
            out("   ❌ can't deploy to Kubernetes from this VM — use the local fallback (go.sh local)")
            return
        env = dict(os.environ, KUBECONFIG=kubeconfig)
        can = subprocess.run([kubectl, "auth", "can-i", "create", "deployments", "-n", ns],
                             capture_output=True, text=True, env=env, timeout=30)
        out(f"   can create deployments: {(can.stdout or can.stderr).strip()[:120]}")
        res = subprocess.run([kubectl, "-n", ns, "get", "ingress", "-o", "json"],
                             capture_output=True, text=True, env=env, timeout=30)
        if res.returncode != 0:
            out(f"   ❌ get ingress: {res.stderr.strip()[:200]}")
            return
        taken = []
        for item in json.loads(res.stdout).get("items", []):
            name = item["metadata"]["name"]
            for rule in item.get("spec", {}).get("rules", []):
                for p in rule.get("http", {}).get("paths", []):
                    out(f"   ingress {name}: {rule.get('host')} {p.get('path')}")
                    if (p.get("path") or "").startswith("/app") and name != APP_NAME:
                        taken.append(name)
        if taken:
            out(f"   ⚠️  /app is already used by {sorted(set(taken))} — deploy.sh will pick another path")

    def auth():
        return {"Authorization": f"Bearer {state.get('token', '')}"}

    if not backend:
        out("\n❌ INGRESS_URL not found — is this the workshop VM?")
    else:
        step("1. VSS login", login)
        if state.get("token"):
            step("2. What's indexed (dashboard)", stats)
            step("3. Filter values", locations)
            step(f'4. Search "{TEST_QUERY}"', search)
            step("5. Download one clip", download)
    def python_env():
        import platform
        pip = subprocess.run(["python3", "-m", "pip", "--version"], capture_output=True, text=True)
        have = []
        for mod in ("fastapi", "uvicorn", "httpx"):
            have.append(f"{mod}={'yes' if subprocess.run(['python3', '-c', f'import {mod}'], capture_output=True).returncode == 0 else 'no'}")
        out(f"   python {platform.python_version()}  pip: {(pip.stdout or pip.stderr).strip()[:60] or 'missing'}  {' '.join(have)}")
        out(f"   ffmpeg: {shutil.which('ffmpeg') or 'not found'}")

    step("6. Cosmos3-Reason (direct clip verification)", cosmos)
    step("7. W&B Inference (query expansion LLM)", wandb)
    step("8. Kubernetes", kube)
    step("9. Python on this VM (for the local fallback)", python_env)

    out("\n>>> Copy everything from 'EDGE-CASE MINER PREFLIGHT' to here and send it to Claude.")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "preflight-report.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
