#!/usr/bin/env python3
"""
subserver.py — Hybrid NX/AGX inference server

Runs on Orin NX. For each user query:
  1. Classifies it with probabilitic confidence (logprob-style via score
     + empirical sampling), KL divergence, and online k-means clustering.
  2. If confidently medical (confidence ≥ CONFIDENCE_THRESHOLD) → answer
     locally on NX.
  3. Otherwise → forward to AGX, relay response.
  4. All model ops (classify + infer) serialised through a thread-safe
     queue to prevent resource contention on NX.
  5. Every query + response + routing decision logged to AGX.

Usage:
  python subserver.py --agx-ip 100.x.y.z             # default port 8765
  python subserver.py --agx-ip 100.x.y.z --port 9000
"""

import json
import logging
import math
import os
import queue
import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import requests
from flask import Flask, Response, jsonify, request, send_file
from flask_sock import Sock

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
)
log = logging.getLogger("subserver")

MODEL = os.path.expanduser(
    "~/llama/HiSLM-8G/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
)
LLAMA_CLI = os.path.expanduser(
    "~/llama/llama.cpp/build-x64-linux-gcc-release/bin/llama-cli"
)
try:
    _BASE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _BASE = "."
ORIN_HTML = os.path.join(_BASE, "orin_index.html")

app = Flask(__name__)
sock = Sock(app)

agx_host: str = ""
agx_port: int = 8000
node_name: str = "nx-subserver"

# ── Routing thresholds ───────────────────────────────────────────────
# Only route to NX if both:
#   1. is_medical == True
#   2. confidence >= CONFIDENCE_THRESHOLD
# Otherwise → AGX.
CONFIDENCE_THRESHOLD = 0.7

# ── Classification ──────────────────────────────────────────────────

MEDICAL_KEYWORDS = {
    # ── Core medical terms (stems cover plurals via substring matching) ──
    "symptom", "diagnosis", "diagnose", "treatment", "treat", "disease",
    "patient", "medication", "dosage", "surgery", "surgical",
    "health", "medical", "medicine", "clinical", "clinic",
    "drug", "infection", "therapy", "therapist",
    "doctor", "physician", "nurse", "hospital",
    "prescription", "pharmacy", "pharmaceutical",
    "pain", "fever", "cough", "injury", "wound", "wound",
    "vaccine", "vaccination", "antibiotic",
    "examination", "exam", "test result", "lab result",
    "blood", "heart", "cardiac", "lung", "pulmonary", "brain",
    "cancer", "tumor", "diabetes", "diabetic",
    "hypertension", "blood pressure", "cholesterol",
    "pneumonia", "asthma", "allergy", "allergic",
    "stroke", "seizure", "concussion", "depression", "anxiety",
    "epilepsy", "arthritis", "obesity", "insulin",
    "kidney", "renal", "liver", "hepatic", "bone", "fracture",
    "muscle", "joint", "spine", "spinal", "skin", "rash",
    "nausea", "vomiting", "dizziness", "fatigue", "swelling",
    "chemotherapy", "radiation", "dialysis", "transplant",
    "MRI", "CT scan", "X-ray", "ultrasound", "endoscopy",
    "symptom", "symptoms",  # explicit plurals
    "medications", "infections", "diseases", "treatments",
    "diagnosed", "prescribed", "hospitalized",
    "recovery", "rehabilitation", "physiotherapy",
    "diet", "nutrition", "exercise", "wellness",
    "emergency", "ambulance", "first aid",
    "overdose", "poisoning", "side effect", "side effects",
    "statins", "opioid", "anesthesia",
    "pulse", "temperature", "weight", "height",
    # ── Greetings & common social phrases ──
    "hello", "hi", "hey", "greetings", "good morning",
    "good evening", "good afternoon", "howdy", "how are you",
    "what's up", "nice to meet you", "thank you", "thanks",
    "how's it going", "what's new", "long time no see",
}

CLASSIFY_SCORE_PROMPT = (
    "Rate the medical relevance of this query from 0.0 to 1.0.\n"
    "0.0 = definitely NOT medical (e.g., programming, recipes)\n"
    "0.5 = uncertain or partially related\n"
    "1.0 = definitely medical (e.g., symptoms, diagnosis, treatment)\n"
    "Output only the number.\n\n"
    "Query: {query}\n"
    "Score: "
)

# ── Probabilitic confidence helpers ─────────────────────────────────

N_SAMPLES = 3              # LLM calls per classify
TEMP_SCHEDULE = [0.0, 0.4, 0.8]  # deterministic + mild + moderate stochastic


def _parse_score(raw: str) -> float | None:
    """Extract a 0.0–1.0 float from model output.
    
    Strips the llama-cli banner (everything before the last "Score: ")
    to avoid picking up numbers from the ASCII art or perf stats.
    """
    text = raw.strip()
    # Keep only text after the last "Score:" marker to skip banner
    idx = text.rfind("Score:")
    if idx >= 0:
        text = text[idx + len("Score:"):].strip()
    # Also strip after [Prompt: or [ Generation to skip perf stats
    for marker in ["[Prompt:", "[ Generation:", "[ prompt:", "Exiting", "\n>"]:
        m = text.find(marker)
        if m >= 0:
            text = text[:m].strip()
    # Now parse the remaining text for 0.0-1.0
    m = re.search(r"([01](?:\.\d+)?|\.\d+)", text)
    if m:
        val = float(m.group(1))
        if 0.0 <= val <= 1.0:
            return val
    m = re.search(r"[01]", text)
    if m:
        return float(m.group())
    return None


def _run_llama_score(query: str, temp: float, timeout_s: int = 30) -> float | None:
    """Run the classifier prompt at a given temperature and return parsed score."""
    prompt = CLASSIFY_SCORE_PROMPT.replace("{query}", query)
    cmd = [
        LLAMA_CLI, "-m", MODEL,
        "-p", prompt,
        "-n", "8",
        "--no-display-prompt", "--single-turn", "--simple-io", "--log-disable",
        "-c", "512", "--temp", str(temp),
    ]
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
        out, _ = proc.communicate(timeout=timeout_s)
        proc.wait()
        return _parse_score(out)
    except Exception as exc:
        log.warning(f"llama-cli (temp={temp}) error: {exc}")
        return None


def _compute_metrics(scores: list[float]) -> dict:
    """From a list of scores, compute probabilitic confidence metrics.

    Treats each score's ≥0.5 as a binary vote → empirical P(medical).
    Returns dict with:
      - p_med:      P(medical) — fraction of samples above 0.5
      - confidence: max(p_med, 1 - p_med) — how decisive the vote is
      - kl_div:     KL(P_empirical || Uniform) in bits
      - mean_score: average raw score across samples
      - std_score:  std of raw scores (high = uncertain)
    """
    n = len(scores)
    votes_med = sum(1 for s in scores if s >= 0.5)
    p_med = votes_med / n

    confidence = max(p_med, 1.0 - p_med)
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores, ddof=1)) if n > 1 else 0.0

    # KL(P || uniform): high KL → model is far from guessing
    # P = [p_med, 1-p_med], Q = [0.5, 0.5]
    eps = 1e-12
    p_clamped = np.clip(p_med, eps, 1.0 - eps)
    kl = p_clamped * np.log2(p_clamped / 0.5) + \
         (1.0 - p_clamped) * np.log2((1.0 - p_clamped) / 0.5)

    return {
        "p_med": round(p_med, 4),
        "confidence": round(confidence, 4),
        "kl_div": round(float(kl), 4),
        "mean_score": round(mean_score, 4),
        "std_score": round(std_score, 4),
        "n_samples": n,
    }


# ── Online K-Means (streaming, 3 clusters) ──────────────────────────
# Features per query:  [confidence, kl_div, keyword_ratio, query_len_norm]
# Clusters correspond to: "confident-medical", "confident-non-medical", "uncertain"

K_HISTORY = deque(maxlen=200)      # raw feature vectors for refit
K_CENTERS: np.ndarray | None = None  # (3, n_features)
K_COUNTS: np.ndarray | None = None   # (3,)  samples per cluster
K_N_FEATURES = 3


def _make_features(confidence: float, kl: float, kw_ratio: float,
                   query_len: int) -> np.ndarray:
    """Build a normalised 3-d feature vector for k-means."""
    return np.array([
        confidence,                      # already 0-1
        min(kl, 2.0) / 2.0,             # KL range ~0-inf, cap at 2.0
        min(kw_ratio, 1.0),             # 0-1
    ])


def _kmeans_init_batch(features: np.ndarray, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Simple k-means++ init on a batch of features."""
    n = features.shape[0]
    rng = np.random.default_rng(42)
    centers = [features[rng.integers(n)]]
    for _ in range(1, k):
        dists = np.min(
            np.array([np.linalg.norm(features - c, axis=1) for c in centers]),
            axis=0,
        )
        dist_sum = dists.sum()
        if dist_sum < 1e-12:
            # All points identical → add uniform jitter
            jitter = rng.uniform(-0.01, 0.01, size=features[0].shape)
            centers.append(features[0] + jitter)
        else:
            probs = dists / dist_sum
            centers.append(features[rng.choice(n, p=probs)])
    return np.array(centers), np.ones(k)


def _kmeans_assign(centers: np.ndarray, x: np.ndarray) -> int:
    """Return index of nearest cluster center."""
    dists = np.linalg.norm(centers - x, axis=1)
    return int(np.argmin(dists))


def _kmeans_update(centers: np.ndarray, counts: np.ndarray,
                   x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Online k-means: move nearest center toward x."""
    idx = _kmeans_assign(centers, x)
    counts[idx] += 1.0
    lr = 1.0 / counts[idx]
    centers[idx] = (1.0 - lr) * centers[idx] + lr * x
    return centers, counts


def update_kmeans(confidence: float, kl: float, kw_ratio: float,
                  query_len: int) -> dict:
    """Update the online k-means model and return cluster info.

    Returns:
      {"cluster": int, "label": str, "dist_to_center": float}
      label = 0 → "confident-medical", 1 → "confident-non-medical", 2 → "uncertain"
      Negative labels mean "not enough data yet".
    """
    global K_CENTERS, K_COUNTS

    x = _make_features(confidence, kl, kw_ratio, query_len)
    K_HISTORY.append(x)

    if K_CENTERS is None:
        if len(K_HISTORY) < 9:
            return {"cluster": -1, "label": "cold-start",
                    "dist_to_center": 0.0}
        # Batch init with k-means++
        arr = np.array(K_HISTORY)
        K_CENTERS, K_COUNTS = _kmeans_init_batch(arr, k=3)

    K_CENTERS, K_COUNTS = _kmeans_update(K_CENTERS, K_COUNTS, x)
    idx = _kmeans_assign(K_CENTERS, x)
    dist = float(np.linalg.norm(K_CENTERS[idx] - x))

    # Label clusters by their mean confidence value
    labels = {0: "confident-medical", 1: "confident-non-medical", 2: "uncertain"}
    cluster_order = np.argsort(K_CENTERS[:, 0])  # sort by confidence ascending
    label_map = {
        cluster_order[0]: "confident-non-medical",
        cluster_order[1]: "uncertain",
        cluster_order[2]: "confident-medical",
    }

    return {
        "cluster": int(idx),
        "label": label_map.get(idx, f"cluster-{idx}"),
        "dist_to_center": round(dist, 4),
    }


# ── Main classifier ─────────────────────────────────────────────────

def classify_with_confidence(query: str) -> dict:
    """Classify query and return a result dict with all metrics.

    Returns:
      {
        "is_medical": bool,
        "confidence": float (0-1),
        "p_med": float,      # empirical P(medical)
        "kl_div": float,     # KL divergence from uniform
        "method": str,
        "kmeans": {...} or None,
        "mean_score": float,
        "std_score": float,
        "n_samples": int,
        "scores": [float],
      }
    """
    t0 = time.time()

    # Stage 1: keyword pre-filter (instant → high confidence)
    query_lower = query.lower()
    kw_count = 0
    for kw in MEDICAL_KEYWORDS:
        # Use word-boundary matching to avoid "hi" matching "history"
        if re.search(r'\b' + re.escape(kw) + r'\b', query_lower):
            kw_count += 1
    kw_ratio = min(kw_count / 5.0, 1.0)  # 5+ keyword hits = max ratio

    if kw_count > 0:
        elapsed = (time.time() - t0) * 1000
        log.info(f"Keyword match ({kw_count}): {query[:40]!r} → medical "
                 f"(c=0.95, {elapsed:.1f}ms)")

        metrics = {
            "p_med": 1.0, "confidence": 0.95, "kl_div": 1.0,
            "mean_score": 0.95, "std_score": 0.0, "n_samples": 1, "scores": [0.95],
        }
        kmeans = update_kmeans(0.95, 1.0, kw_ratio, len(query))
        return {
            "is_medical": True,
            "confidence": 0.95,
            "method": "keyword",
            "kmeans": kmeans,
            **metrics,
        }

    # Stage 2: multi-sample LLM scoring (logprob approximation)
    scores: list[float] = []
    for temp in TEMP_SCHEDULE:
        s = _run_llama_score(query, temp, timeout_s=30)
        if s is not None:
            scores.append(s)
        else:
            log.warning(f"Score sample failed at temp={temp}")

    # If ALL samples failed, fall back to safe default
    if not scores:
        log.error(f"All {N_SAMPLES} score samples failed for {query[:40]!r}")
        return {
            "is_medical": True, "confidence": 0.5, "method": "all_failed",
            "p_med": 0.5, "kl_div": 0.0, "mean_score": 0.5, "std_score": 0.0,
            "n_samples": 0, "scores": [],
            "kmeans": update_kmeans(0.5, 0.0, kw_ratio, len(query)),
        }

    metrics = _compute_metrics(scores)
    p_med = metrics["p_med"]
    confidence = metrics["confidence"]
    kl = metrics["kl_div"]
    # Use mean_score >= 0.4 as threshold (smoother than binary voting with few samples)
    is_med = metrics["mean_score"] >= 0.4

    # K-means clustering
    kmeans = update_kmeans(confidence, kl, kw_ratio, len(query))

    elapsed = (time.time() - t0) * 1000
    log.info(
        f"Classify: {query[:40]!r} → p_med={p_med:.2f} "
        f"c={confidence:.2f} kl={kl:.2f} "
        f"method=llm n={len(scores)} ({elapsed:.0f}ms)"
    )

    return {
        "is_medical": is_med,
        "confidence": round(confidence, 4),
        "method": "llm",
        "kmeans": kmeans,
        **metrics,
        "scores": [round(s, 4) for s in scores],
    }


# ── Model Queue (serialises all llama-cli operations) ────────────────

_model_queue: queue.Queue = queue.Queue()
_worker_thread: threading.Thread | None = None


def _run_classify(query: str) -> dict:
    """Blocking classify call (called from worker thread)."""
    return classify_with_confidence(query)


def _run_infer(prompt: str, max_tokens: int = 512) -> str:
    """Blocking inference call (called from worker thread)."""
    cmd = [
        LLAMA_CLI, "-m", MODEL,
        "-p", prompt,
        "-n", str(max_tokens),
        "--no-display-prompt", "--single-turn", "--simple-io",
        "-c", "4096", "--temp", "0.7",
    ]
    log.info(f"Local inference (queued): n={max_tokens}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
    )
    out, _ = proc.communicate()
    proc.wait()
    return _extract_response(out)


def _model_worker():
    """Process classify/infer requests from the queue one-by-one."""
    log.info("Model worker started")
    while True:
        item = _model_queue.get()
        if item is None:
            log.info("Model worker stopping")
            _model_queue.task_done()
            break
        op = item["op"]
        try:
            if op == "classify":
                result = _run_classify(item["query"])
            elif op == "infer":
                result = _run_infer(item["prompt"], item.get("max_tokens", 512))
            else:
                result = {"error": f"unknown op: {op}"}
            item["result_holder"]["result"] = result
        except Exception as exc:
            log.error(f"Model worker {op} failed: {exc}")
            item["result_holder"]["result"] = {"error": str(exc)}
        finally:
            item["event"].set()
            _model_queue.task_done()


def _ensure_worker():
    global _worker_thread
    if _worker_thread is None or not _worker_thread.is_alive():
        _worker_thread = threading.Thread(target=_model_worker, daemon=True)
        _worker_thread.start()
        log.info("Model worker launched")


def enqueue_classify(query: str, timeout_s: float = 60,
                     timing: dict | None = None) -> dict:
    """Enqueue a classify request and wait for the result dict."""
    _ensure_worker()
    t0 = time.time()
    result_holder: dict = {}
    event = threading.Event()
    _model_queue.put({
        "op": "classify", "query": query,
        "result_holder": result_holder, "event": event,
    })
    if not event.wait(timeout=timeout_s):
        log.error(f"Classify timed out after {timeout_s}s")
        fallback = {
            "is_medical": True, "confidence": 0.5, "method": "timeout",
            "p_med": 0.5, "kl_div": 0.0, "mean_score": 0.5,
            "std_score": 0.0, "n_samples": 0, "scores": [],
            "kmeans": None,
        }
        if timing is not None:
            timing["classify_ms"] = (time.time() - t0) * 1000
            timing["classify_method"] = "timeout"
            timing["confidence"] = 0.5
        return fallback
    r = result_holder.get("result", {})
    if isinstance(r, dict) and "error" in r:
        fallback = {
            "is_medical": True, "confidence": 0.5, "method": "queue_error",
            "p_med": 0.5, "kl_div": 0.0, "mean_score": 0.5,
            "std_score": 0.0, "n_samples": 0, "scores": [],
            "kmeans": None,
        }
        if timing is not None:
            timing["classify_ms"] = (time.time() - t0) * 1000
            timing["classify_method"] = "queue_error"
            timing["confidence"] = 0.5
        return fallback
    if timing is not None:
        timing["classify_ms"] = (time.time() - t0) * 1000
        timing["classify_method"] = r.get("method", "queue")
        timing["confidence"] = r.get("confidence", 0.5)
        timing["p_med"] = r.get("p_med", 0.5)
        timing["kl_div"] = r.get("kl_div", 0.0)
        timing["kmeans_label"] = (r.get("kmeans") or {}).get("label", "cold-start")
    return r


def enqueue_infer(prompt: str, max_tokens: int = 512,
                  timeout_s: float = 600) -> str:
    _ensure_worker()
    result_holder: dict = {}
    event = threading.Event()
    _model_queue.put({
        "op": "infer", "prompt": prompt, "max_tokens": max_tokens,
        "result_holder": result_holder, "event": event,
    })
    if not event.wait(timeout=timeout_s):
        log.error(f"Inference timed out after {timeout_s}s")
        return "[NX timeout]"
    r = result_holder.get("result", "[NX error]")
    if isinstance(r, dict) and "error" in r:
        return f"[NX error: {r['error']}]"
    return r if isinstance(r, str) else "[NX error]"


# ── AGX Communication (REST-based) ─────────────────────────────────

def agx_base_url() -> str:
    return f"http://{agx_host}:{agx_port}"


def fetch_from_agx(query: str) -> str:
    base = agx_base_url()
    log.info(f"Forwarding to AGX: {base}/send")
    try:
        r = requests.post(
            f"{base}/send",
            json={"sender": node_name, "text": query},
            timeout=120,
        )
        r.raise_for_status()
        body = r.json()
        # AGX may return the reply inline — use it directly
        if "reply" in body:
            reply = body["reply"]
            text = reply.get("text", "")
            if text:
                log.info(f"AGX reply inline ({len(text)} chars)")
                return text
    except requests.RequestException as exc:
        log.error(f"AGX REST send failed: {exc}")
        return "[AGX unreachable]"

    deadline = time.time() + 120
    poll_interval = 1.0
    last_seen = time.time()

    while time.time() < deadline:
        try:
            r = requests.get(f"{base}/messages?limit=20", timeout=10)
            r.raise_for_status()
            msgs = r.json().get("messages", [])
        except requests.RequestException as exc:
            log.warning(f"AGX poll failed: {exc}")
            time.sleep(poll_interval)
            continue
        for msg in reversed(msgs):
            role = msg.get("role", "")
            sender = msg.get("sender", "")
            text = msg.get("text", "")
            if role == "server" and sender != "system":
                msg_time = _parse_timestamp(msg.get("timestamp", ""))
                if msg_time and msg_time > last_seen:
                    log.info(f"AGX response ({len(text)} chars)")
                    return text
        time.sleep(poll_interval)
    log.error("AGX response timed out")
    return "[AGX timeout]"


def _parse_timestamp(ts: str) -> float:
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.timestamp()
    except (ValueError, AttributeError):
        return 0.0


def log_to_agx(entry: dict):
    try:
        requests.post(
            f"{agx_base_url()}/log",
            json={"sender": node_name, "type": "subserver_log", "payload": entry},
            timeout=5,
        )
    except requests.RequestException:
        pass


# ── Local inference helpers ─────────────────────────────────────────

def build_prompt(user_msg: str, system: str = "",
                 messages: list[dict] | None = None) -> str:
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>")
    if messages:
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "assistant":
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
            else:
                parts.append(f"<|im_start|>user\n{content}<|im_end|>")
    parts.append(f"<|im_start|>user\n{user_msg}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _extract_response(raw: str) -> str:
    lines = raw.split("\n")
    last_asst = -1
    for i, line in enumerate(lines):
        if line.startswith("<|im_start|>assistant"):
            last_asst = i
    if last_asst < 0:
        return ""
    sep = -1
    for i in range(last_asst + 1, len(lines)):
        if not lines[i].strip():
            sep = i
            break
    if sep < 0:
        return ""
    gen_lines = []
    for j in range(sep + 1, len(lines)):
        if lines[j].startswith("[ Prompt:") or lines[j].startswith("Exiting"):
            break
        gen_lines.append(lines[j])
    return "\n".join(gen_lines).strip()


# ── Routes ──────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file(ORIN_HTML)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": str(MODEL)})


@app.route("/classify", methods=["POST"])
def classify():
    text = request.json.get("text", "")
    t0 = time.time()
    timing: dict = {}
    result = enqueue_classify(text, timing=timing)
    total_ms = (time.time() - t0) * 1000
    is_med = result["is_medical"]
    conf = result["confidence"]
    route_nx = is_med and conf >= CONFIDENCE_THRESHOLD
    return jsonify({
        "is_medical": is_med,
        "confidence": conf,
        "p_med": result.get("p_med", 0.5),
        "kl_div": result.get("kl_div", 0.0),
        "method": result.get("method", "unknown"),
        "n_samples": result.get("n_samples", 0),
        "kmeans": result.get("kmeans"),
        "route": "NX" if route_nx else "AGX",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "classify_ms": timing.get("classify_ms", 0),
        "total_ms": round(total_ms, 1),
    })


@app.route("/chat", methods=["POST"])
def chat():
    t_start = time.time()
    data = request.get_json(force=True)
    user_msg = data.get("message", data.get("content", ""))
    system = data.get("system", "")
    stream = data.get("stream", True)
    messages = data.get("messages")

    timing: dict = {}
    result = enqueue_classify(user_msg, timing=timing)
    is_med = result["is_medical"]
    conf = result["confidence"]
    route_agx = not (is_med and conf >= CONFIDENCE_THRESHOLD)

    log.info(
        f"Query: {user_msg[:60]!r}  "
        f"is_med={is_med}  c={conf:.3f}  "
        f"kl={result.get('kl_div', 0):.2f}  "
        f"kmeans={result.get('kmeans', {}).get('label', '?')}  "
        f"route={'AGX' if route_agx else 'NX'}"
    )

    if route_agx:
        full = fetch_from_agx(user_msg)
    else:
        prompt = build_prompt(user_msg, system, messages)
        full = enqueue_infer(prompt)

    t_end = time.time()
    timing["inference_ms"] = (
        t_end - t_start - timing.get("classify_ms", 0) / 1000
    ) * 1000
    timing["total_ms"] = (t_end - t_start) * 1000
    timing["route"] = "AGX" if route_agx else "NX"
    timing["response_len"] = len(full)

    log.info(
        f"Timing: classify={timing.get('classify_ms', 0):.0f}ms "
        f"({result.get('method', '?')})  "
        f"c={conf:.3f}  "
        f"infer={timing['inference_ms']:.0f}ms  "
        f"total={timing['total_ms']:.0f}ms  "
        f"route={timing['route']}  "
        f"resp={timing['response_len']}chars"
    )

    log_to_agx({
        "query": user_msg,
        "response": full,
        "is_medical": is_med,
        "confidence": conf,
        "p_med": result.get("p_med"),
        "kl_div": result.get("kl_div"),
        "kmeans": result.get("kmeans"),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "routed_to": "AGX" if route_agx else "NX",
        "timing_ms": timing,
    })

    source = "AGX" if route_agx else "NX"

    if not stream:
        return jsonify({
            "content": full.strip(),
            "source": source,
            "timing_ms": timing,
            "confidence": conf,
            "p_med": result.get("p_med"),
            "kl_div": result.get("kl_div"),
            "kmeans_label": (result.get("kmeans") or {}).get("label"),
        })

    def generate():
        for i in range(0, len(full), 80):
            yield f"data: {json.dumps({'token': full[i:i+80], 'source': source})}\n\n"
        done = {
            "done": True, "source": source, "content": full.strip(),
            "timing_ms": timing, "confidence": conf,
            "p_med": result.get("p_med"), "kl_div": result.get("kl_div"),
            "kmeans_label": (result.get("kmeans") or {}).get("label"),
        }
        yield f"data: {json.dumps(done)}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@sock.route("/ws")
def ws_chat(ws):
    log.info("WebSocket connected")
    while True:
        raw = ws.receive()
        if raw is None:
            break
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            ws.send(json.dumps({"type": "error", "message": "Invalid JSON"}))
            continue
        msg_type = data.get("type", "")
        if msg_type == "ping":
            ws.send(json.dumps({"type": "pong"}))
            continue
        if msg_type != "message":
            continue

        user_msg = data.get("content", "")
        system = data.get("system", "")
        messages = data.get("messages")

        ws_start = time.time()
        timing: dict = {}
        result = enqueue_classify(user_msg, timing=timing)
        is_med = result["is_medical"]
        conf = result["confidence"]
        route_agx = not (is_med and conf >= CONFIDENCE_THRESHOLD)

        log.info(
            f"Query: {user_msg[:60]!r}  "
            f"is_med={is_med}  c={conf:.3f}  "
            f"kl={result.get('kl_div', 0):.2f}  "
        f"kmeans={(result.get('kmeans') or {}).get('label', '?')}  "
            f"route={'AGX' if route_agx else 'NX'}"
        )

        if route_agx:
            full = fetch_from_agx(user_msg)
        else:
            prompt = build_prompt(user_msg, system, messages)
            full = enqueue_infer(prompt)

        ws_end = time.time()
        timing["inference_ms"] = (
            ws_end - ws_start - timing.get("classify_ms", 0) / 1000
        ) * 1000
        timing["total_ms"] = (ws_end - ws_start) * 1000
        timing["route"] = "AGX" if route_agx else "NX"
        timing["response_len"] = len(full)

        log.info(
            f"Timing: classify={timing.get('classify_ms', 0):.0f}ms "
            f"({result.get('method', '?')})  "
            f"c={conf:.3f}  "
            f"infer={timing['inference_ms']:.0f}ms  "
            f"total={timing['total_ms']:.0f}ms  "
            f"route={timing['route']}  "
            f"resp={timing['response_len']}chars"
        )

        log_to_agx({
            "query": user_msg,
            "response": full,
            "is_medical": is_med,
            "confidence": conf,
            "p_med": result.get("p_med"),
            "kl_div": result.get("kl_div"),
            "kmeans": result.get("kmeans"),
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "routed_to": "AGX" if route_agx else "NX",
            "timing_ms": timing,
        })

        source = "AGX" if route_agx else "NX"

        for i in range(0, len(full), 80):
            ws.send(json.dumps({"type": "chunk", "content": full[i:i+80], "source": source}))
        ws.send(json.dumps({
            "type": "done",
            "content": full.strip(),
            "source": source,
            "timing_ms": timing,
            "confidence": conf,
            "p_med": result.get("p_med"),
            "kl_div": result.get("kl_div"),
            "kmeans_label": (result.get("kmeans") or {}).get("label"),
        }))


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid NX/AGX subserver")
    parser.add_argument("--agx-ip", required=True, help="AGX Tailscale IP")
    parser.add_argument("--agx-port", type=int, default=8000)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--node-name", default="nx-subserver")
    parser.add_argument("--confidence", type=float, default=0.7,
                        help="Min confidence to route to NX (0-1, default=0.7)")
    args = parser.parse_args()

    agx_host = args.agx_ip
    agx_port = args.agx_port
    node_name = args.node_name
    CONFIDENCE_THRESHOLD = args.confidence

    _ensure_worker()

    print(f"\n  Subserver:  http://{args.host}:{args.port}")
    print(f"  AGX:        http://{agx_host}:{agx_port}")
    print(f"  Node:       {node_name}")
    print(f"  WS:         ws://{args.host}:{args.port}/ws")
    print(f"  Chat API:   http://{args.host}:{args.port}/chat (POST)")
    print(f"  Confidence: >= {CONFIDENCE_THRESHOLD} → NX,  < {CONFIDENCE_THRESHOLD} → AGX")
    print(f"  Model q:    {{classify, infer}} (serialised)")
    print(f"  Classifier: multi-sample ({N_SAMPLES}x) + KL divergence + online k-means")
    print()

    app.run(host=args.host, port=args.port, threaded=True)
