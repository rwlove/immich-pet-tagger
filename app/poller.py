"""Poller: incremental classification using a local CLIP model.
No DB access. Embeddings computed from thumbnails via the Immich HTTP API."""

import logging
import os
import pickle
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import io
import numpy as np
import open_clip
import requests
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import data
import immich as imm

log = logging.getLogger("poller")

THRESHOLD = float(os.environ.get("THRESHOLD", 0.8))
DRY_RUN = os.environ.get("DRY_RUN", "").lower() in ("1", "true", "yes")
GPU_WORKERS = int(os.environ.get("GPU_WORKERS", 2))
# On CPU, inference is slow enough that 8 fetchers saturate the queue.
# On GPU, each worker spends ~65% of its time blocked in the GPU queue, so we need
# enough workers to keep batch sizes high: GPU_WORKERS * 32 fills ~16-image batches.
_default_scan_workers = GPU_WORKERS * 32 if torch.cuda.is_available() else 8
SCAN_WORKERS = int(os.environ.get("SCAN_WORKERS", _default_scan_workers))
CLIP_BATCH_SIZE = int(os.environ.get("CLIP_BATCH_SIZE", 32))

_count_lock = threading.Lock()

CLIP_MODEL_NAME = os.environ.get("CLIP_MODEL", "ViT-B-16")
CLIP_PRETRAINED = os.environ.get("CLIP_PRETRAINED", "openai")

_embed_cache: dict[str, np.ndarray] = {}
_cache_path: Path | None = None

# ---------------------------------------------------------------------------
# CLIP batch workers
# ---------------------------------------------------------------------------

# Preprocess transform shared across worker threads (set by first CLIP worker).
# Worker threads do CPU preprocessing; batch threads only stack + run GPU.
_clip_preprocess_fn = None
_clip_preprocess_ready = threading.Event()


class _EmbedReq:
    __slots__ = ("tensor", "event", "result")
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor
        self.event = threading.Event()
        self.result: np.ndarray | None = None

_embed_queue: queue.Queue[_EmbedReq] = queue.Queue()
_clip_worker_threads: list[threading.Thread] = []
_clip_worker_lock = threading.Lock()

# batch stats (reset each poll cycle)
_clip_batch_total = 0
_clip_batch_count = 0
_stats_lock = threading.Lock()


def _clip_batch_loop(worker_id: int) -> None:
    global _clip_batch_total, _clip_batch_count, _clip_preprocess_fn
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"CLIP worker {worker_id} loading on {device}...")
    model, preprocess, _ = open_clip.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED)
    model.eval().to(device)
    if not _clip_preprocess_ready.is_set():
        _clip_preprocess_fn = preprocess
        _clip_preprocess_ready.set()
    stream = torch.cuda.Stream() if device == "cuda" else None
    log.info(f"CLIP worker {worker_id} ready")

    while True:
        first = _embed_queue.get()
        batch = [first]
        try:
            while len(batch) < CLIP_BATCH_SIZE:
                batch.append(_embed_queue.get_nowait())
        except queue.Empty:
            pass

        with _stats_lock:
            _clip_batch_total += len(batch)
            _clip_batch_count += 1

        try:
            # tensors are already preprocessed by worker threads
            stacked = torch.stack([req.tensor for req in batch])
            if stream is not None:
                with torch.cuda.stream(stream):
                    tensors = stacked.to(device, non_blocking=True)
                    with torch.no_grad():
                        feats = model.encode_image(tensors)
                        feats = feats / feats.norm(dim=-1, keepdim=True)
                stream.synchronize()
                vecs = feats.cpu().numpy()
            else:
                with torch.no_grad():
                    feats = model.encode_image(stacked.to(device))
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                vecs = feats.cpu().numpy()
        except Exception as e:
            log.warning(f"CLIP worker {worker_id} batch error: {e}")
            vecs = [None] * len(batch)

        for req, vec in zip(batch, vecs):
            req.result = vec
            req.event.set()


def _ensure_clip_workers() -> None:
    with _clip_worker_lock:
        alive = [t for t in _clip_worker_threads if t.is_alive()]
        for i in range(len(alive), GPU_WORKERS):
            t = threading.Thread(target=_clip_batch_loop, args=(i,), daemon=True, name=f"clip-batch-{i}")
            t.start()
            _clip_worker_threads.append(t)


# ---------------------------------------------------------------------------
# Date range helpers
# ---------------------------------------------------------------------------

from datetime import date


def parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except (ValueError, TypeError):
        return None


def asset_in_range(time_str: str, since: str | None, until: str | None) -> bool:
    d = parse_date(time_str)
    if d is None:
        return True
    if since and d < date.fromisoformat(since):
        return False
    if until and d > date.fromisoformat(until):
        return False
    return True


# ---------------------------------------------------------------------------
# Thumbnail fetch and CLIP embedding
# ---------------------------------------------------------------------------

def fetch_thumbnail(asset_id: str) -> Image.Image | None:
    try:
        r = requests.get(
            f"{imm.IMMICH_URL}/api/assets/{asset_id}/thumbnail?size=preview",
            headers={"x-api-key": imm.IMMICH_API_KEY},
            timeout=15,
        )
        if r.status_code == 200 and r.content:
            return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        log.warning(f"fetch_thumbnail {asset_id}: {e}")
    return None


def embed_image(img: Image.Image) -> np.ndarray | None:
    _ensure_clip_workers()
    _clip_preprocess_ready.wait()  # blocks only until first CLIP worker is up
    tensor = _clip_preprocess_fn(img)  # CPU preprocessing in caller's thread
    req = _EmbedReq(tensor)
    _embed_queue.put(req)
    req.event.wait()
    return req.result


def crop_animals(img: Image.Image) -> list[tuple[tuple, Image.Image]]:
    """Detect animals and return (bbox_norm, crop) pairs. Empty list means no animals found."""
    try:
        from detector import detect_animals
        boxes = detect_animals(img)
    except Exception as e:
        log.warning(f"YOLO detection failed: {e}")
        return []
    w, h = img.size
    return [
        (bbox, img.crop((int(bbox[0] * w), int(bbox[1] * h), int(bbox[2] * w), int(bbox[3] * h))))
        for bbox in boxes
    ]


def load_embed_cache(data_dir: Path) -> None:
    global _cache_path
    _cache_path = data_dir / "embeddings.pkl"
    if _cache_path.exists():
        try:
            with open(_cache_path, "rb") as f:
                _embed_cache.update(pickle.load(f))
            log.info(f"Loaded {len(_embed_cache)} cached embeddings from {_cache_path}")
        except Exception as e:
            log.warning(f"Could not load embedding cache: {e}")


def _save_embed_cache() -> None:
    if _cache_path is None:
        return
    tmp = _cache_path.with_suffix(".tmp")
    try:
        with open(tmp, "wb") as f:
            pickle.dump(_embed_cache, f)
        tmp.replace(_cache_path)
    except Exception as e:
        log.warning(f"Could not save embedding cache: {e}")


def embed_asset(asset_id: str) -> np.ndarray | None:
    if asset_id in _embed_cache:
        return _embed_cache[asset_id]
    img = fetch_thumbnail(asset_id)
    if img is None:
        return None
    crops = crop_animals(img)
    vec = embed_image(crops[0][1]) if crops else None
    if vec is None:
        vec = embed_image(img)
    if vec is not None:
        _embed_cache[asset_id] = vec
        _save_embed_cache()
    return vec


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def build_classifier(
    pet_names: list[str],
    ref_ids_per_pet: dict[str, list[str]],
    negative_ids: list[str] | None = None,
) -> tuple[list[str], LogisticRegression, StandardScaler] | None:
    all_vecs = []
    all_labels = []
    unknown_idx = len(pet_names)
    names = pet_names + ["unknown"]

    for i, name in enumerate(pet_names):
        ids = ref_ids_per_pet.get(name, [])
        log.info(f"Embedding {len(ids)} refs for '{name}'...")
        for aid in ids:
            vec = embed_asset(aid)
            if vec is not None:
                all_vecs.append(vec)
                all_labels.append(i)
            else:
                log.warning(f"  Could not embed ref {aid} for '{name}'")

    # Balance negatives to ~3x total pet refs
    total_refs = sum(len(ids) for ids in ref_ids_per_pet.values())
    if negative_ids:
        target = total_refs * 3
        if len(negative_ids) > target:
            negative_ids = random.sample(negative_ids, target)
            log.info(f"Subsampled negatives to {target} (3x {total_refs} refs)")

        log.info(f"Embedding {len(negative_ids)} negative samples...")
        for aid in negative_ids:
            vec = embed_asset(aid)
            if vec is not None:
                all_vecs.append(vec)
                all_labels.append(unknown_idx)

    if not all_vecs:
        log.warning("No embeddings computed, skipping classifier training.")
        return None

    X = np.array(all_vecs, dtype=np.float64)
    y = np.array(all_labels, dtype=np.intp)

    if unknown_idx not in y:
        X = np.vstack([X, np.zeros((1, X.shape[1]))])
        y = np.append(y, unknown_idx)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, random_state=0)
    clf.fit(X_scaled, y)
    log.info(f"Classifier trained on {len(y)} samples, classes: {names} ({sum(y==unknown_idx)} unknown)")
    return names, clf, scaler


def classify(vec, names, clf, scaler) -> tuple[str, float]:
    v = np.asarray(vec, dtype=np.float64).reshape(1, -1)
    probs = clf.predict_proba(scaler.transform(v))[0]
    i = int(np.argmax(probs))
    return names[i], float(probs[i])


# ---------------------------------------------------------------------------
# Face assignment
# ---------------------------------------------------------------------------

def post_face(asset_id: str, person_id: str, bbox_norm=None, img_size=None) -> str | None:
    """Returns face_id on success, None on failure."""
    if bbox_norm is not None and img_size is not None:
        x1, y1, x2, y2 = bbox_norm
        iw, ih = img_size
        bx, by = int(x1 * iw), int(y1 * ih)
        bw, bh = int((x2 - x1) * iw), int((y2 - y1) * ih)
    else:
        bx, by, bw, bh = 0, 0, imm.FACE_BOX_SIZE, imm.FACE_BOX_SIZE
        iw, ih = imm.FACE_BOX_SIZE, imm.FACE_BOX_SIZE
    try:
        r = requests.post(
            f"{imm.IMMICH_URL}/api/faces",
            json={"assetId": asset_id, "personId": person_id,
                  "width": bw, "height": bh,
                  "imageWidth": iw, "imageHeight": ih,
                  "x": bx, "y": by},
            headers={**imm.headers(), "Content-Type": "application/json"},
            timeout=30,
        )
        if r.status_code not in (200, 201):
            log.warning(f"post_face {asset_id} -> {r.status_code}: {r.text[:200]}")
            return None
        fr = requests.get(f"{imm.IMMICH_URL}/api/faces", headers=imm.headers(), params={"id": asset_id}, timeout=15)
        if fr.status_code == 200:
            for face in fr.json():
                if (face.get("person") or {}).get("id") == person_id:
                    return face.get("id")
        log.warning(f"post_face: created but could not retrieve face_id for asset {asset_id}")
        return None
    except Exception as e:
        log.error(f"post_face error: {e}")
        return None


# ---------------------------------------------------------------------------
# Main poll cycle
# ---------------------------------------------------------------------------

def run_poll_cycle(data_dir: str, on_date=None, cancel=None, low_conf_out=None, live_counts: dict | None = None) -> None:
    log.info(f"Poll cycle | threshold={THRESHOLD} dry_run={DRY_RUN}")
    dd = Path(data_dir)
    now = datetime.now(timezone.utc).isoformat()
    data.write_poll_status(dd, {"status": "running", "started_at": now})

    counts = live_counts if live_counts is not None else {}
    for k in ("added", "low_confidence", "unknown", "out_of_range", "already_tagged", "failed", "no_thumb"):
        counts[k] = 0
    try:
        _run_poll_cycle(dd, counts, on_date, cancel, low_conf_out)
    except Exception as e:
        data.write_poll_status(dd, {"status": "error", "ran_at": datetime.now(timezone.utc).isoformat(), "error": str(e), "counts": counts})
        raise
    else:
        data.write_poll_status(dd, {"status": "idle", "ran_at": datetime.now(timezone.utc).isoformat(), "counts": counts})


def _run_poll_cycle(dd: Path, counts: dict, on_date=None, cancel=None, low_conf_out=None) -> None:
    config = data.load_config(dd)
    if not config:
        log.warning("config.json empty or missing, no pets configured yet.")
        return

    all_pet_names = list(config.keys())
    all_ref_ids = {name: data.load_pet_asset_ids(name, dd) for name in all_pet_names}

    pet_names = [n for n in all_pet_names if all_ref_ids.get(n)]
    ref_ids_per_pet = {n: all_ref_ids[n] for n in pet_names}
    skipped = [n for n in all_pet_names if n not in pet_names]

    if skipped:
        log.warning(f"Skipping pets with no refs: {skipped}")
    if not pet_names:
        log.warning("No pets with reference assets, enroll pets via the UI first.")
        return

    log.info(f"Pets: {', '.join(f'{n}({len(ref_ids_per_pet[n])} refs)' for n in pet_names)}")

    negative_ids = data.load_negative_ids(dd)
    if negative_ids:
        log.info(f"Loaded {len(negative_ids)} negative samples")

    result = build_classifier(pet_names, ref_ids_per_pet, negative_ids)
    if result is None:
        return
    names, clf, scaler = result

    last_ts = data.load_last_timestamp(dd)
    log.info(f"Fetching assets taken after: {last_ts}")

    t0 = time.time()
    assets = imm.fetch_assets_taken_after(last_ts)
    log.info(f"Fetched {len(assets)} assets in {time.time()-t0:.1f}s")

    if not assets:
        log.info("No new assets.")
        data.save_last_timestamp(datetime.now(timezone.utc).isoformat(), dd)
        return

    latest_ts = max((ts for _, ts in assets), default=last_ts)

    def process_asset(aid: str, time_str: str) -> None:
        if cancel and cancel.is_set():
            return

        if on_date:
            on_date(time_str[:10])

        img = fetch_thumbnail(aid)
        if img is None:
            with _count_lock:
                counts["no_thumb"] += 1
            return
        crops = crop_animals(img)
        if not crops:
            crops = [(None, img)]
        elif len(crops) > 1:
            log.info(f"YOLO detected {len(crops)} animals in {aid} ({time_str[:10]})")
        vecs = [(bbox_norm, embed_image(crop)) for bbox_norm, crop in crops]

        existing_persons: set | None = None
        tagged_in_photo: set[str] = set()

        for bbox_norm, vec in vecs:
            if vec is None:
                continue

            pet_name, prob = classify(vec, names, clf, scaler)

            if pet_name == "unknown":
                with _count_lock:
                    counts["unknown"] += 1
                continue

            if prob < THRESHOLD:
                with _count_lock:
                    counts["low_confidence"] += 1
                if low_conf_out is not None:
                    low_conf_out.append({"asset_id": aid, "pet_name": pet_name, "prob": prob, "date": time_str[:10]})
                continue

            cfg = config.get(pet_name, {})
            if not asset_in_range(time_str, cfg.get("since"), cfg.get("until")):
                with _count_lock:
                    counts["out_of_range"] += 1
                continue

            person_id = cfg.get("person_id")
            if not person_id:
                log.warning(f"Pet '{pet_name}' has no person_id in config.")
                continue

            if person_id in tagged_in_photo:
                with _count_lock:
                    counts["already_tagged"] += 1
                continue

            if existing_persons is None:
                existing_persons = imm.fetch_asset_face_person_ids(aid)

            if person_id in existing_persons:
                with _count_lock:
                    counts["already_tagged"] += 1
                continue

            log.info(f"{imm.IMMICH_URL}/search/photos/{aid} -> {pet_name} ({prob:.3f}) | {time_str[:10]}")

            if DRY_RUN:
                log.info(f"  dry-run: would add {pet_name}")
                with _count_lock:
                    counts["added"] += 1
                tagged_in_photo.add(person_id)
            else:
                face_id = post_face(aid, person_id, bbox_norm, img.size if bbox_norm is not None else None)
                tagged_in_photo.add(person_id)
                with _count_lock:
                    if face_id:
                        counts["added"] += 1
                    else:
                        counts["failed"] += 1

    import detector as _det
    with _stats_lock:
        global _clip_batch_total, _clip_batch_count
        _clip_batch_total = _clip_batch_count = 0
    with _det._yolo_stats_lock:
        _det.yolo_batch_total = _det.yolo_batch_count = 0

    log.info(f"Processing {len(assets)} assets with {SCAN_WORKERS} workers")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as executor:
        futures = {executor.submit(process_asset, aid, ts): aid for aid, ts in assets}
        for future in as_completed(futures):
            if cancel and cancel.is_set():
                executor.shutdown(wait=False, cancel_futures=True)
                log.info("Scan cancelled.")
                return
            try:
                future.result()
            except Exception as e:
                log.warning(f"Asset {futures[future]} failed: {e}")

    elapsed = time.time() - t0
    with _stats_lock:
        clip_avg = _clip_batch_total / _clip_batch_count if _clip_batch_count else 0
    with _det._yolo_stats_lock:
        yolo_avg = _det.yolo_batch_total / _det.yolo_batch_count if _det.yolo_batch_count else 0
    log.info(
        f"STATS | assets={len(assets)} elapsed={elapsed:.1f}s "
        f"throughput={len(assets)/elapsed:.1f}/s "
        f"yolo_batch={yolo_avg:.1f} clip_batch={clip_avg:.1f} "
        f"counts={counts}"
    )

    if not DRY_RUN:
        data.save_last_timestamp(latest_ts, dd)
        log.info(f"Saved timestamp: {latest_ts}")
