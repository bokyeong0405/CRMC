"""Per-device inference server.

One container per simulation device. The orchestrator dispatches a request
to the first server in the GSPDA-decided chain; each server runs only its
assigned submodel slice, then forwards the intermediate tensor to the next
server in the chain. The last server's output (logits) bubbles back up the
synchronous response chain to the orchestrator.

Wire format on /run is multipart:
  - ``metadata`` (form field, JSON): plan + this hop's index + tensor dtype/shape
  - ``tensor`` (file field, raw bytes): numpy buffer

The same format is used to forward to the next server, so the payload is
self-describing all the way through the chain.
"""

import json
import logging
import os
import threading
import time
from typing import Optional

import httpx
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from model.resnet_split import build_submodels, load_full_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("device")

DEVICE_ID = os.environ["DEVICE_ID"]                                  # required, e.g. "server-1"
COMPUTE_POWER = float(os.environ.get("COMPUTE_POWER", "1.0"))         # simulation D_C value
WEIGHT_PATH = os.environ.get("WEIGHT_PATH") or None
NUM_CLASSES = int(os.environ.get("NUM_CLASSES", "200"))                # TinyImageNet
INPUT_SHAPE = tuple(int(x) for x in os.environ.get("INPUT_SHAPE", "64,64,3").split(","))
PORT = int(os.environ.get("PORT", "8000"))

app = FastAPI(title=f"CRMC device: {DEVICE_ID}")


_full_model = None
_full_model_lock = threading.Lock()
_submodel_cache: dict[tuple, list] = {}
_submodel_cache_lock = threading.Lock()


def _get_full_model():
    """Load the full ResNet34 once on first use. TF model loading is heavy
    (~hundreds of MB resident), so we keep a single instance and let
    ``split_resnet`` slice it on demand."""
    global _full_model
    if _full_model is None:
        with _full_model_lock:
            if _full_model is None:
                if WEIGHT_PATH and os.path.exists(WEIGHT_PATH):
                    logger.info("loading weights from %s", WEIGHT_PATH)
                else:
                    logger.warning(
                        "WEIGHT_PATH=%r missing; falling back to random init "
                        "(plumbing-only mode, accuracy will be meaningless)",
                        WEIGHT_PATH,
                    )
                _full_model = load_full_model(
                    weight_path=WEIGHT_PATH,
                    num_classes=NUM_CLASSES,
                    input_shape=INPUT_SHAPE,
                )
    return _full_model


def _get_submodels(splitting_points: list[int]) -> list:
    key = tuple(splitting_points)
    if key in _submodel_cache:
        return _submodel_cache[key]
    with _submodel_cache_lock:
        if key in _submodel_cache:
            return _submodel_cache[key]
        full = _get_full_model()
        # build_submodels reloads weights from disk — we already have them,
        # so call split_resnet + assign directly.
        from model.resnet_split import assign_weights_to_submodels, split_resnet
        submodels = split_resnet(full, list(splitting_points), input_shape=INPUT_SHAPE)
        assign_weights_to_submodels(full, submodels)
        _submodel_cache[key] = submodels
        logger.info("built submodels for splitting_points=%s (%d slices)", key, len(submodels))
        return submodels


@app.get("/capacity")
def capacity():
    """Report this server's compute capacity for the orchestrator's GSPDA
    input. We expose only the simulation-level number — actual cgroup
    limits are configured by docker-compose, not measured at runtime."""
    return {
        "device_id": DEVICE_ID,
        "compute_power": COMPUTE_POWER,
    }


@app.get("/healthz")
def healthz():
    return {"ok": True, "device_id": DEVICE_ID}


def _forward_to_next(next_device_id: str, metadata: dict, tensor_bytes: bytes) -> bytes:
    """Sync POST to the next server's /run with the same multipart format.
    Returns the raw response body (which is the tensor that came back from
    the downstream chain)."""
    url = f"http://{next_device_id}:{PORT}/run"
    files = {"tensor": ("tensor.bin", tensor_bytes, "application/octet-stream")}
    data = {"metadata": json.dumps(metadata)}
    # Generous timeout — model load on a cold downstream container can take
    # tens of seconds.
    with httpx.Client(timeout=httpx.Timeout(120.0)) as client:
        r = client.post(url, files=files, data=data)
        r.raise_for_status()
        return r.content, dict(r.headers)


@app.post("/run")
async def run(metadata: str = Form(...), tensor: UploadFile = File(...)):
    md = json.loads(metadata)

    splitting_points: list[int] = md["splitting_points"]
    hops: list[dict] = md["hops"]
    current: int = md["current_hop_index"]
    dtype = md["tensor_dtype"]
    shape = tuple(md["tensor_shape"])

    if current >= len(hops):
        raise HTTPException(status_code=400, detail="current_hop_index out of range")
    my_hop = hops[current]
    if my_hop["device_id"] != DEVICE_ID:
        raise HTTPException(
            status_code=400,
            detail=f"hop targets {my_hop['device_id']!r} but I am {DEVICE_ID!r}",
        )

    raw = await tensor.read()
    arr = np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape).copy()
    x = tf.convert_to_tensor(arr)

    submodels = _get_submodels(splitting_points)
    slice_index = my_hop["slice_index"]
    if slice_index >= len(submodels):
        raise HTTPException(
            status_code=400,
            detail=f"slice_index {slice_index} >= submodel count {len(submodels)}",
        )

    t0 = time.perf_counter()
    out = submodels[slice_index](x, training=False)
    compute_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "hop=%d device=%s slice=%d in=%s out=%s compute=%.2fms",
        current, DEVICE_ID, slice_index, list(shape), out.shape.as_list(), compute_ms,
    )

    out_np = out.numpy()
    out_bytes = out_np.tobytes()
    out_dtype = str(out_np.dtype)
    out_shape = list(out_np.shape)

    is_last = current == len(hops) - 1
    if is_last:
        # End of chain — return the logits as raw bytes with shape/dtype in
        # response headers (orchestrator deserialises).
        return Response(
            content=out_bytes,
            media_type="application/octet-stream",
            headers={
                "x-tensor-dtype": out_dtype,
                "x-tensor-shape": json.dumps(out_shape),
                "x-compute-ms": f"{compute_ms:.3f}",
            },
        )

    next_md = {
        **md,
        "current_hop_index": current + 1,
        "tensor_dtype": out_dtype,
        "tensor_shape": out_shape,
    }
    next_device = hops[current + 1]["device_id"]

    t1 = time.perf_counter()
    downstream_bytes, downstream_headers = _forward_to_next(next_device, next_md, out_bytes)
    forward_ms = (time.perf_counter() - t1) * 1000
    logger.info("hop=%d forwarded to %s; downstream took %.2fms", current, next_device, forward_ms)

    # Bubble the downstream response straight back. Preserve the dtype/shape
    # headers; aggregate compute timings as a comma-separated trail so the
    # orchestrator can attribute time per hop.
    headers = {
        "x-tensor-dtype": downstream_headers.get("x-tensor-dtype", out_dtype),
        "x-tensor-shape": downstream_headers.get("x-tensor-shape", json.dumps(out_shape)),
        "x-compute-ms": f"{compute_ms:.3f},{downstream_headers.get('x-compute-ms', '')}",
        "x-forward-ms": f"{forward_ms:.3f}",
    }
    return Response(content=downstream_bytes, media_type="application/octet-stream", headers=headers)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
