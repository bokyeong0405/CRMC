"""CRMC orchestrator.

User → ``POST /infer`` with a preprocessed tensor (base64-encoded raw bytes).
The orchestrator queries each device's compute capacity, runs GSPDA against
``(D_C, D_tt, D_BER)`` to pick a split plan, then dispatches the tensor to
the first server in the chain. The chain returns the final logits
synchronously, which the orchestrator returns to the user along with the
plan summary and latency breakdown.
"""

import asyncio
import base64
import json
import logging
import os
import time
from dataclasses import asdict
from typing import Optional

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request

from policy import compute_split_plan

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("orchestrator")


CONFIG_PATH = os.environ.get("CONFIG_PATH", "/app/config.json")
with open(CONFIG_PATH) as f:
    config = json.load(f)

SERVERS: list[str] = config["servers"]
D_TT: list[float] = config["D_tt"]
D_BER: list[float] = config["D_BER"]
ACC_THRESH: float = config.get("acc_thresh", 0.49)
ENERGY_THRESH: float = config.get("energy_thresh", 500)
SERVER_PORT: int = config.get("server_port", 8000)

if len(SERVERS) != 3:
    raise ValueError(f"need exactly 3 servers, got {SERVERS!r}")
if len(D_TT) != 3 or len(D_BER) != 3:
    raise ValueError("D_tt and D_BER must each be length 3")


app = FastAPI(title="CRMC orchestrator")

# Capacity is queried lazily on the first /infer request — startup ordering
# in compose is best-effort, so the device containers may not be listening
# yet when the orchestrator boots.
_capacity_cache: Optional[list[float]] = None
_capacity_lock = asyncio.Lock()


async def _query_capacity(client: httpx.AsyncClient, server_id: str) -> float:
    url = f"http://{server_id}:{SERVER_PORT}/capacity"
    r = await client.get(url, timeout=10.0)
    r.raise_for_status()
    return float(r.json()["compute_power"])


async def _get_capacities() -> list[float]:
    global _capacity_cache
    if _capacity_cache is not None:
        return _capacity_cache
    async with _capacity_lock:
        if _capacity_cache is not None:
            return _capacity_cache
        async with httpx.AsyncClient() as client:
            caps = await asyncio.gather(*(_query_capacity(client, sid) for sid in SERVERS))
        _capacity_cache = list(caps)
        logger.info("queried server capacities: %s", _capacity_cache)
        return _capacity_cache


@app.get("/healthz")
def healthz():
    return {"ok": True, "servers": SERVERS, "D_tt": D_TT, "D_BER": D_BER}


@app.post("/refresh-capacity")
async def refresh_capacity():
    """Drop the cached capacities so the next /infer re-queries each server.
    Useful when sweeping COMPUTE_POWER values across experiments without
    restarting the orchestrator."""
    global _capacity_cache
    _capacity_cache = None
    caps = await _get_capacities()
    return {"servers": SERVERS, "compute_power": caps}


@app.post("/infer")
async def infer(request: Request):
    body = await request.json()

    tensor_b64 = body["tensor_b64"]
    tensor_dtype = body["tensor_dtype"]
    tensor_shape = list(body["tensor_shape"])

    raw = base64.b64decode(tensor_b64)
    expected = int(np.prod(tensor_shape)) * np.dtype(tensor_dtype).itemsize
    if len(raw) != expected:
        raise HTTPException(
            status_code=400,
            detail=f"tensor bytes={len(raw)} but shape×dtype expects {expected}",
        )

    D_C = await _get_capacities()
    plan = compute_split_plan(
        D_C=D_C,
        D_tt=D_TT,
        D_BER=D_BER,
        acc_thresh=ACC_THRESH,
        energy_thresh=ENERGY_THRESH,
    )

    if not plan.hops:
        raise HTTPException(status_code=500, detail="GSPDA returned a plan with zero compute hops")

    logger.info(
        "plan: splitting_points=%s hops=%s lat=%.3f acc=%s",
        plan.splitting_points,
        [(h.device_id, h.slice_index, h.start_layer, h.end_layer) for h in plan.hops],
        plan.estimated_latency,
        plan.estimated_accuracy,
    )

    metadata = {
        "splitting_points": plan.splitting_points,
        "hops": [{"device_id": h.device_id, "slice_index": h.slice_index} for h in plan.hops],
        "current_hop_index": 0,
        "tensor_dtype": tensor_dtype,
        "tensor_shape": tensor_shape,
    }

    first_url = f"http://{plan.hops[0].device_id}:{SERVER_PORT}/run"
    files = {"tensor": ("tensor.bin", raw, "application/octet-stream")}
    data = {"metadata": json.dumps(metadata)}

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        r = await client.post(first_url, files=files, data=data)
        r.raise_for_status()
    end_to_end_ms = (time.perf_counter() - t0) * 1000

    out_dtype = r.headers.get("x-tensor-dtype", "float32")
    out_shape = json.loads(r.headers.get("x-tensor-shape", "[]"))
    out_arr = np.frombuffer(r.content, dtype=np.dtype(out_dtype)).reshape(out_shape)

    # Squeeze the batch dimension when it's 1 — gives the user a clean
    # vector of logits to argmax over.
    logits = out_arr[0] if out_arr.ndim == 2 and out_arr.shape[0] == 1 else out_arr

    return {
        "predicted_class": int(np.argmax(logits)),
        "logits": logits.tolist(),
        "end_to_end_latency_ms": round(end_to_end_ms, 3),
        "compute_ms_per_hop": r.headers.get("x-compute-ms", ""),
        "plan": {
            "splitting_points": plan.splitting_points,
            "hops": [asdict(h) for h in plan.hops],
            "estimated_latency": plan.estimated_latency,
            "estimated_accuracy": plan.estimated_accuracy,
            "raw_path": plan.raw_path,
        },
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
