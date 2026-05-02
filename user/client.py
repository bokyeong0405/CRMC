"""CRMC user client.

Sends preprocessed TinyImageNet validation images to the orchestrator's
``/infer`` endpoint one at a time and prints per-request results plus
aggregate accuracy / latency stats.

The validation tensors are loaded from local ``.npy`` files (the upstream
TinyImageNet preprocessing pipeline already produced ``(64, 64, 3) float32``
tensors normalised to ``[0, 1]``), so the client doesn't need TF, image
libraries, or network access — keeping the user container small.
"""

import argparse
import base64
import os
import statistics
import sys
import time

import httpx
import numpy as np


DEFAULT_X_PATH = os.environ.get("X_VAL_PATH", "/data/X_val_s.npy")
DEFAULT_Y_PATH = os.environ.get("Y_VAL_PATH", "/data/y_val_encoded.npy")


def load_validation(x_path: str, y_path: str):
    """Memory-map ``X`` (it's ~1 GB) so we don't pay the full RAM cost when
    the caller only needs a handful of samples."""
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"X tensor not found: {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"y labels not found: {y_path}")
    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"X/y length mismatch: {x.shape[0]} vs {y.shape[0]}")
    return x, y


def send_one(client, orchestrator_url, x_image: np.ndarray):
    # x_image is a single (H, W, C) sample; add a batch dim and copy out of
    # the mmap into a contiguous buffer so .tobytes() is well-defined.
    x = np.ascontiguousarray(x_image[np.newaxis, ...], dtype=np.float32)

    payload = {
        "tensor_b64": base64.b64encode(x.tobytes()).decode("ascii"),
        "tensor_dtype": "float32",
        "tensor_shape": list(x.shape),
    }

    t0 = time.perf_counter()
    r = client.post(f"{orchestrator_url}/infer", json=payload)
    r.raise_for_status()
    wall_ms = (time.perf_counter() - t0) * 1000
    return r.json(), wall_ms


def main():
    parser = argparse.ArgumentParser(description="CRMC user client (TinyImageNet)")
    parser.add_argument(
        "--orchestrator-url",
        default=os.environ.get("ORCHESTRATOR_URL", "http://orchestrator:8080"),
        help="orchestrator base URL (default: env ORCHESTRATOR_URL or http://orchestrator:8080)",
    )
    parser.add_argument(
        "--num-requests", "-n", type=int,
        default=int(os.environ.get("NUM_REQUESTS", "10")),
        help="how many validation images to send (default: env NUM_REQUESTS or 10)",
    )
    parser.add_argument(
        "--x-path", default=DEFAULT_X_PATH,
        help=f"path to X_val tensor (default: env X_VAL_PATH or {DEFAULT_X_PATH})",
    )
    parser.add_argument(
        "--y-path", default=DEFAULT_Y_PATH,
        help=f"path to y_val labels (default: env Y_VAL_PATH or {DEFAULT_Y_PATH})",
    )
    parser.add_argument(
        "--shuffle", action="store_true",
        default=os.environ.get("SHUFFLE", "").lower() in ("1", "true", "yes"),
        help="pick validation images at random (default: in order)",
    )
    parser.add_argument(
        "--seed", type=int,
        default=int(os.environ.get("SEED", "0")),
        help="random seed when --shuffle is set",
    )
    parser.add_argument(
        "--timeout", type=float, default=300.0,
        help="HTTP timeout in seconds (default: 300)",
    )
    args = parser.parse_args()

    print(f"loading validation tensors from {args.x_path} / {args.y_path} ...", file=sys.stderr)
    x_val, y_val = load_validation(args.x_path, args.y_path)
    print(f"  X: {x_val.shape} {x_val.dtype}   y: {y_val.shape} ({len(np.unique(y_val))} classes)",
          file=sys.stderr)

    n = min(args.num_requests, len(x_val))
    if args.shuffle:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(x_val), size=n, replace=False)
    else:
        indices = np.arange(n)

    print(f"sending {n} requests to {args.orchestrator_url}/infer", file=sys.stderr)

    correct = 0
    orchestrator_latencies = []
    wall_latencies = []
    plan_seen = None

    with httpx.Client(timeout=args.timeout) as client:
        for i, idx in enumerate(indices):
            true_label = int(y_val[idx])
            try:
                res, wall_ms = send_one(client, args.orchestrator_url, x_val[idx])
            except httpx.HTTPError as e:
                print(f"[{i + 1}/{n}] request failed: {e}", file=sys.stderr)
                continue

            pred = res["predicted_class"]
            is_correct = pred == true_label
            correct += int(is_correct)
            orchestrator_latencies.append(res["end_to_end_latency_ms"])
            wall_latencies.append(wall_ms)

            # The plan is the same across requests as long as the orchestrator
            # capacity cache doesn't refresh — print it once and then condense.
            plan = res["plan"]
            if plan_seen is None:
                plan_seen = plan
                print(f"plan: splitting_points={plan['splitting_points']}")
                for h in plan["hops"]:
                    print(f"  → {h['device_id']}  slice={h['slice_index']}  "
                          f"layers {h['start_layer']}..{h['end_layer']}")
                print(f"  estimated_latency={plan['estimated_latency']:.3f}  "
                      f"estimated_accuracy={plan['estimated_accuracy']}")
                print()

            mark = "OK" if is_correct else "  "
            print(
                f"[{i + 1:>3}/{n}] true={true_label:>3} pred={pred:>3} {mark}  "
                f"orch={res['end_to_end_latency_ms']:>7.1f}ms  "
                f"wall={wall_ms:>7.1f}ms  "
                f"hops={res.get('compute_ms_per_hop', '')}"
            )

    if not orchestrator_latencies:
        print("no successful requests", file=sys.stderr)
        sys.exit(1)

    print()
    print("=" * 60)
    print(f"accuracy:  {correct}/{n} = {correct / n:.4f}")
    print(
        f"orchestrator latency: mean={statistics.mean(orchestrator_latencies):.1f}ms  "
        f"median={statistics.median(orchestrator_latencies):.1f}ms  "
        f"max={max(orchestrator_latencies):.1f}ms"
    )
    print(
        f"wall latency:         mean={statistics.mean(wall_latencies):.1f}ms  "
        f"median={statistics.median(wall_latencies):.1f}ms  "
        f"max={max(wall_latencies):.1f}ms"
    )


if __name__ == "__main__":
    main()
