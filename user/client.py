"""CRMC user client.

Sends preprocessed CIFAR-10 test images to the orchestrator's /infer endpoint
one at a time and prints per-request results plus aggregate accuracy / latency
stats. The CIFAR-10 dataset is loaded from the standard upstream tarball
(cached under /tmp) — we deliberately avoid pulling TensorFlow in here so the
user container stays small.
"""

import argparse
import base64
import json
import os
import pickle
import statistics
import sys
import tarfile
import time
from urllib.request import urlretrieve

import httpx
import numpy as np


CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_CACHE_DIR = os.environ.get("CIFAR_CACHE_DIR", "/tmp/cifar10")

CIFAR10_LABELS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def load_cifar10_test():
    """Return ``(images, labels)`` for the CIFAR-10 test split.

    images: uint8 array of shape (10000, 32, 32, 3) in HWC order
    labels: int array of shape (10000,)
    """
    os.makedirs(CIFAR_CACHE_DIR, exist_ok=True)
    tar_path = os.path.join(CIFAR_CACHE_DIR, "cifar-10-python.tar.gz")
    extracted = os.path.join(CIFAR_CACHE_DIR, "cifar-10-batches-py")

    if not os.path.exists(extracted):
        if not os.path.exists(tar_path):
            print(f"downloading CIFAR-10 to {tar_path} ...", file=sys.stderr)
            urlretrieve(CIFAR_URL, tar_path)
        with tarfile.open(tar_path) as t:
            t.extractall(CIFAR_CACHE_DIR)

    with open(os.path.join(extracted, "test_batch"), "rb") as f:
        d = pickle.load(f, encoding="latin1")

    # The pickle stores images as (N, 3072) flat in CHW order.
    images = d["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)
    labels = np.asarray(d["labels"], dtype=np.int64)
    return images, labels


def preprocess(image_uint8):
    """Match the original Resnet_SC.py test pipeline: cast to float32 and
    normalize to [0, 1]. Add a batch dimension so the device server's
    ``submodel(x)`` call gets a 4-D tensor."""
    x = image_uint8.astype(np.float32) / 255.0
    return x[np.newaxis, ...]  # (1, 32, 32, 3)


def send_one(client, orchestrator_url, image_uint8):
    x = preprocess(image_uint8)
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
    parser = argparse.ArgumentParser(description="CRMC user client")
    parser.add_argument(
        "--orchestrator-url",
        default=os.environ.get("ORCHESTRATOR_URL", "http://orchestrator:8080"),
        help="orchestrator base URL (default: env ORCHESTRATOR_URL or http://orchestrator:8080)",
    )
    parser.add_argument(
        "--num-requests", "-n", type=int,
        default=int(os.environ.get("NUM_REQUESTS", "10")),
        help="how many test images to send (default: env NUM_REQUESTS or 10)",
    )
    parser.add_argument(
        "--shuffle", action="store_true",
        default=os.environ.get("SHUFFLE", "").lower() in ("1", "true", "yes"),
        help="pick test images at random (default: in order)",
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

    print(f"loading CIFAR-10 test split ...", file=sys.stderr)
    images, labels = load_cifar10_test()

    n = min(args.num_requests, len(images))
    if args.shuffle:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(images), size=n, replace=False)
    else:
        indices = np.arange(n)

    print(f"sending {n} requests to {args.orchestrator_url}/infer", file=sys.stderr)

    correct = 0
    orchestrator_latencies = []
    wall_latencies = []
    plan_seen = None

    with httpx.Client(timeout=args.timeout) as client:
        for i, idx in enumerate(indices):
            true_label = int(labels[idx])
            try:
                res, wall_ms = send_one(client, args.orchestrator_url, images[idx])
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
                f"[{i + 1:>3}/{n}] true={CIFAR10_LABELS[true_label]:<10} "
                f"pred={CIFAR10_LABELS[pred]:<10} {mark}  "
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
