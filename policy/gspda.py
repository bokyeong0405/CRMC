"""GSPDA — Graph-based Shortest Path Distributed Algorithm.

Ported from ``resnet34_TinyImageNet/GSPDA_resnet.py``. The algorithm is kept
faithful to the original; only the CIFAR-10 branch (``l=0``) is dropped,
since this project targets ResNet34 only.
"""

import logging

import networkx as nx
import numpy as np

from .accuracy import predict_accuracy_res
from .inference_time import inference_time
from .set_ss_sb import set_ss_sb

logger = logging.getLogger(__name__)


def _passes_accuracy(accuracy, acc_thresh):
    # ``predict_accuracy_res`` returns ``None`` when calibration data is
    # missing — treat as "skip filter" so the system stays usable.
    if accuracy is None:
        return True
    return accuracy > acc_thresh


def _set_inf(A, P):
    """Block the just-considered edge so the next ``shortest_path`` call
    yields a different one (mirrors original semantics)."""
    if len(P) == 1:
        A[P[0], P[0]] = np.inf
    elif len(P) == 2:
        A[P[0], P[1]] = np.inf
    elif len(P) == 3:
        A[P[0], P[1]] = np.inf
        A[P[1], P[2]] = np.inf
    return A


def _candidate_paths(start_point, end_point0, G, A, SS):
    """Yield (path, length) for every reachable (start, end) pair in the
    graph, blocking each as we go so a subsequent shortest_path call returns
    a *different* candidate. Mirrors the original double-loop in
    find_infer_short / GSPDA."""
    for i in start_point:
        if i % SS == 6:
            P = nx.shortest_path(G, source=i, target=i, weight="weight")
            d = G[i][i]["weight"] if G.has_edge(i, i) else np.inf
            if len(P) > 3:
                A = _set_inf(A, P)
                G = nx.DiGraph(A)
                continue
            yield P, d, A, G
        elif i // SS == 0:
            for j in end_point0[i % SS]:
                P = nx.shortest_path(G, source=i, target=j, weight="weight")
                d = nx.shortest_path_length(G, source=i, target=j, weight="weight")
                if len(P) > 3:
                    A = _set_inf(A, P)
                    G = nx.DiGraph(A)
                    continue
                yield P, d, A, G


def gspda(
    start_point,
    end_point0,
    G,
    SS_f,
    D_C,
    end,
    D_BER,
    acc_thresh,
    A,
    start,
    SS_d,
    D_tt,
    SS,
    energy_thresh,
):
    """Run GSPDA and return the chosen path plus its metrics.

    Returns
    -------
    optimal_path : list[int]
    inference_latency : float
    energy_consumption : list[float]
    accuracy : float | None
    """
    fallback_path = None
    fallback_acc = None

    candidates = []  # paths that pass the accuracy filter
    accuracies = []

    for P, d, A, G in _candidate_paths(start_point, end_point0, G, A, SS):
        sp, ber = set_ss_sb(P, end, D_BER, SS)
        accuracy = predict_accuracy_res(sp, ber)

        # Track the very first reachable path as a fallback (matches the
        # original ``find_infer_short`` behaviour: shortest path regardless of
        # accuracy).
        if fallback_path is None:
            fallback_path = P
            fallback_acc = accuracy

        if _passes_accuracy(accuracy, acc_thresh):
            candidates.append(P)
            accuracies.append(accuracy)

        A = _set_inf(A, P)
        G = nx.DiGraph(A)

    if not candidates:
        if fallback_path is None:
            raise RuntimeError("GSPDA: no reachable path in the graph")
        candidates.append(fallback_path)
        accuracies.append(fallback_acc)

    inferences = []
    com_consumptions = []
    for P in candidates:
        inf, com, _trans = inference_time(P, SS_f, D_C, D_BER, start, end, SS_d, D_tt, SS)
        inferences.append(inf)
        com_consumptions.append(com)

    while True:
        idx = int(np.argmin(inferences))
        com = com_consumptions[idx]
        if all(c < energy_thresh for c in com):
            return candidates[idx], inferences[idx], com, accuracies[idx]

        del inferences[idx]
        del com_consumptions[idx]
        del candidates[idx]
        del accuracies[idx]

        if not candidates:
            # All candidates breached energy threshold — fall back to the
            # latency-min path computed earlier (same as the original).
            logger.warning("GSPDA: energy threshold exceeded by all candidates; using fallback path")
            inf, com, _ = inference_time(fallback_path, SS_f, D_C, D_BER, start, end, SS_d, D_tt, SS)
            return fallback_path, inf, com, fallback_acc
