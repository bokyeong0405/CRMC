"""Public policy API: ``compute_split_plan`` returns a ``SplitPlan`` that the
orchestrator can ship straight to the device containers.

Path encoding from GSPDA: each entry is a node index in [0, N_V). Decoded as
``device = idx // SS`` and ``scenario = idx % SS``; the scenario indexes into
the ``start`` / ``end`` arrays for the layer range. Scenarios with
``end == 0`` are "forward-only" — the source device just relays input.
"""

from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

from .constants import (
    END_TO_SPLIT_POINT,
    N_V,
    SS,
    SS_d,
    SS_f,
    acc_thresh as default_acc_thresh,
    end,
    end_point0,
    energy_thresh as default_energy_thresh,
    start,
    start_point,
)
from .graph import create_graph
from .gspda import gspda


@dataclass
class Hop:
    """One compute step in the plan, executed by exactly one server."""
    device_id: str          # "server-1" / "server-2" / "server-3"
    slice_index: int        # which submodel index from split_resnet's output
    start_layer: int        # informational (original 1..34 numbering)
    end_layer: int


@dataclass
class SplitPlan:
    """Orchestrator → servers contract.

    Every server receives ``splitting_points`` so it can call
    ``model.resnet_split.split_resnet(model, splitting_points)`` and get a
    consistent submodel list, then run only its assigned ``slice_index``.
    """
    splitting_points: list[int]
    hops: list[Hop]
    estimated_latency: float
    estimated_accuracy: Optional[float]
    raw_path: list[int] = field(default_factory=list)


def _device_id(device_idx: int) -> str:
    return f"server-{device_idx + 1}"


def _path_to_plan(path, latency, accuracy) -> SplitPlan:
    # Compute hops (nodes that actually run layers).
    compute_nodes = [n for n in path if end[n % SS] != 0]

    # Splits sit *between* consecutive compute hops, at the end-layer of all
    # compute nodes except the last one.
    splitting_points = [END_TO_SPLIT_POINT[end[n % SS]] for n in compute_nodes[:-1]]
    # split_resnet expects only "real" split points (0..4); 7 is the
    # whole-model sentinel and should never appear here because a whole-model
    # node is the *last* compute hop.
    splitting_points = [sp for sp in splitting_points if sp != 7]

    hops = [
        Hop(
            device_id=_device_id(n // SS),
            slice_index=i,
            start_layer=start[n % SS],
            end_layer=end[n % SS],
        )
        for i, n in enumerate(compute_nodes)
    ]

    return SplitPlan(
        splitting_points=splitting_points,
        hops=hops,
        estimated_latency=latency,
        estimated_accuracy=accuracy,
        raw_path=list(path),
    )


def compute_split_plan(
    D_C: list[float],
    D_tt: list[float],
    D_BER: list[float],
    acc_thresh: float = default_acc_thresh,
    energy_thresh: float = default_energy_thresh,
) -> SplitPlan:
    """Run GSPDA against the current measurements / config and return the
    plan the orchestrator should dispatch.

    Parameters
    ----------
    D_C : 3-element list of compute capacities (FLOPS-equivalent units used
        by the simulation). Index ``i`` corresponds to ``server-{i+1}``.
    D_tt : 3-element list of inter-device link bandwidths. Index 0 is the
        server-1↔server-2 link, 1 is server-2↔server-3, 2 is server-1↔server-3.
    D_BER : 3-element list of per-link BER values, same ordering as ``D_tt``.
    """
    if len(D_C) != 3 or len(D_tt) != 3 or len(D_BER) != 3:
        raise ValueError("D_C, D_tt, D_BER must each be length 3")

    A = create_graph(N_V, SS_f, D_C, D_BER, SS_d, D_tt, start, end, SS)
    G = nx.DiGraph(A)

    optimal_path, latency, _consumption, accuracy = gspda(
        start_point=start_point,
        end_point0=end_point0,
        G=G,
        SS_f=SS_f,
        D_C=D_C,
        end=end,
        D_BER=D_BER,
        acc_thresh=acc_thresh,
        A=A,
        start=start,
        SS_d=SS_d,
        D_tt=D_tt,
        SS=SS,
        energy_thresh=energy_thresh,
    )

    return _path_to_plan(optimal_path, latency, accuracy)
