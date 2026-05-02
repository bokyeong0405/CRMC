"""Shared constants for CRMC split-computing policies.

Lifted verbatim from ``resnet34_TinyImageNet/CRMC_eval_PDR.py`` so that the
graph builder, GSPDA, and inference-time estimator stay in lockstep with the
original simulation. Values must not drift from the simulation; if the model
or the split candidates change, update both.
"""

# Number of compute devices in the system.
D = 3

# Number of split-scenario nodes per device. Each scenario picks a (start, end)
# pair of layers — see the ``start`` / ``end`` arrays below.
SS = 22

# Total nodes in the graph: every (device, scenario) pair is a vertex.
N_V = D * SS  # 66

# Per-scenario FLOPs (MFLOPs) the assigned device must compute. Indexed by
# ``node_index % SS`` so it is the same for every device.
SS_f = [
    0, 4.82, 118.06, 260.64, 478.7, 583.54, 583.55,
    113.24, 255.82, 473.88, 578.72, 578.73,
    142.58, 360.64, 465.48, 465.49,
    218.06, 322.9, 323,
    104.84, 104.85, 0.01,
]

# Intermediate feature-map sizes (bytes-ish units) for the 6 valid split
# boundaries: input / after-layer-2 / after-layer-7 / after-14 / after-18 /
# after-31. Used to compute transfer latency = SS_d / D_tt.
SS_d = [3072 * 64, 4096, 2048, 1024, 512, 512]

# Scenario → (start_layer, end_layer) using the original ResNet34 1-based
# layer numbering used in the simulation. ``end == 0`` is the special
# "no-compute / forward only" scenario for the source device.
start = [0, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 8, 8, 8, 8, 15, 15, 15, 19, 19, 32]
end =   [0, 2, 7, 14, 18, 31, 34, 7, 14, 18, 31, 34, 14, 18, 31, 34, 18, 31, 34, 31, 34, 34]

# Reachable end-points for paths starting from each source scenario in
# device 0. Lifted from CRMC_eval_PDR.py — driven by the constraint that
# ``i_e_l + 1 == j_s_l`` in the graph builder.
end_point0 = [
    [28, 33, 37, 40, 42, 43, 50, 55, 59, 62, 64, 65],
    [33, 37, 40, 42, 43, 55, 59, 62, 64, 65],
    [37, 40, 42, 43, 59, 62, 64, 65],
    [40, 42, 43, 62, 64, 65],
    [42, 43, 64, 65],
    [43, 65],
]
end_point1: list[list[int]] = []
end_point2: list[list[int]] = []

# Default source scenarios on device 0 — start_layer == 0 (raw input) plus the
# forward-only scenario at index 0.
start_point = [0, 1, 2, 3, 4, 5, 6]

# Defaults pulled from CRMC_eval_PDR.py.
acc_thresh = 0.49
energy_thresh = 500
e_com = 0
e_trans = 0

# End-layer value (in original ResNet34 layer numbering) → split-point index
# used by ``model.resnet_split.split_resnet`` (0..4). 34 / 0 = whole-model
# scenarios mapped to the sentinel 7.
END_TO_SPLIT_POINT = {
    2: 0,
    7: 1,
    14: 2,
    18: 3,
    31: 4,
    34: 7,
    0: 7,
}
