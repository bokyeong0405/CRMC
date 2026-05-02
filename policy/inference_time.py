"""End-to-end inference latency estimator for a chosen GSPDA path.

Ported from ``resnet34_TinyImageNet/inference_time_resnet.py``.
"""


def inference_time(path, SS_f, D_C, D_BER, start, end, SS_d, D_tt, SS):
    compute_consumption = [0, 0, 0]
    trans_consumption = [0, 0, 0]

    if len(path) == 1:
        i_device = path[0] // SS
        inference = SS_f[6] / D_C[i_device]
        compute_consumption[i_device] = SS_f[6]
        return inference, compute_consumption, trans_consumption

    if len(path) == 2:
        i, j = path
        i_e_l = end[i % SS]
        i_device = i // SS
        j_device = j // SS

        x = _data_size_index(i_e_l)
        y = _link_index(i_device, j_device)

        if i_device == 0 and i_e_l == 0:
            if j_device == 1:
                inference = SS_f[6] / D_C[j_device] + SS_d[0] / D_tt[0] * (1 / (1 - D_BER[0] / 100))
            elif j_device == 2:
                inference = SS_f[6] / D_C[j_device] + SS_d[0] / D_tt[2] * (1 / (1 - D_BER[2] / 100))
        else:
            inference = SS_f[i % SS] / D_C[i_device] + SS_f[j % SS] / D_C[j_device] + SS_d[x] / D_tt[y]

        compute_consumption[i_device] = SS_f[i % SS]
        compute_consumption[j_device] = SS_f[j % SS]
        trans_consumption[i_device] = SS_d[x]
        return inference, compute_consumption, trans_consumption

    # len(path) == 3
    i, j, k = path
    i_e_l = end[i % SS]
    j_e_l = end[j % SS]
    i_device = i // SS
    j_device = j // SS
    k_device = k // SS

    x1 = _data_size_index(i_e_l)
    x2 = _data_size_index(j_e_l)
    y1 = _link_index(i_device, j_device)
    y2 = _link_index(k_device, j_device)

    if i_device == 0 and i_e_l == 0:
        if j_device == 1:
            inference = (
                SS_f[j % SS] / D_C[j_device]
                + SS_f[k % SS] / D_C[k_device]
                + SS_d[x1] / D_tt[y1] * (1 / (1 - D_BER[0] / 100))
                + SS_d[x2] / D_tt[y2]
            )
        elif j_device == 2:
            inference = (
                SS_f[j % SS] / D_C[j_device]
                + SS_f[k % SS] / D_C[k_device]
                + SS_d[x1] / D_tt[y1] * (1 / (1 - D_BER[2] / 100))
                + SS_d[x2] / D_tt[y2]
            )
    else:
        inference = (
            SS_f[i % SS] / D_C[i_device]
            + SS_f[j % SS] / D_C[j_device]
            + SS_f[k % SS] / D_C[k_device]
            + SS_d[x1] / D_tt[y1]
            + SS_d[x2] / D_tt[y2]
        )

    compute_consumption[i_device] = SS_f[i % SS]
    compute_consumption[j_device] = SS_f[j % SS]
    compute_consumption[k_device] = SS_f[k % SS]
    trans_consumption[i_device] = SS_d[x1]
    trans_consumption[j_device] = SS_d[x2]
    return inference, compute_consumption, trans_consumption


def _data_size_index(end_layer):
    return {0: 0, 2: 1, 7: 2, 14: 3, 18: 4, 31: 5}[end_layer]


def _link_index(a, b):
    pair = frozenset((a, b))
    if pair == frozenset((0, 1)):
        return 0
    if pair == frozenset((1, 2)):
        return 1
    if pair == frozenset((0, 2)):
        return 2
    raise ValueError(f"Unknown device pair: ({a}, {b})")
