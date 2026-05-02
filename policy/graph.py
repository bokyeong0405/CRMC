"""Graph construction for CRMC GSPDA.

Ported verbatim from ``resnet34_TinyImageNet/shortest_path_graph_resnet.py``
with light cleanup (no behavioural changes).
"""

import numpy as np


def create_graph(N_V, SS_f, D_C, D_BER, SS_d, D_tt, start, end, SS):
    A = np.zeros((N_V, N_V))
    for i in range(N_V):
        for j in range(N_V):
            i_s_l = start[i % SS]
            i_e_l = end[i % SS]
            j_s_l = start[j % SS]
            j_e_l = end[j % SS]
            i_device = i // SS
            j_device = j // SS

            if i_device == j_device:
                # One device computes the entire 1..34 layer range.
                if i_s_l == 1 and j_s_l == 1 and i_e_l == 34 and j_e_l == 34:
                    if i_device == 2:
                        A[i, j] = SS_f[6] / D_C[i_device] + SS_d[0] / D_tt[2] * (1 / (1 - D_BER[2] / 100))
                    elif i_device == 1:
                        A[i, j] = SS_f[6] / D_C[i_device] + SS_d[0] / D_tt[0] * (1 / (1 - D_BER[0] / 100))
                    else:
                        A[i, j] = SS_f[6] / D_C[i_device]
            else:
                # Inter-device hop is valid only when layers are contiguous.
                if i_e_l + 1 == j_s_l:
                    if i_e_l == 2:
                        x = 1
                    elif i_e_l == 7:
                        x = 2
                    elif i_e_l == 14:
                        x = 3
                    elif i_e_l == 18:
                        x = 4
                    elif i_e_l == 31:
                        x = 5
                    elif i_e_l == 0:
                        x = 0

                    if (i_device == 0 and j_device == 1) or (i_device == 1 and j_device == 0):
                        y = 0
                    elif (i_device == 1 and j_device == 2) or (i_device == 2 and j_device == 1):
                        y = 1
                    elif (i_device == 2 and j_device == 0) or (i_device == 0 and j_device == 2):
                        y = 2
                    else:
                        # Should be unreachable for D=3.
                        continue

                    if i_device == 0:
                        # Source device "forward-only" scenario (i_e_l == 0)
                        # already accounts for the BER-induced retransmit
                        # penalty on the user→server link.
                        if i_e_l == 0:
                            if j_device == 1:
                                A[i, j] = SS_f[j % SS] / D_C[j_device] + SS_d[0] / D_tt[0] * (1 / (1 - D_BER[0] / 100))
                            elif j_device == 2:
                                A[i, j] = SS_f[j % SS] / D_C[j_device] + SS_d[0] / D_tt[2] * (1 / (1 - D_BER[2] / 100))
                        else:
                            A[i, j] = SS_f[i % SS] / D_C[i_device] + SS_f[j % SS] / D_C[j_device] + SS_d[x] / D_tt[y]
                    else:
                        A[i, j] = SS_f[j % SS] / D_C[j_device] + SS_d[x] / D_tt[y]

    return A
