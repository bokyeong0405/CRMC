import numpy as np
import networkx as nx

def inference_time (path, SS_f, D_C, D_BER, start, end, SS_d, D_tt, SS):
    compute_consumption = [0, 0, 0]
    trans_consumption = [0, 0, 0]
    if len(path) == 1:
        i_device = path[0] // SS
        inference = SS_f[6] / D_C[i_device]
        compute_consumption[i_device] = SS_f[6]
    elif len(path) == 2:
        # print(path)
        i = path[0]
        j = path[1]
        i_s_l = start[i % SS]
        i_e_l = end[i % SS]
        j_s_l = start[j % SS]
        j_e_l = end[j % SS]
        i_device = (i // SS)
        j_device = (j // SS)

        # data size 결정
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
        # 전송 속도 결정
        if (i_device == 0 and j_device == 1) or (i_device == 1 and j_device == 0):
            y = 0
        elif (i_device == 1 and j_device == 2) or (i_device == 2 and j_device == 1):
            y = 1
        elif (i_device == 2 and j_device == 0) or (i_device == 0 and j_device == 2):
            y = 2

        if i_device == 0 and i_e_l == 0:
            if j_device == 1:
                inference = SS_f[6] / D_C[j_device] + SS_d[0] / D_tt[0] * (1 / (1 - D_BER[0] / 100))
            elif j_device == 2:
                inference = SS_f[6] / D_C[j_device] + SS_d[0] / D_tt[2] * (1 / (1 - D_BER[2] / 100))
        else:
            inference = SS_f[i % SS] / D_C[i_device] + SS_f[j % SS] / D_C[j_device] + SS_d[x] / D_tt[y]  # computing latency + transfer latency(data size / 전송 속도)
        compute_consumption[i_device] = SS_f[i % SS]  # device 별 compute flops 저장
        compute_consumption[j_device] = SS_f[j % SS]
        trans_consumption[i_device] = SS_d[x]  # device 간 전송 데이터 저장
    elif len(path) == 3:
        i = path[0]
        j = path[1]
        k = path[2]
        i_s_l = start[i % SS]
        i_e_l = end[i % SS]
        j_s_l = start[j % SS]
        j_e_l = end[j % SS]
        k_s_l = start[k % SS]
        k_e_l = end[k % SS]
        i_device = (i // SS)
        j_device = (j // SS)
        k_device = (k // SS)
        # data size 결정
        if i_e_l == 2:
            x1 = 1
        elif i_e_l == 7:
            x1 = 2
        elif i_e_l == 14:
            x1 = 3
        elif i_e_l == 18:
            x1 = 4
        elif i_e_l == 31:
            x1 = 5
        elif i_e_l == 0:
            x1 = 0

        if j_e_l == 2:
            x2 = 1
        elif j_e_l == 7:
            x2 = 2
        elif j_e_l == 14:
            x2 = 3
        elif j_e_l == 18:
            x2 = 4
        elif j_e_l == 31:
            x2 = 5
        elif j_e_l == 0:
            x2 = 0

        # 전송 속도 결정
        if (i_device == 0 and j_device == 1) or (i_device == 1 and j_device == 0):
            y1 = 0
        elif (i_device == 1 and j_device == 2) or (i_device == 2 and j_device == 1):
            y1 = 1
        elif (i_device == 2 and j_device == 0) or (i_device == 0 and j_device == 2):
            y1 = 2

        if (k_device == 0 and j_device == 1) or (k_device == 1 and j_device == 0):
            y2 = 0
        elif (k_device == 1 and j_device == 2) or (k_device == 2 and j_device == 1):
            y2 = 1
        elif (k_device == 2 and j_device == 0) or (k_device == 0 and j_device == 2):
            y2 = 2

        if i_device == 0 and i_e_l == 0:
            if j_device == 1:
                inference = SS_f[j % SS] / D_C[j_device] + SS_f[k % SS] / D_C[k_device] + SS_d[x1] / D_tt[y1] * (1 / (1 - D_BER[0] / 100)) + SS_d[x2] / D_tt[y2]
            elif j_device == 2:
                inference = SS_f[j % SS] / D_C[j_device] + SS_f[k % SS] / D_C[k_device] + SS_d[x1] / D_tt[y1] * (1 / (1 - D_BER[2] / 100)) + SS_d[x2] / D_tt[y2]
        else:
            inference = SS_f[i % SS] / D_C[i_device] + SS_f[j % SS] / D_C[j_device] + SS_f[k % SS] / D_C[k_device] + SS_d[x1] / D_tt[y1] + SS_d[x2] / D_tt[y2]
        compute_consumption[i_device] = SS_f[i % SS]  # device 별 compute flops 저장
        compute_consumption[j_device] = SS_f[j % SS]
        compute_consumption[k_device] = SS_f[k % SS]
        trans_consumption[i_device] = SS_d[x1]  # device 간 전송 데이터 저장
        trans_consumption[j_device] = SS_d[x2]
    # print("inference time:", inference)
    return inference, compute_consumption, trans_consumption