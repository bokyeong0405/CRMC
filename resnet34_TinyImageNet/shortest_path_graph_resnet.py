import numpy as np

def create_graph (N_V, SS_f, D_C, D_BER, SS_d, D_tt, start, end, SS):
    A = np.zeros((N_V, N_V))
    # SS = 22, N_V = 66
    for i in range(N_V):
        for j in range(N_V):
            i_s_l = start[i % SS]
            i_e_l = end[i % SS]
            j_s_l = start[j % SS]
            j_e_l = end[j % SS]
            i_device = (i // SS)
            j_device = (j // SS)

            if i_device == j_device:
                if i_s_l == 1 and j_s_l == 1 and i_e_l == 34 and j_e_l == 34:  # 한 device가 1-16이 다 계산하는 경우
                    # computing latency = flops / computing power
                    if i_device == 2:
                        A[i, j] = SS_f[6] / D_C[i_device] + SS_d[0] / D_tt[2] * (1/(1-D_BER[2]/100))
                    elif i_device == 1:
                        A[i, j] = SS_f[6] / D_C[i_device] + SS_d[0] / D_tt[0] * (1/(1-D_BER[0]/100))
                    else:
                        A[i, j] = SS_f[6] / D_C[i_device]
            else:
                if i_e_l + 1 == j_s_l:
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
                    else:
                        print("error !! y set")

                    if i_device == 0:
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