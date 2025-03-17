# BER이 가장 낮은 device에게 offload : layer 4에서 split
from ResNet_predict_accuracy import *

def low_BER (D_tt, D_C, SS_d, SS_f, e_com, e_trans, D_BER):
    energy_consumption = [0, 0, 0]
    min_ber = D_BER[0]
    if D_BER[2] < D_BER[0]:
        min_ber = D_BER[2]

    if min_ber == D_BER[0]:
        inference_latency = SS_f[1] / D_C[0] + SS_d[1] / D_tt[0] + SS_f[11] / D_C[1]
        # energy_consumption[0] = SS_f[1] * e_com + SS_d[1] * e_trans
        # energy_consumption[1] = SS_f[9] * e_com
        energy_consumption[0] = SS_f[1]
        energy_consumption[1] = SS_f[11]
    elif min_ber == D_BER[2]:
        inference_latency = SS_f[1] / D_C[0] + SS_d[1] / D_tt[2] + SS_f[11] / D_C[2]
        # energy_consumption[0] = SS_f[1] * e_com + SS_d[1] * e_trans
        # energy_consumption[2] = SS_f[9] * e_com
        energy_consumption[0] = SS_f[1]
        energy_consumption[2] = SS_f[11]

    ber = min_ber
    sp = [2]
    accuracy = predict_accuracy_res(sp, ber)

    return inference_latency, accuracy, energy_consumption