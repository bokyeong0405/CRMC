from ResNet_predict_accuracy import *

def min_inter (D_tt, D_C, D_BER, e_com, e_trans):
    SS_f = [478.7, 104.85]  ## 파라미터 정리
    SS_d = 512
    energy_consumption = [0, 0, 0]
    com = D_C[1]
    if D_C[1] > D_C[2]:
        com = D_C[2]

    if com == D_C[1]:
        inference_latency = SS_f[0] / D_C[0] + SS_f[1] / com + SS_d / D_tt[0]
        # energy_consumption[0] = SS_f[0] * e_com + SS_d * e_trans
        # energy_consumption[1] = SS_f[1] * e_com
        energy_consumption[0] = SS_f[0]
        energy_consumption[1] = SS_f[1]
    elif com == D_C[2]:
        inference_latency = SS_f[0] / D_C[0] + SS_f[1] / com + SS_d / D_tt[2]
        # energy_consumption[0] = SS_f[0] * e_com + SS_d * e_trans
        # energy_consumption[2] = SS_f[1] * e_com
        energy_consumption[0] = SS_f[0]
        energy_consumption[2] = SS_f[1]

    ber = D_BER[0]
    if com == D_C[2]:
        ber = D_BER[2]

    sp = [18]
    accuracy = predict_accuracy_res(sp, ber)

    return inference_latency, accuracy, energy_consumption