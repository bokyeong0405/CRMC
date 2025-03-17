# computing power가 가장 좋은 device에게 전체 inference 할당

def com_best (D_tt, D_C, D_BER, SS_d, SS_f, e_com, e_trans):
    energy_consumption = [0, 0, 0]
    com_b = D_C[1]
    if D_C[2] > D_C[1]:
        com_b = D_C[2]

    if com_b == D_C[1]:
        inference_latency = SS_f[6] / D_C[1] + SS_d[0] / D_tt[0]
        # energy_consumption[0] = SS_d[0] * e_trans * (1/(1-D_BER[0]/100))
        # energy_consumption[1] = SS_f[5] * e_com
        energy_consumption[1] = SS_f[6]
    elif com_b == D_C[2]:
        inference_latency = SS_f[6] / D_C[2] + SS_d[0] / D_tt[2]
        # energy_consumption[0] = SS_d[0] * e_trans * (1/(1-D_BER[2]/100))
        # energy_consumption[2] = SS_f[5] * e_com
        energy_consumption[2] = SS_f[6]

    accuracy = 0.5052

    # print(energy_consumption)
    return inference_latency, accuracy, energy_consumption