import sys
from silence_tensorflow import silence_tensorflow

import networkx as nx

# from set_ss_sb_5_device import *
from set_ss_sb_resnet import *  # resnet 맞춤
from predict_accuracy import *  # 안씀
from ResNet_predict_accuracy import *  # resnet
# from inference_time_5_device import *
from inference_time_resnet import *  # resnet

def set_inf (A, P):
    # print("set_inf run")
    if len(P) == 1:
        A[P[0], P[0]] = np.inf
    elif len(P) == 2:
        A[P[0], P[1]] = np.inf
    elif len(P) == 3:
        A[P[0], P[1]] = np.inf
        A[P[1], P[2]] = np.inf
    return A

def acc_check_and_inf (A, P, end, D_BER, SS, acc_thresh, possible_path, acc, l, d):
    # print("acc_check_and_inf run")
    sp, b = set_ss_sb(P, end, D_BER, SS)  # accuracy 계산을 위해 Ss와 SB 구하는 코드
    if l == 0:
        accuracy = predict_accuracy(sp, b)
    elif l == 1:
        accuracy = predict_accuracy_res(sp, b)

    if accuracy > acc_thresh:
        possible_path.append(P)
        acc.append(accuracy)
        A = set_inf(A, P)
        G = nx.DiGraph(A)
        # print(d)
    else:
        A = set_inf(A, P)
        G = nx.DiGraph(A)
    return G, possible_path, acc

def find_infer_short (start_point, end_point0, G, A, end, D_BER, SS, l):
    # print("find_infer_short run")
    optimal = []
    distan = []
    for i in start_point:  # shortest path 모든 경우의 수 구하기
        if i % SS == 6:  # device 1이 전부 계산할 경우
            P = nx.shortest_path(G, source=i, target=i, weight='weight')
            d = G[i][i]['weight']
            # d = nx.shortest_path_length(G, source=i, target=i, weight='weight')
            # d = SS_f[4] / D_C[i // 15]
            # print("Shortest path: ", P)
            # print("Shortest distance: ", d)
            if len(P) > 3:
                A = set_inf(A, P)
                G = nx.DiGraph(A)
                continue
            optimal.append(P)
            distan.append(d)
        elif i // SS == 0:  # device 1에서 2,3 쪽으로 끝날 수 있는 노드
            for j in end_point0[i % SS]:
                P = nx.shortest_path(G, source=i, target=j, weight='weight')
                d = nx.shortest_path_length(G, source=i, target=j, weight='weight')
                # print("Shortest path: ", P)
                # print("Shortest distance: ", d)
                if len(P) > 3:
                    A = set_inf(A, P)
                    G = nx.DiGraph(A)
                    continue
                optimal.append(P)
                distan.append(d)
    # print("optimal")
    # print(optimal)
    # print("distan")
    # print(distan)

    # 모든 경우 중 가장 짧은 path 찾기
    minindex = np.argmin(distan)
    # print(minindex)
    if distan[minindex] == np.inf:
        return False
    short_path = optimal[minindex]
    sp, b = set_ss_sb(short_path, end, D_BER, SS)  # accuracy 계산을 위해 Ss와 SB 구하는 코드
    if l == 0:
        accuracy = predict_accuracy(sp, b)
    elif l == 1:
        accuracy = predict_accuracy_res(sp, b)

    # print("sp, b:", sp, b)
    # print(short_path, accuracy)
    # exit()
    return short_path, accuracy

def GSPDA (start_point, end_point0, end_point1, end_point2, G, SS_f, D_C, end, D_BER, acc_thresh, A, start, SS_d, D_tt, l, SS, e_com, e_trans, energy_thresh):

    acc_thresh_min = 0.6  ## 최소 threshold 추가 ##

    # 그래프에서 shortest_path 탐색 : 파일명 shortestpath_num_of_case
    infer_short_path, infer_short_acc = find_infer_short(start_point, end_point0, G, A, end, D_BER, SS, l)

    possible_path = []  # accuracy threshold 만족하는 path들만 고른 상태
    acc = []  # 각 path의 accuracy
    for i in start_point:  # shortest path 모든 경우의 수 구하기
        if i % SS == 6:  # device 1이 전부 계산할 경우
            P = nx.shortest_path(G, source=i, target=i, weight='weight')
            d = G[i][i]['weight']
            if len(P) > 3:
                A = set_inf(A, P)
                G = nx.DiGraph(A)
                continue
            # print("path 후보", P)

            G, possible_path, acc = acc_check_and_inf(A, P, end, D_BER, SS, acc_thresh_min, possible_path, acc, l, d)  ## 최소 acc 기준으로 path 찾기!! ##
        elif i // SS == 0:  # device 1에서 2,3 쪽으로 끝날 수 있는 노드
            for j in end_point0[i % SS]:
                P = nx.shortest_path(G, source=i, target=j, weight='weight')
                d = nx.shortest_path_length(G, source=i, target=j, weight='weight')
                if len(P) > 3:
                    A = set_inf(A, P)
                    G = nx.DiGraph(A)
                    continue
                # print("path 후보", P)
                G, possible_path, acc = acc_check_and_inf(A, P, end, D_BER, SS, acc_thresh_min, possible_path, acc, l, d)  ## 최소 acc 기준으로 path 찾기!! ##
        if P is False:
            break

    # print('possible path', possible_path)
    # print('accuracy', acc)
    # exit()

    if len(possible_path) == 0:  # 조건을 만족하는 path가 없는 경우 inference latency가 가장 빠른 걸로 기기링  ## 최소 tresh도 만족 못하면 infer 가장 짧은거!
        possible_path.append(infer_short_path)
        acc.append(infer_short_acc)

    # 각 path의 inference 시간
    short_inference = []

    # energy consumption
    com_consumption = []
    tran_consumption = []
    energy_consumption = []
    print("possible_path")
    print("len PP", len(possible_path))
    # print(possible_path)
    for i in range(len(possible_path)):
        print(possible_path[i])
        inference, compute_consumption, trans_consumption = inference_time(possible_path[i], SS_f, D_C, D_BER, start, end, SS_d, D_tt, SS)
        short_inference.append(inference)
        com_consumption.append(compute_consumption)
        tran_consumption.append(trans_consumption)
        print("inference", inference)
        print("compute_consumption", compute_consumption)
        print("trans_consumption", trans_consumption)


    # device 별 계산한 flops를 energy_consumption에 그대로 저장
    energy_consumption = com_consumption
    # print(short_inference)

    flag = True
    # print("short_inference :", short_inference)
    while(flag):
        minindex = np.argmin(short_inference)
        # print("minindex: ", minindex)
        if energy_consumption[minindex][0] < energy_thresh and energy_consumption[minindex][1] < energy_thresh and energy_consumption[minindex][2] < energy_thresh:
            optimal_path = possible_path[minindex]
            accuracy = acc[minindex]
            flag = False
        else:
            del short_inference[minindex]
            del energy_consumption[minindex]
            del possible_path[minindex]
            del acc[minindex]

            if len(short_inference) == 0:
                possible_path.append(infer_short_path)
                acc.append(infer_short_acc)
                inference, compute_consumption, trans_consumption = inference_time(possible_path[0], SS_f, D_C, D_BER, start, end, SS_d, D_tt, SS)
                short_inference.append(inference)
                com_consumption.append(compute_consumption)
                tran_consumption.append(trans_consumption)

                minindex = np.argmin(short_inference)
                optimal_path = possible_path[minindex]
                energy_consumption = com_consumption
                accuracy = acc[minindex]
                print("GSPDA/load threshold/exception execute")
                flag = False



    # print(optimal_path)
    # print(accuracy)
    # print("inference latency :", short_inference[minindex])
    # print(energy_consumption)
    return optimal_path, short_inference, energy_consumption, minindex, accuracy