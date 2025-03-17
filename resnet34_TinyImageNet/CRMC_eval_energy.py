import time
from GSPDA_resnet import *  # resnet 맞춤 GSPDA
from inference_time_resnet import *  #
from shortest_path_graph_resnet import *  # resnet 맞춤 GSPDA
from scheme_one_device_resnet import *  #
from scheme_min_inter_data_resnet import *  #
from scheme_com_power_best_resnet import *  #
from scheme_PDR_low_resnet import *  #

silence_tensorflow()
np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)  # 배열 전체 출력하는 코드

D = 3  # num of device
D_tt = [1000, 1000, 1000]  # device 간 전송 속도 1 <-> 2, 2 <-> 3, 3 <-> 1
D_C = [3, 25, 50]  # computing power
D_BER = [15, 15, 15]  # device 간 BER 1 <-> 2, 2 <-> 3, 3 <-> 1
# 가능한 split 경우의 수 splitting point 0, 1, 2, 3, 4
SS = 22
# 각 노드의 결정에서 계산해야 할 flops : 단위 MFLOPs
# SS_f = [0, 0.08, 5.4, 26.68, 80.51, 211.73, 211.74, 5.32, 26.6, 80.43, 211.65, 211.66, 21.28, 75.11, 206.33, 206.34, 53.83, 185.05, 185.06, 131.22, 131.23, 0.01]
SS_f = [0, 4.82, 118.06, 260.64, 478.7, 583.54, 583.55, 113.24, 255.82, 473.88, 578.72, 578.73, 142.58, 360.64, 465.48, 465.49, 218.06, 322.9, 323, 104.84, 104.85, 0.01]
# size of intermediate data input, 0, 1, 2, 3, 4 l
SS_d = [3072*64, 4096, 2048, 1024, 512, 512]
acc_thresh = 0.49  # accuracy 임계값 0.8, 0.8, 0.8, 0.8,
e_com = 0  # computation energy consumption 안씀
e_trans = 0  # transmission energy consumption 안씀
energy_thresh = 400
start = [0, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 8, 8, 8, 8, 15, 15, 15, 19, 19, 32]  # 노드에 할당된 시작 레이어
end = [0, 2, 7, 14, 18, 31, 34, 7, 14, 18, 31, 34, 14, 18, 31, 34, 18, 31, 34, 31, 34, 34]  # 노드에 할당된 끝 레이어
N_V = D * SS  # 노드 개수 (전체 경우의 수) 22*3 66

tmp = [100, 105, 110, 115, 120]
tmp = [300, 400, 500, 600, 700]  # 분할 지점 별 maximum computing 량 고려해서 설정해야 함.
# tmp = [300, 400, 500]
# tmp = [0, 10, 20, 30, 40]
tmp_cp = [3, 25, 50]
random_infer = []
random_accuracy = []
random_energy = []
burst_infer = []
burst_accuracy = []
burst_energy = []
one_infer = []
one_accuracy = []
one_energy = []
min_infer = []
min_accuracy = []
min_energy = []
comb_infer = []
comb_accuracy = []
comb_energy = []
lowB_infer = []
lowB_accuracy = []
lowB_energy = []


# 표준편차 저장용 리스트
burst_infer_std = []
burst_accuracy_std = []
burst_energy_std = []
one_infer_std = []
one_accuracy_std = []
one_energy_std = []
min_infer_std = []
min_accuracy_std = []
min_energy_std = []
comb_infer_std = []
comb_accuracy_std = []
comb_energy_std = []
lowB_infer_std = []
lowB_accuracy_std = []
lowB_energy_std = []

iter_num = 10000

for j in tmp:
    print(j)
    #random_val = 0
    # random_acc = 0
    # random_ene = [0, 0, 0]
    # burst_val = 0
    # burst_acc = 0
    # burst_ene = [0, 0, 0]
    # one_val = 0
    # one_acc = 0
    # one_ene = [0, 0, 0]
    # min_val = 0
    # min_acc = 0
    # min_ene = [0, 0, 0]
    # comb_val = 0
    # comb_acc = 0
    # comb_ene = [0, 0, 0]
    # lowB_val = 0
    # lowB_acc = 0
    # lowB_ene = [0, 0, 0]

    burst_val = []
    burst_acc = []
    burst_ene = []

    one_val = []
    one_acc = []
    one_ene = []

    min_val = []
    min_acc = []
    min_ene = []

    comb_val = []
    comb_acc = []
    comb_ene = []

    lowB_val = []
    lowB_acc = []
    lowB_ene = []

    energy_thresh = j

    for i in range(iter_num):
        print(i)
        D_C = [3]
        for tmp_c in range(len(tmp_cp) - 1):
            mean = tmp_cp[tmp_c + 1]
            std_dev = 5
            random_number = np.random.normal(mean, std_dev)
            while random_number < 3:
                random_number = np.random.normal(mean, std_dev)
            D_C.append(random_number)
            std_dev += 5
        # print('computing power : ', D_C)

        D_tt = []
        for d_l in range(3):
            mean = 1000
            std_dev = 200
            random_number = np.random.normal(mean, std_dev)
            while random_number < 0:
                random_number = np.random.normal(mean, std_dev)
            D_tt.append(random_number)

        D_BER = []
        for d_l in range(3):
            mean = 15
            std_dev = 2
            random_number = np.random.normal(mean, std_dev)
            while random_number < 0:
                random_number = np.random.normal(mean, std_dev)
            D_BER.append(random_number)

        A = create_graph(N_V, SS_f, D_C, D_BER, SS_d, D_tt, start, end, SS)  # 그래프 에지 정보 생성 : 파일명 shortest_path_graph.py
        G = nx.DiGraph(A)  # 에지 정보들로 그래프 생성
        # print("A", A)
        # print(G)
        # exit()
        start_point = [0, 1, 2, 3, 4, 5, 6]

        # start point에 따라 가능한 end point 각각 설정
        end_point0 = [[28, 33, 37, 40, 42, 43, 50, 55, 59, 62, 64, 65], [33, 37, 40, 42, 43, 55, 59, 62, 64, 65],
                      [37, 40, 42, 43, 59, 62, 64, 65], [40, 42, 43, 62, 64, 65], [42, 43, 64, 65], [43, 65]]
        end_point1 = []
        end_point2 = []

        # # print('thresh hold : ', acc_thresh[t])
        # optimal_path, short_inference, energy_consumption, minindex, accuracy = GSPDA(start_point, end_point0, end_point1, end_point2, G, SS_f, D_C, end, D_BER, acc_thresh[t], A, start, SS_d, D_tt, 0, SS, e_com, e_trans, energy_thresh)
        # # print("*****************  Random Error  *********************")
        # # print("random optimal path : ", optimal_path)
        # # print("inference latency : ", short_inference[minindex])
        # # print('random accuracy : ', accuracy)
        # random_acc = random_acc + accuracy
        # random_val = random_val + short_inference[minindex]
        # random_ene = [x + y for x, y in zip(random_ene, energy_consumption[minindex])]
        # # print('##########################################################################')
        # print('random optimal path :', optimal_path)
        # # print('random accuracy :', accuracy)
        # # print('random energy consumption : ', energy_consumption[minindex])
        # # print('--------------------------------------------------------------------------')

        # A = create_graph(N_V, SS_f, D_C, D_BER, SS_d, D_tt, start, end, SS)  # 그래프 에지 정보 생성 : 파일명 shortest_path_graph.py
        # G = nx.DiGraph(A)  # 에지 정보들로 그래프 생성
        #
        optimal_path, short_inference, energy_consumption, minindex, accuracy = GSPDA(start_point, end_point0, end_point1, end_point2, G, SS_f, D_C, end, D_BER, acc_thresh, A, start, SS_d, D_tt, 1, SS, e_com, e_trans, energy_thresh)
        print()
        print("*****************  Burst Error  *********************")
        print("burst optimal path : ", optimal_path)
        print("inference latency : ", short_inference[minindex])
        print("load : ", energy_consumption[minindex])
        print("accuracy : ", accuracy)
        print("*****************************************************")
        print()

        # burst_acc = burst_acc + accuracy
        # burst_val = burst_val + short_inference[minindex]
        # burst_ene = [x + y for x, y in zip(burst_ene, energy_consumption[minindex])]
        # # print('burst optimal path :', optimal_path)
        # # print('burst accuracy :', accuracy)
        # # print('burst energy consumption :', energy_consumption[minindex])
        #
        # inference_latency, accuracy, energy_consumption = one_device(D_C, SS_f, e_com)
        # one_val = one_val + inference_latency
        # one_acc = one_acc + accuracy
        # one_ene[0] = one_ene[0] + energy_consumption
        #
        # inference_latency, accuracy, energy_consumption = min_inter(D_tt, D_C, D_BER, e_com, e_trans)
        # min_val = min_val + inference_latency
        # min_acc = min_acc + accuracy
        # min_ene = [x + y for x, y in zip(min_ene, energy_consumption)]
        #
        # inference_latency, accuracy, energy_consumption = com_best(D_tt, D_C, D_BER, SS_d, SS_f, e_com, e_trans)
        # comb_val = comb_val + inference_latency
        # comb_acc = comb_acc + accuracy
        # comb_ene = [x + y for x, y in zip(comb_ene, energy_consumption)]
        #
        # inference_latency, accuracy, energy_consumption = low_BER(D_tt, D_C, SS_d, SS_f, e_com, e_trans, D_BER)
        # lowB_val = lowB_val + inference_latency
        # lowB_acc = lowB_acc + accuracy
        # lowB_ene = [x + y for x, y in zip(lowB_ene, energy_consumption)]

        burst_val.append(short_inference[minindex])
        burst_acc.append(accuracy)
        burst_ene.append(energy_consumption[minindex])

        inference_latency, accuracy, energy_consumption = one_device(D_C, SS_f, e_com)
        one_val.append(inference_latency)
        one_acc.append(accuracy)
        one_ene.append([energy_consumption, 0, 0])

        inference_latency, accuracy, energy_consumption = min_inter(D_tt, D_C, D_BER, e_com, e_trans)
        min_val.append(inference_latency)
        min_acc.append(accuracy)
        min_ene.append(energy_consumption)

        inference_latency, accuracy, energy_consumption = com_best(D_tt, D_C, D_BER, SS_d, SS_f, e_com, e_trans)
        comb_val.append(inference_latency)
        comb_acc.append(accuracy)
        comb_ene.append(energy_consumption)

        inference_latency, accuracy, energy_consumption = low_BER(D_tt, D_C, SS_d, SS_f, e_com, e_trans, D_BER)
        lowB_val.append(inference_latency)
        lowB_acc.append(accuracy)
        lowB_ene.append(energy_consumption)

    # random_infer.append(random_val/iter_num)
    # random_accuracy.append(random_acc/iter_num)
    # for ran_i in range(len(random_ene)):
    #     random_ene[ran_i] = random_ene[ran_i] / iter_num
    # random_energy.append(random_ene)

    # burst_infer.append(burst_val/iter_num)
    # burst_accuracy.append(burst_acc/iter_num)
    # for bur_i in range(len(burst_ene)):
    #     burst_ene[bur_i] = burst_ene[bur_i] / iter_num
    # burst_energy.append(burst_ene)
    #
    # one_infer.append(one_val/iter_num)
    # one_accuracy.append(one_acc/iter_num)
    # for one_i in range(len(one_ene)):
    #     one_ene[one_i] = one_ene[one_i] / iter_num
    # one_energy.append(one_ene)
    #
    # min_infer.append(min_val / iter_num)
    # min_accuracy.append(min_acc / iter_num)
    # for min_i in range(len(min_ene)):
    #     min_ene[min_i] = min_ene[min_i] / iter_num
    # min_energy.append(min_ene)
    #
    # comb_infer.append(comb_val / iter_num)
    # comb_accuracy.append(comb_acc / iter_num)
    # for comb_i in range(len(comb_ene)):
    #     comb_ene[comb_i] = comb_ene[comb_i] / iter_num
    # comb_energy.append(comb_ene)
    #
    # lowB_infer.append(lowB_val / iter_num)
    # lowB_accuracy.append(lowB_acc / iter_num)
    # for lowB_i in range(len(lowB_ene)):
    #     lowB_ene[lowB_i] = lowB_ene[lowB_i] / iter_num
    # lowB_energy.append(lowB_ene)

    burst_infer.append(np.mean(burst_val))
    burst_accuracy.append(np.mean(burst_acc))
    burst_energy.append(np.mean(burst_ene, axis=0))
    burst_infer_std.append(np.std(burst_val))
    burst_accuracy_std.append(np.std(burst_acc))
    burst_energy_std.append(np.std(burst_ene, axis=0))

    one_infer.append(np.mean(one_val))
    one_accuracy.append(np.mean(one_acc))
    one_energy.append(np.mean(one_ene, axis=0))
    one_infer_std.append(np.std(one_val))
    one_accuracy_std.append(np.std(one_acc))
    one_energy_std.append(np.std(one_ene, axis=0))

    min_infer.append(np.mean(min_val))
    min_accuracy.append(np.mean(min_acc))
    min_energy.append(np.mean(min_ene, axis=0))
    min_infer_std.append(np.std(min_val))
    min_accuracy_std.append(np.std(min_acc))
    min_energy_std.append(np.std(min_ene, axis=0))

    comb_infer.append(np.mean(comb_val))
    comb_accuracy.append(np.mean(comb_acc))
    comb_energy.append(np.mean(comb_ene, axis=0))
    comb_infer_std.append(np.std(comb_val))
    comb_accuracy_std.append(np.std(comb_acc))
    comb_energy_std.append(np.std(comb_ene, axis=0))

    lowB_infer.append(np.mean(lowB_val))
    lowB_accuracy.append(np.mean(lowB_acc))
    lowB_energy.append(np.mean(lowB_ene, axis=0))
    lowB_infer_std.append(np.std(lowB_val))
    lowB_accuracy_std.append(np.std(lowB_acc))
    lowB_energy_std.append(np.std(lowB_ene, axis=0))

# print("*****************  Random Error  *********************")
# print('random inference time : ', random_infer)
# print('random accuracy : ', random_accuracy)
# print('random energy consumption : ', random_energy)
# np.save('C:/Users/user/PycharmProjects/SplitComputing/eval_rand_infer_ber', random_infer)
# np.save('C:/Users/user/PycharmProjects/SplitComputing/eval_rand_acc_ber', random_accuracy)
# np.save('C:/Users/user/PycharmProjects/SplitComputing/eval_rand_energy_ber', random_energy)

WHERE_TO_SAVE = 'C:/Users/user/PycharmProjects/SplitComputingv2/resnet/eval'

print("******************  Burst Error  *********************")
print('burst_inference time :', burst_infer, burst_infer_std)
print('burst accuracy : ', burst_accuracy, burst_accuracy_std)
print('burst energy consumption : ', burst_energy, burst_energy_std)
np.save(WHERE_TO_SAVE + '/eval_burst_infer_energy', [burst_infer, burst_infer_std])
np.save(WHERE_TO_SAVE + '/eval_burst_acc_energy', [burst_accuracy, burst_accuracy_std])
np.save(WHERE_TO_SAVE + '/eval_burst_energy_energy', [burst_energy, burst_energy_std])
print("******************  one-device  *********************")
print('inference time :', one_infer, one_infer_std)
print('accuracy : ', one_accuracy, one_accuracy_std)
print('energy consumption : ', one_energy, one_energy_std)
np.save(WHERE_TO_SAVE + '/eval_one_infer_energy', [one_infer, one_infer_std])
np.save(WHERE_TO_SAVE + '/eval_one_acc_energy', [one_accuracy, one_accuracy_std])
np.save(WHERE_TO_SAVE + '/eval_one_energy_energy', [one_energy, one_energy_std])
print("******************  min-intermediate data  *********************")
print('inference time :', min_infer, min_infer_std)
print('accuracy : ', min_accuracy, min_accuracy_std)
print('energy consumption : ', min_energy, min_energy_std)
np.save(WHERE_TO_SAVE + '/eval_min_infer_energy', [min_infer, min_infer_std])
np.save(WHERE_TO_SAVE + '/eval_min_acc_energy', [min_accuracy, min_accuracy_std])
np.save(WHERE_TO_SAVE + '/eval_min_energy_energy', [min_energy, min_energy_std])
print("******************  Best computational power  *********************")
print('inference time :', comb_infer, comb_infer_std)
print('accuracy : ', comb_accuracy, comb_infer_std)
print('energy consumption : ', comb_energy, comb_infer_std)
np.save(WHERE_TO_SAVE + '/eval_comb_infer_energy', [comb_infer, comb_infer_std])
np.save(WHERE_TO_SAVE + '/eval_comb_acc_energy', [comb_accuracy, comb_accuracy_std])
np.save(WHERE_TO_SAVE + '/eval_comb_energy_energy', [comb_energy, comb_energy_std])
print("******************  send low BER  *********************")
print('inference time :', lowB_infer, lowB_infer_std)
print('accuracy : ', lowB_accuracy, lowB_accuracy_std)
print('energy consumption : ', lowB_energy, lowB_energy_std)
np.save(WHERE_TO_SAVE + '/eval_lowB_infer_energy', [lowB_infer, lowB_infer_std])
np.save(WHERE_TO_SAVE + '/eval_lowB_acc_energy', [lowB_accuracy, lowB_accuracy_std])
np.save(WHERE_TO_SAVE + '/eval_lowB_energy_energy', [lowB_energy, lowB_energy_std])

# print("******************  Burst Error  *********************")
# print('burst_inference time :', burst_infer)
# print('burst accuracy : ', burst_accuracy)
# print('burst energy consumption : ', burst_energy)
# np.save(WHERE_TO_SAVE + '/eval_burst_infer_energy', burst_infer)
# np.save(WHERE_TO_SAVE + '/eval_burst_acc_energy', burst_accuracy)
# np.save(WHERE_TO_SAVE + '/eval_burst_energy_energy', burst_energy)
# print("******************  one-device  *********************")
# print('inference time :', one_infer)
# print('accuracy : ', one_accuracy)
# print('energy consumption : ', one_energy)
# np.save(WHERE_TO_SAVE + '/eval_one_infer_energy', one_infer)
# np.save(WHERE_TO_SAVE + '/eval_one_acc_energy', one_accuracy)
# np.save(WHERE_TO_SAVE + '/eval_one_energy_energy', one_energy)
# print("******************  min-intermediate data  *********************")
# print('inference time :', min_infer)
# print('accuracy : ', min_accuracy)
# print('energy consumption : ', min_energy)
# np.save(WHERE_TO_SAVE + '/eval_min_infer_energy', min_infer)
# np.save(WHERE_TO_SAVE + '/eval_min_acc_energy', min_accuracy)
# np.save(WHERE_TO_SAVE + '/eval_min_energy_energy', min_energy)
# print("******************  Best computational power  *********************")
# print('inference time :', comb_infer)
# print('accuracy : ', comb_accuracy)
# print('energy consumption : ', comb_energy)
# np.save(WHERE_TO_SAVE + '/eval_comb_infer_energy', comb_infer)
# np.save(WHERE_TO_SAVE + '/eval_comb_acc_energy', comb_accuracy)
# np.save(WHERE_TO_SAVE + '/eval_comb_energy_energy', comb_energy)
# print("******************  send low BER  *********************")
# print('inference time :', lowB_infer)
# print('accuracy : ', lowB_accuracy)
# print('energy consumption : ', lowB_energy)
# np.save(WHERE_TO_SAVE + '/eval_lowB_infer_energy', lowB_infer)
# np.save(WHERE_TO_SAVE + '/eval_lowB_acc_energy', lowB_accuracy)
# np.save(WHERE_TO_SAVE + '/eval_lowB_energy_energy', lowB_energy)
