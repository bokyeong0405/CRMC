import numpy as np
import matplotlib.pyplot as plt

# print("BASS")
# a = np.load('C:/Users/user/PycharmProjects/SplitComputing/eval_rand_infer_ber.npy')
# rand_inference_latency = np.array(a)
# print(rand_inference_latency)
# a = np.load('C:/Users/user/PycharmProjects/SplitComputing/eval_rand_acc_ber.npy')
# rand_accuracy = np.array(a)
# print(rand_accuracy)
# a = np.load('C:/Users/user/PycharmProjects/SplitComputing/eval_rand_energy_ber.npy')
# rand_energy_consumption = np.array(a)
# print(rand_energy_consumption)
# print()
# rand_ec = []
# for i in range(5):
#     maxindex = np.argmax(rand_energy_consumption[i])
#     rand_ec.append(rand_energy_consumption[i][maxindex]/10000)
# rand_ec_1 = []
# for i in range(5):
#     rand_ec_1.append(rand_energy_consumption[i][0])
#
WHERE_FROM_LOAD = 'C:/Users/user/PycharmProjects/SplitComputingv2/resnet/eval'

a = np.load(WHERE_FROM_LOAD + '/eval_burst_infer_energy.npy')
burst_inference_latency = np.array(a[0])
burst_inference_latency_std = np.array(a[1])
print(burst_inference_latency)
print(burst_inference_latency_std)
a = np.load(WHERE_FROM_LOAD + '/eval_burst_acc_energy.npy')
burst_accuracy = np.array(a[0])
burst_accuracy_std = np.array(a[1])
print(burst_accuracy)
print(burst_accuracy_std)
a = np.load(WHERE_FROM_LOAD + '/eval_burst_energy_energy.npy')
burst_energy_consumption = np.array(a[0])
burst_energy_consumption_std = np.array(a[1])
print(burst_energy_consumption)
print(burst_energy_consumption_std)
burst_ec = []
burst_ec_std = []

for i in range(5):
    maxindex = np.argmax(burst_energy_consumption[i])
    burst_ec.append(burst_energy_consumption[i][maxindex])
    burst_ec_std.append(burst_energy_consumption_std[i][maxindex])
# burst_ec_1 = []
# for i in range(5):
#     burst_ec_1.append(burst_energy_consumption[i][0])

print(burst_ec)
# print(burst_ec_1)
# exit()

##
print("0ne device")
a = np.load(WHERE_FROM_LOAD + '/eval_one_infer_energy.npy')
one_inference_latency = np.array(a[0])
one_inference_latency_std = np.array(a[1])
print(one_inference_latency)
print(one_inference_latency_std)
a = np.load(WHERE_FROM_LOAD + '/eval_one_acc_energy.npy')
one_accuracy = np.array(a[0])
one_accuracy_std = np.array(a[1])
print(one_accuracy)
print(one_accuracy_std)
a = np.load(WHERE_FROM_LOAD + '/eval_one_energy_energy.npy')
one_energy_consumption = np.array(a[0])
one_energy_consumption_std = np.array(a[1])
print(one_energy_consumption)
print(one_energy_consumption_std)
print()
one_ec = []
one_ec_std = []
for i in range(5):
    maxindex = np.argmax(one_energy_consumption[i])
    one_ec.append(one_energy_consumption[i][maxindex])
for i in range(5):
    maxindex = np.argmax(one_energy_consumption_std[i])
    one_ec_std.append(one_energy_consumption_std[i][maxindex])
# one_ec_1 = []
# for i in range(5):
#     one_ec_1.append(one_energy_consumption[i][0])
print(one_ec)

##
print('min intermediate data')
a = np.load(WHERE_FROM_LOAD + '/eval_min_infer_energy.npy')
min_inference_latency = np.array(a[0])
min_inference_latency_std = np.array(a[1])
print(min_inference_latency)
a = np.load(WHERE_FROM_LOAD + '/eval_min_acc_energy.npy')
min_accuracy = np.array(a[0])
min_accuracy_std = np.array(a[1])
print(min_accuracy)
a = np.load(WHERE_FROM_LOAD + '/eval_min_energy_energy.npy')
min_energy_consumption = np.array(a[0])
min_energy_consumption_std = np.array(a[1])
print(min_energy_consumption)
print()
min_ec = []
min_ec_std = []
for i in range(5):
    maxindex = np.argmax(min_energy_consumption[i])
    min_ec.append(min_energy_consumption[i][maxindex])
for i in range(5):
    maxindex = np.argmax(min_energy_consumption_std[i])
    min_ec_std.append(min_energy_consumption_std[i][maxindex])
# min_ec_1 = []
# for i in range(5):
#     min_ec_1.append(min_energy_consumption[i][0])
print(min_ec)

##
print('best computing power')
a = np.load(WHERE_FROM_LOAD + '/eval_comb_infer_energy.npy')
comb_inference_latency = np.array(a[0])
comb_inference_latency_std = np.array(a[1])
print(comb_inference_latency)
a = np.load(WHERE_FROM_LOAD + '/eval_comb_acc_energy.npy')
comb_accuracy = np.array(a[0])
comb_accuracy_std = np.array(a[1])
print(comb_accuracy)
a = np.load(WHERE_FROM_LOAD + '/eval_comb_energy_energy.npy')
comb_energy_consumption = np.array(a[0])
comb_energy_consumption_std = np.array(a[1])
print(comb_energy_consumption)
print()
comb_ec = []
comb_ec_std = []
for i in range(5):
    maxindex = np.argmax(comb_energy_consumption[i])
    comb_ec.append(comb_energy_consumption[i][maxindex])
for i in range(5):
    maxindex = np.argmax(comb_energy_consumption_std[i])
    comb_ec_std.append(comb_energy_consumption_std[i][maxindex])
# comb_ec_1 = []
# for i in range(5):
#     comb_ec_1.append(comb_energy_consumption[i][0])
print(comb_ec)

##
print('low BER')
a = np.load(WHERE_FROM_LOAD + '/eval_lowB_infer_energy.npy')
lowB_inference_latency = np.array(a[0])
lowB_inference_latency_std = np.array(a[1])
print(lowB_inference_latency)
a = np.load(WHERE_FROM_LOAD + '/eval_lowB_acc_energy.npy')
lowB_accuracy = np.array(a[0])
lowB_accuracy_std = np.array(a[1])
print(lowB_accuracy)
a = np.load(WHERE_FROM_LOAD + '/eval_lowB_energy_energy.npy')
lowB_energy_consumption = np.array(a[0])
lowB_energy_consumption_std = np.array(a[1])
print(lowB_energy_consumption)
print()
lowB_ec = []
lowB_ec_std = []
for i in range(5):
    maxindex = np.argmax(lowB_energy_consumption[i])
    lowB_ec.append(lowB_energy_consumption[i][maxindex])
for i in range(5):
    maxindex = np.argmax(lowB_energy_consumption_std[i])
    lowB_ec_std.append(lowB_energy_consumption_std[i][maxindex])
# lowB_ec_1 = []
# for i in range(5):
#     lowB_ec_1.append(lowB_energy_consumption[i][0])
print(lowB_ec)

##
x = np.arange(10, 60, 10)
x = [300, 400, 500, 600, 700]
# x = np.arange(0, 50, 10)
capsize = 3  # Size of the horizontal caps
capthick = 1.6  # Thickness of the horizontal caps
elinewidth = 1  # Thickness of the vertical error bar
markersize = 6
# e-com 141
# inference latency
# plt.title('inference latency by average BER')
plt.grid()
# plt.plot(x, rand_inference_latency, '.-', x, one_inference_latency, '^-', x, min_inference_latency, 'd-', x, comb_inference_latency, '*-', x, lowB_inference_latency, 'x-')
# plt.plot(x, rand_inference_latency, linestyle='solid', marker='o', markersize=8)
plt.errorbar(x, burst_inference_latency, burst_inference_latency_std, capsize=capsize, capthick=capthick, elinewidth=elinewidth, linestyle='solid', marker='o', markersize=markersize)
plt.errorbar(x, one_inference_latency, one_inference_latency_std, capsize=capsize, capthick=capthick, elinewidth=elinewidth, linestyle='dashed', marker='^', markersize=markersize)
plt.errorbar(x, min_inference_latency, min_inference_latency_std, capsize=capsize, capthick=capthick, elinewidth=elinewidth, linestyle='dashdot', marker='d', markersize=markersize)
plt.errorbar(x, comb_inference_latency, comb_inference_latency_std, capsize=capsize, capthick=capthick, elinewidth=elinewidth, linestyle='dotted', marker='p', markersize=markersize)
plt.errorbar(x, lowB_inference_latency, lowB_inference_latency_std, capsize=capsize, capthick=capthick, elinewidth=elinewidth, linestyle=(0, (1, 5)), marker='v', markersize=markersize)
plt.legend(['CRMC', 'NON-OFF', 'MIN-INT', 'BEST-COM', 'LOW-PDR'], loc='lower left', bbox_to_anchor=(0, 0.1))
plt.xlabel('load threshold (MFLOPs)')
plt.ylabel('inference latency (sec)')
plt.show()

# accuracy
# plt.title('accuracy by average BER')
plt.grid()
# plt.plot(x, rand_accuracy, '.-', x, one_accuracy, '^-', x, min_accuracy, 'd-', x, comb_accuracy, '*-', x, lowB_accuracy, 'x-')
# plt.plot(x, rand_accuracy, linestyle='solid', marker='o', markersize=8)
plt.errorbar(x, burst_accuracy, burst_accuracy_std, capsize=capsize, capthick=capthick, elinewidth=elinewidth, linestyle='solid', marker='o', markersize=markersize)
plt.errorbar(x, one_accuracy, one_accuracy_std, capsize=capsize, capthick=capthick, elinewidth=elinewidth, linestyle='dashed', marker='^', markersize=markersize)
plt.errorbar(x, min_accuracy, min_accuracy_std, capsize=capsize, capthick=capthick, elinewidth=elinewidth, linestyle='dashdot', marker='d', markersize=markersize)
plt.errorbar(x, comb_accuracy, comb_accuracy_std, capsize=capsize, capthick=capthick, elinewidth=elinewidth, linestyle='dotted', marker='p', markersize=markersize)
plt.errorbar(x, lowB_accuracy, lowB_accuracy_std, capsize=capsize, capthick=capthick, elinewidth=elinewidth, linestyle=(0, (1, 5)), marker='v', markersize=markersize)
plt.legend(['CRMC', 'NON-OFF', 'MIN-INT', 'BEST-COM', 'LOW-PDR'])
plt.xlabel('load threshold (MFLOPs)')
plt.ylabel('accuracy (%)')
plt.show()

# energy consumption
# plt.title('maximum energy consumption by average BER')
plt.grid()
# plt.plot(x, rand_ec, '.-', x, one_ec, '^-', x, min_ec, 'd-', x, comb_ec, '*-', x, lowB_ec, 'x-')
# plt.plot(x, rand_ec, linestyle='solid', marker='o', markersize=8)
plt.errorbar(x, burst_ec, burst_ec_std, capsize=capsize, capthick=capthick, elinewidth=elinewidth, linestyle='solid', marker='o', markersize=markersize)
plt.errorbar(x, one_ec, one_ec_std, capsize=capsize, capthick=capthick, elinewidth=elinewidth, linestyle='dashed', marker='^', markersize=markersize)
plt.errorbar(x, min_ec, min_ec_std, capsize=capsize, capthick=capthick, elinewidth=elinewidth, linestyle='dashdot', marker='d', markersize=markersize)
plt.errorbar(x, comb_ec, comb_ec_std, capsize=capsize, capthick=capthick, elinewidth=elinewidth, linestyle='dotted', marker='p', markersize=markersize)
plt.errorbar(x, lowB_ec, lowB_ec_std, capsize=capsize, capthick=capthick, elinewidth=elinewidth, linestyle=(0, (1, 5)), marker='v', markersize=markersize)
plt.legend(['CRMC', 'NON-OFF', 'MIN-INT', 'BEST-COM', 'LOW-PDR'])
plt.xlabel('load threshold (MFLOPs)')
plt.ylabel('maximum load (MFLOPs)')
plt.show()

# plt.title('energy consumption of device 1 by average BER')
# plt.grid()
# plt.plot(x, rand_ec_1, '.-', x, one_ec_1, '^-', x, min_ec_1, 'd-', x, comb_ec_1, '*-', x, lowB_ec_1, 'x-')
# plt.legend(['BASS', 'one-device', 'min-intermediate', 'best computing', 'Low BER'])
# plt.xlabel('Average BER (%)')
# plt.ylabel('energy consumption')
# plt.show()
#
# plt.grid()
# # plt.plot(x, rand_inference_latency, '.-', x, one_inference_latency, '^-', x, min_inference_latency, 'd-', x, comb_inference_latency, '*-', x, lowB_inference_latency, 'x-')
# # plt.plot(x, rand_inference_latency, linestyle='solid', marker='o', markersize=8)
# plt.plot(x, burst_inference_latency, linestyle='solid', marker='o', markersize=markersize)
# plt.plot(x, one_inference_latency, linestyle='dashed', marker='^', markersize=markersize)
# plt.plot(x, min_inference_latency, linestyle='dashdot', marker='d', markersize=markersize)
# plt.plot(x, comb_inference_latency, linestyle='dotted', marker='p', markersize=markersize)
# plt.plot(x, lowB_inference_latency, linestyle=(0, (1, 5)), marker='v', markersize=markersize)
# plt.legend(['CRMC', 'NON-OFF', 'MIN-INT', 'BEST-COM', 'LOW-PDR'])
# plt.xlabel('load threshold (MFLOPs)')
# plt.ylabel('inference latency (sec)')
# plt.show()
#
# # accuracy
# # plt.title('accuracy by average energy threshold')
# plt.grid()
# # plt.plot(x, rand_accuracy, '.-', x, one_accuracy, '^-', x, min_accuracy, 'd-', x, comb_accuracy, '*-', x, lowB_accuracy, 'x-')
# # plt.plot(x, rand_accuracy, linestyle='solid', marker='o', markersize=8)
# plt.plot(x, burst_accuracy, linestyle='solid', marker='o', markersize=markersize)
# plt.plot(x, one_accuracy, linestyle='dashed', marker='^', markersize=markersize)
# plt.plot(x, min_accuracy, linestyle='dashdot', marker='d', markersize=markersize)
# plt.plot(x, comb_accuracy, linestyle='dotted', marker='p', markersize=markersize)
# plt.plot(x, lowB_accuracy, linestyle=(0, (1, 5)), marker='v', markersize=markersize)
# plt.legend(['CRMC', 'NON-OFF', 'MIN-INT', 'BEST-COM', 'LOW-PDR'], loc='lower right', bbox_to_anchor=(1, 0.1))
# plt.xlabel('load threshold (MFLOPs)')
# plt.ylabel('accuracy (%)')
# plt.show()
#
# # energy consumption
# # plt.title('maximum energy consumption by average energy threshold')
# plt.grid()
# # plt.plot(x, rand_ec, '.-', x, one_ec, '^-', x, min_ec, 'd-', x, comb_ec, '*-', x, lowB_ec, 'x-')
# # plt.plot(x, rand_ec, linestyle='solid', marker='o', markersize=8)
# plt.plot(x, burst_ec, linestyle='solid', marker='o', markersize=markersize)
# plt.plot(x, one_ec, linestyle='dashed', marker='^', markersize=markersize)
# plt.plot(x, min_ec, linestyle='dashdot', marker='d', markersize=markersize)
# plt.plot(x, comb_ec, linestyle='dotted', marker='p', markersize=markersize)
# plt.plot(x, lowB_ec, linestyle=(0, (1, 5)), marker='v', markersize=markersize)
# plt.legend(['CRMC', 'NON-OFF', 'MIN-INT', 'BEST-COM', 'LOW-PDR'], loc='upper right', bbox_to_anchor=(1, 0.9))
# plt.xlabel('load threshold (MFLOPs)')
# plt.ylabel('maximum load (MFLOPs)')
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
#
# # print("BASS")
# # a = np.load('C:/Users/user/PycharmProjects/SplitComputing/eval_rand_infer_energy.npy')
# # rand_inference_latency = np.array(a)
# # print(rand_inference_latency)
# # a = np.load('C:/Users/user/PycharmProjects/SplitComputing/eval_rand_acc_energy.npy')
# # rand_accuracy = np.array(a)
# # print(rand_accuracy)
# # a = np.load('C:/Users/user/PycharmProjects/SplitComputing/eval_rand_energy_energy.npy')
# # rand_energy_consumption = np.array(a)
# # print(rand_energy_consumption)
# # print()
# # rand_ec = []
# # for i in range(5):
# #     maxindex = np.argmax(rand_energy_consumption[i])
# #     rand_ec.append(rand_energy_consumption[i][maxindex]/10000)
# # rand_ec_1 = []
# # for i in range(5):
# #     rand_ec_1.append(rand_energy_consumption[i][0])
#
# WHERE_FROM_LOAD = 'C:/Users/user/PycharmProjects/SplitComputingv2/resnet/eval'
#
# a = np.load(WHERE_FROM_LOAD + '/eval_burst_infer_energy.npy')
# burst_inference_latency = np.array(a)
# print(burst_inference_latency)
# a = np.load(WHERE_FROM_LOAD + '/eval_burst_acc_energy.npy')
# burst_accuracy = np.array(a)
# print(burst_accuracy)
# a = np.load(WHERE_FROM_LOAD + '/eval_burst_energy_energy.npy')
# burst_energy_consumption = np.array(a)
# print(burst_energy_consumption)
# burst_ec = []
# for i in range(5):
#     maxindex = np.argmax(burst_energy_consumption[i])
#     burst_ec.append(burst_energy_consumption[i][maxindex])
# burst_ec_1 = []
# for i in range(5):
#     burst_ec_1.append(burst_energy_consumption[i][0])
# print(burst_ec)
# ##
# print("0ne device")
# a = np.load(WHERE_FROM_LOAD + '/eval_one_infer_energy.npy')
# one_inference_latency = np.array(a)
# print(one_inference_latency)
# a = np.load(WHERE_FROM_LOAD + '/eval_one_acc_energy.npy')
# one_accuracy = np.array(a)
# print(one_accuracy)
# a = np.load(WHERE_FROM_LOAD + '/eval_one_energy_energy.npy')
# one_energy_consumption = np.array(a)
# print(one_energy_consumption)
# print()
# one_ec = []
# for i in range(5):
#     maxindex = np.argmax(one_energy_consumption[i])
#     one_ec.append(one_energy_consumption[i][maxindex])
# one_ec_1 = []
# for i in range(5):
#     one_ec_1.append(one_energy_consumption[i][0])
# print(one_ec)
# ##
# print('min intermediate data')
# a = np.load(WHERE_FROM_LOAD + '/eval_min_infer_energy.npy')
# min_inference_latency = np.array(a)
# print(min_inference_latency)
# a = np.load(WHERE_FROM_LOAD + '/eval_min_acc_energy.npy')
# min_accuracy = np.array(a)
# print(min_accuracy)
# a = np.load(WHERE_FROM_LOAD + '/eval_min_energy_energy.npy')
# min_energy_consumption = np.array(a)
# print(min_energy_consumption)
# print()
# min_ec = []
# for i in range(5):
#     maxindex = np.argmax(min_energy_consumption[i])
#     min_ec.append(min_energy_consumption[i][maxindex])
# min_ec_1 = []
# for i in range(5):
#     min_ec_1.append(min_energy_consumption[i][0])
# print(min_ec)
# ##
# print('best computing power')
# a = np.load(WHERE_FROM_LOAD + '/eval_comb_infer_energy.npy')
# comb_inference_latency = np.array(a)
# print(comb_inference_latency)
# a = np.load(WHERE_FROM_LOAD + '/eval_comb_acc_energy.npy')
# comb_accuracy = np.array(a)
# print(comb_accuracy)
# a = np.load(WHERE_FROM_LOAD + '/eval_comb_energy_energy.npy')
# comb_energy_consumption = np.array(a)
# print(comb_energy_consumption)
# print()
# comb_ec = []
# for i in range(5):
#     maxindex = np.argmax(comb_energy_consumption[i])
#     comb_ec.append(comb_energy_consumption[i][maxindex])
# comb_ec_1 = []
# for i in range(5):
#     comb_ec_1.append(comb_energy_consumption[i][0])
# print(comb_ec)
# ##
# print('low BER')
# a = np.load(WHERE_FROM_LOAD + '/eval_lowB_infer_energy.npy')
# lowB_inference_latency = np.array(a)
# print(lowB_inference_latency)
# a = np.load(WHERE_FROM_LOAD + '/eval_lowB_acc_energy.npy')
# lowB_accuracy = np.array(a)
# print(lowB_accuracy)
# a = np.load(WHERE_FROM_LOAD + '/eval_lowB_energy_energy.npy')
# lowB_energy_consumption = np.array(a)
# print(lowB_energy_consumption)
# print()
# lowB_ec = []
# for i in range(5):
#     maxindex = np.argmax(lowB_energy_consumption[i])
#     lowB_ec.append(lowB_energy_consumption[i][maxindex])
# lowB_ec_1 = []
# for i in range(5):
#     lowB_ec_1.append(lowB_energy_consumption[i][0])
# print(lowB_ec)
# # x = [145, 150, 155, 160, 165]
# # x = [90, 100, 110, 120, 130]
# # x = [100, 105, 110, 115, 120]
# x = [300, 400, 500, 600, 700]
# # inference latency
# # plt.title('inference latency by average energy threshold')
# plt.grid()
# # plt.plot(x, rand_inference_latency, '.-', x, one_inference_latency, '^-', x, min_inference_latency, 'd-', x, comb_inference_latency, '*-', x, lowB_inference_latency, 'x-')
# # plt.plot(x, rand_inference_latency, linestyle='solid', marker='o', markersize=8)
# plt.plot(x, burst_inference_latency, linestyle='solid', marker='o', markersize=8)
# plt.plot(x, one_inference_latency, linestyle='dashed', marker='^', markersize=8)
# plt.plot(x, min_inference_latency, linestyle='dashdot', marker='d', markersize=8)
# plt.plot(x, comb_inference_latency, linestyle='dotted', marker='p', markersize=8)
# plt.plot(x, lowB_inference_latency, linestyle=(0, (1, 5)), marker='v', markersize=8)
# plt.legend(['CRMC', 'NON-OFF', 'MIN-INT', 'BEST-COM', 'LOW-PDR'])
# plt.xlabel('load threshold (MFLOPs)')
# plt.ylabel('inference latency (sec)')
# plt.show()
#
# # accuracy
# # plt.title('accuracy by average energy threshold')
# plt.grid()
# # plt.plot(x, rand_accuracy, '.-', x, one_accuracy, '^-', x, min_accuracy, 'd-', x, comb_accuracy, '*-', x, lowB_accuracy, 'x-')
# # plt.plot(x, rand_accuracy, linestyle='solid', marker='o', markersize=8)
# plt.plot(x, burst_accuracy, linestyle='solid', marker='o', markersize=8)
# plt.plot(x, one_accuracy, linestyle='dashed', marker='^', markersize=8)
# plt.plot(x, min_accuracy, linestyle='dashdot', marker='d', markersize=8)
# plt.plot(x, comb_accuracy, linestyle='dotted', marker='p', markersize=8)
# plt.plot(x, lowB_accuracy, linestyle=(0, (1, 5)), marker='v', markersize=8)
# plt.legend(['CRMC', 'NON-OFF', 'MIN-INT', 'BEST-COM', 'LOW-PDR'], loc='lower right', bbox_to_anchor=(1, 0.1))
# plt.xlabel('load threshold (MFLOPs)')
# plt.ylabel('accuracy (%)')
# plt.show()
#
# # energy consumption
# # plt.title('maximum energy consumption by average energy threshold')
# plt.grid()
# # plt.plot(x, rand_ec, '.-', x, one_ec, '^-', x, min_ec, 'd-', x, comb_ec, '*-', x, lowB_ec, 'x-')
# # plt.plot(x, rand_ec, linestyle='solid', marker='o', markersize=8)
# plt.plot(x, burst_ec, linestyle='solid', marker='o', markersize=8)
# plt.plot(x, one_ec, linestyle='dashed', marker='^', markersize=8)
# plt.plot(x, min_ec, linestyle='dashdot', marker='d', markersize=8)
# plt.plot(x, comb_ec, linestyle='dotted', marker='p', markersize=8)
# plt.plot(x, lowB_ec, linestyle=(0, (1, 5)), marker='v', markersize=8)
# plt.legend(['CRMC', 'NON-OFF', 'MIN-INT', 'BEST-COM', 'LOW-PDR'], loc='upper right', bbox_to_anchor=(1, 0.9))
# plt.xlabel('load threshold (MFLOPs)')
# plt.ylabel('maximum load (MFLOPs)')
# plt.show()
#
# # plt.title('energy consumption of device 1 by average energy threshold')
# # plt.grid()
# # plt.plot(x, rand_ec_1, '.-', x, one_ec_1, '^-', x, min_ec_1, 'd-', x, comb_ec_1, '*-', x, lowB_ec_1, 'x-')
# # plt.legend(['BASS', 'one-device', 'min-intermediate', 'best computing', 'Low BER'])
# # plt.xlabel('Average energy threshold')
# # plt.ylabel('energy consumption')
# # plt.show()
#
#
#
#
