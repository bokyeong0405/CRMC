def set_ss_sb (short_path, end, D_BER, SS):
    # accuracy 계산을 위해 Ss와 SB 구하는 코드
    sp = []  # split point set
    b = []  # BER set

    if len(short_path) != 1:
        for i in range(len(short_path) - 1):
            tmp = end[short_path[i] % SS]
            sp.append(tmp)  # split 지점 추가
            if (short_path[i] // SS) == 0:  # 각 지점의 BER 추가
                if (short_path[i + 1] // SS) == 1:
                    b.append(D_BER[0])
                elif (short_path[i + 1] // SS) == 2:
                    b.append(D_BER[2])
            elif (short_path[i] // SS) == 1:
                if (short_path[i + 1] // SS) == 2:
                    b.append(D_BER[1])
                elif (short_path[i + 1] // SS) == 0:
                    b.append(D_BER[0])
            elif (short_path[i] // SS) == 2:
                if (short_path[i + 1] // SS) == 1:
                    b.append(D_BER[1])
                elif (short_path[i + 1] // SS) == 0:
                    b.append(D_BER[2])
    elif len(short_path) == 1:
        sp.append(34)
        b.append(0)
    # print("set_ss_sb_resnet.py")
    # print("sp:", sp)
    # print("b:", b)

    return sp, b


# end = [0, 2, 7, 14, 18, 31, 34, 7, 14, 18,
#        31, 34, 14, 18, 31, 34, 18, 31, 34, 31, 34, 34]  # 노드에 할당된 끝 레이어
# D_BER = [10, 12, 8]
# SS = 22
# short_path = [0, 47, 40]
#
# sp, b = set_ss_sb (short_path, end, D_BER, SS)
# print(sp)
# print(b)
#
# from ResNet_predict_accuracy import *  # resnet
#
# accuracy = predict_accuracy_res(sp, b)
# print(accuracy)