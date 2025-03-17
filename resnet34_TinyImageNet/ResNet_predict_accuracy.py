import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import scipy.linalg

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import scipy.linalg
import os
import matplotlib.pyplot as plt

## accuracy 예측

mapping_dict = {
    2: 0,
    7: 1,
    14: 2,
    18: 3,
    31: 4,
    34: 7,
    0: 7
}

def map_values(input_list):
    mapped_list = []
    for value in input_list:
        if value in mapping_dict:
            mapped_value = mapping_dict[value]
            mapped_list.append(mapped_value)
        else:
            raise ValueError(f"매핑되지 않은 입력 값: {value}")
    return mapped_list

def predict_accuracy_res(split_points, pdr_values, data_dir='C:/Users/user/PycharmProjects/SplitComputingv2/resnet/acc/'):

    if not isinstance(split_points, (list, tuple)):
        raise ValueError("split_points should be a list or tuple of split points.")
    # print("split_points", split_points)
    # print("pdr_values:", pdr_values)

    if len(split_points) >= 2 and split_points[0] == 0:
        split_points = split_points[1:]
        pdr_values = pdr_values[1:]

    split_points = map_values(split_points)

    if not isinstance(pdr_values, (list, tuple)):
        pdr_values = [pdr_values]

    # print("split_points", split_points)
    # print("pdr_values", pdr_values)

    if split_points[0] == 7:
        predicted_accuracy = 0.5052
        return predicted_accuracy

    num_splits = len(split_points)



    if num_splits == 1:

        # if not isinstance(pdr_values, (list, tuple)) or len(pdr_values) != 1:
        #     raise ValueError("For single split points, pdr_values should be a list or tuple with one float.")

        # 단일 분할 지점: 1D 회귀
        split_name = '_'.join(map(str, split_points))
        file_path = os.path.join(data_dir, f"acc_{split_name}.npy")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")

        # 데이터 로드
        data_loaded = np.load(file_path)  # 예상 형태: (N, 2) 열 0=PDR, 열 1=Accuracy
        x = np.arange(10, 60, 10)  # PDR: 10, 20, 30, 40, 50
        y = data_loaded[:, 1]  # Accuracy

        # 다항 회귀 모델 학습 (degree=4)
        poly = PolynomialFeatures(degree=4, include_bias=False)
        poly_features = poly.fit_transform(x.reshape(-1, 1))
        model = LinearRegression()
        model.fit(poly_features, y)

        # Accuracy 예측
        pdr_input = np.array([[pdr_values[0]]])
        pdr_poly = poly.transform(pdr_input)
        predicted_accuracy = model.predict(pdr_poly)[0]

        return predicted_accuracy

    elif num_splits == 2:

        if not isinstance(pdr_values, (list, tuple)) or len(pdr_values) != 2:
            raise ValueError("For two split points, pdr_values should be a list or tuple with two floats.")

        # 두 개의 분할 지점: 2D 회귀
        split_name = '_'.join(map(str, split_points))
        file_path = os.path.join(data_dir, f"acc_{split_name}.npy")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")

        # 데이터 로드
        data_loaded = np.load(file_path)  # 예상 형태: (len(er_x), len(er_y))
        x = np.arange(10, 60, 10)  # 첫 번째 분할 지점의 PDR: 10, 20, 30, 40, 50
        y = np.arange(10, 60, 10)  # 두 번째 분할 지점의 PDR: 10, 20, 30, 40, 50
        x, y = np.meshgrid(x, y)
        z = data_loaded

        x = np.array(x).flatten().tolist()
        y = np.array(y).flatten().tolist()
        z = np.array(z).flatten().tolist()

        data = np.stack([x, y, z], 1)
        A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:, :2] ** 2]
        C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])  # coefficients

        pdr_1 = pdr_values[0]
        pdr_2 = pdr_values[1]
        predicted_accuracy = np.dot([1, pdr_1, pdr_2, pdr_1 * pdr_2, pdr_1 ** 2, pdr_2 ** 2], C)

        return predicted_accuracy

    else:
        raise ValueError("Only supports 1 or 2 split points.")
#
#
# mapping_dict = {
#     2: 0,
#     7: 1,
#     14: 2,
#     18: 3,
#     31: 4,
#     34: 7,
#     0: 7
# }
#
# PDR = [14]
# splitting_point = [31]
# print(predict_accuracy_res(splitting_point, PDR))
#
# PDR = [20, 10]
# splitting_point = [18, 31]
# print(predict_accuracy_res(splitting_point, PDR))
#

# data_loaded = np.load('C:/Users/user/PycharmProjects/SplitComputingv2/resnet/acc_4.npy')
# x = np.arange(10, 60, 10)
# y = data_loaded[:, 1]
#
#
# # y = b0 + b1*x + b2*x^2 + b3*x^3 + b4*x^4
# poly = PolynomialFeatures(degree=4, include_bias=False)
#
# poly_features = poly.fit_transform(x.reshape(-1, 1))
# poly_reg_model = LinearRegression()
# poly_reg_model.fit(poly_features, y)
# y_predicted = poly_reg_model.predict(poly_features)
#
# x_new = np.array([[15]])
# x_new_poly = poly.transform(x_new)
# print("x_new_poly", x_new_poly)
# acc = poly_reg_model.predict(x_new_poly)[0]
#
# print(acc)
# #
# # plt.figure(figsize=(10,6))
# # plt.title('layer 4 prediction model')
# # plt.scatter(x, y, label='Data')
# # plt.xlabel('BER')
# # plt.ylabel('Predicted Accuracy')
# # plt.plot(x, y_predicted, c='red', label='Fitted polynomial')
# # plt.legend()
# # plt.show()
#
# data_loaded = np.load('C:/Users/user/PycharmProjects/SplitComputingv2/resnet/acc_3_4.npy')
# x = np.arange(10, 60, 10)
# y = np.arange(10, 60, 10)
# x, y = np.meshgrid(x, y)
# z = data_loaded
#
# x = np.array(x).flatten().tolist()
# y = np.array(y).flatten().tolist()
# z = np.array(z).flatten().tolist()
#
# x = np.array(x)
# y = np.array(y)
# z = np.array(z)
#
# # regular grid covering the domain of the data
# X, Y = np.meshgrid(np.arange(10, 60, 10), np.arange(10, 60, 10))
# XX = X.flatten()
# YY = Y.flatten()
#
# data = np.stack([x, y, z], 1)
# f = 2  # 1: linear, 2: quadratic
# A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:, :2] ** 2]
# C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])  # coefficients
# # evaluate it on a grid
# Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX ** 2, YY ** 2], C).reshape(X.shape)
#
# a = 10
# b = 10
# acc = np.dot([1, a, b, a * b, a ** 2, b ** 2], C)
# print(acc)
#
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# # # 2차원 그리드 데이터를 사용해야 합니다. 여기서 Z는 위에서 계산된 결과를 사용합니다.
# # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
# # ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', s=50)
# # plt.xlabel('first PDR (%)')
# # plt.ylabel('second PDR (%)')
# # ax.set_zlabel('accuracy')
# # ax.axis('tight')
# # plt.show()
