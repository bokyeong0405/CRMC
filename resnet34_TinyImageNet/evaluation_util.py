import numpy as np

def evaluation_for_SC (output_results, test_label):
    # 예측 클래스 인덱스
    prediction_digit = np.argmax(output_results, axis=1)
    # print("pred_digit")
    # print(prediction_digit)
    # 라벨은 이미 정수 형태
    label_digit = test_label

    accuracy = (prediction_digit == label_digit).mean()
    return accuracy