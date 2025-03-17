#
# import torch
# from basicsr.archs.rrdbnet_arch import RRDBNet
# from basicsr.utils.download_util import load_file_from_url
# from realesrgan import RealESRGANer
# import cv2
# import os
# import numpy as np
#
#
# def initialize_realesrgan(model_name='RealESRGAN_x4plus'):
#     """Real-ESRGAN 모델 초기화"""
#     model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
#
#     # 모델 URL 선택
#     model_urls = {
#         'RealESRGAN_x4plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
#         'RealESRGAN_x2plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
#     }
#
#     # 모델 가중치 다운로드
#     model_path = load_file_from_url(model_urls[model_name], model_dir='weights')
#
#     # GPU 사용 가능 여부 확인
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # 업스케일러 초기화
#     upsampler = RealESRGANer(
#         scale=4,
#         model_path=model_path,
#         model=model,
#         tile=0,  # 타일 크기, 0은 전체 이미지 처리
#         tile_pad=10,
#         pre_pad=0,
#         half=False,  # FP16 (half precision)
#         device=device
#     )
#
#     return upsampler
#
#
# def enhance_image(input_path, output_path, upsampler=None):
#     """이미지 초해상화 실행"""
#     try:
#         # 이미지 읽기
#         img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
#         if img is None:
#             raise ValueError(f"이미지를 읽을 수 없습니다: {input_path}")
#
#         # BGR to RGB
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#         # 초해상화 실행
#         output, _ = upsampler.enhance(img)
#
#         # RGB to BGR
#         output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
#
#         # 결과 저장
#         cv2.imwrite(output_path, output)
#
#         print(f"초해상화 완료: {output_path}")
#         return True
#
#     except Exception as e:
#         print(f"에러 발생: {str(e)}")
#         return False
#
#
# def main():
#     # 필요한 디렉토리 생성
#     os.makedirs('weights', exist_ok=True)
#     os.makedirs('results', exist_ok=True)
#
#     # 입출력 경로 설정
#     input_path = "bug.jpeg"  # 입력 이미지 경로
#     output_path = "enhanced.jpg"  # 출력 이미지 경로
#
#     try:
#         # 모델 초기화
#         print("모델 초기화 중...")
#         upsampler = initialize_realesrgan()
#
#         # 이미지 처리
#         print("이미지 처리 중...")
#         success = enhance_image(input_path, output_path, upsampler)
#
#         if success:
#             print("처리가 완료되었습니다!")
#     except Exception as e:
#         print(f"처리 중 에러 발생: {str(e)}")
#
#
# if __name__ == '__main__':
#     main()





import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from basicsr.utils.download_util import load_file_from_url
import torch
import cv2

# 경로 설정
base_dir = "TinyImageNet"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "val")

# 이미지 크기 설정
image_size = (64, 64)

# Real-ESRGAN 초기화 함수
def initialize_realesrgan(model_name='RealESRGAN_x4plus'):
    """Real-ESRGAN 모델 초기화"""
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model_urls = {
        'RealESRGAN_x4plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    }
    model_path = load_file_from_url(model_urls[model_name], model_dir='weights')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=device
    )
    return upsampler

# 초해상도 적용 함수
def apply_super_resolution(image_path, upsampler):
    """이미지에 초해상도를 적용"""
    img = np.array(Image.open(image_path).convert("RGB"))  # PIL 이미지를 NumPy 배열로 변환
    output, _ = upsampler.enhance(img)  # Real-ESRGAN 초해상도 처리
    return Image.fromarray(output)  # NumPy 배열을 PIL 이미지로 변환

# 데이터 로드 및 초해상화 적용
def load_combined_data_with_sr(train_dir, test_dir, image_size, upsampler):
    X_data = []
    y_data = []
    label_names = []

    # Train 데이터 로드
    for label_folder in os.listdir(train_dir):
        label_path = os.path.join(train_dir, label_folder, "images")
        if not os.path.isdir(label_path):
            continue

        label_names.append(label_folder)
        for image_file in os.listdir(label_path):
            if image_file.endswith(".JPEG"):
                image_path = os.path.join(label_path, image_file)
                sr_image = apply_super_resolution(image_path, upsampler)  # 초해상화 적용
                resized_image = sr_image.resize(image_size)  # 최종 크기로 리사이즈
                X_data.append(np.array(resized_image))
                y_data.append(label_folder)

    # Test 데이터 로드
    test_annotations_path = os.path.join(test_dir, "val_annotations.txt")
    test_images_path = os.path.join(test_dir, "images")

    # val_annotations.txt 읽기
    with open(test_annotations_path, "r") as f:
        test_annotations = f.readlines()

    test_label_map = {}
    for line in test_annotations:
        parts = line.strip().split("\t")
        test_label_map[parts[0]] = parts[1]

    # Val 데이터 로드
    for image_file in os.listdir(test_images_path):
        if image_file.endswith(".JPEG"):
            image_path = os.path.join(test_images_path, image_file)
            label = test_label_map[image_file]
            sr_image = apply_super_resolution(image_path, upsampler)  # 초해상화 적용
            resized_image = sr_image.resize(image_size)  # 최종 크기로 리사이즈
            X_data.append(np.array(resized_image))
            y_data.append(label)

    return np.array(X_data), np.array(y_data), label_names

# 모델 초기화
print("Real-ESRGAN 모델 초기화 중...")
upsampler = initialize_realesrgan()

# Combined 데이터 로드
X_data, y_data, label_names = load_combined_data_with_sr(train_dir, test_dir, image_size, upsampler)

# Label Encoding
label_encoder = LabelEncoder()
label_encoder.fit(label_names)

y_encoded = label_encoder.transform(y_data)

# 데이터 셋 나누기 (80000:20000:10000)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_data, y_encoded, train_size=80000, stratify=y_encoded, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, train_size=20000, stratify=y_temp, random_state=42
)

# 데이터 전처리 : 스케일링
X_train_s = X_train.astype('float32') / 255.0
X_val_s = X_val.astype('float32') / 255.0
X_test_s = X_test.astype('float32') / 255.0

# 데이터 저장 경로
save_dir = "TinyImageNet/Processed_data"
os.makedirs(save_dir, exist_ok=True)

# 데이터 저장
def save_data(save_dir, X_train_s, y_train, X_val_s, y_val, X_test_s, y_test):
    np.save(os.path.join(save_dir, "X_train_s.npy"), X_train_s)
    np.save(os.path.join(save_dir, "y_train_encoded.npy"), y_train)
    np.save(os.path.join(save_dir, "X_val_s.npy"), X_val_s)
    np.save(os.path.join(save_dir, "y_val_encoded.npy"), y_val)
    np.save(os.path.join(save_dir, "X_test_s.npy"), X_test_s)
    np.save(os.path.join(save_dir, "y_test_encoded.npy"), y_test)
    print("Data saved successfully!")

save_data(save_dir, X_train_s, y_train, X_val_s, y_val, X_test_s, y_test)

# 데이터 확인
print(f"X_train_s shape: {X_train_s.shape}, y_train shape: {y_train.shape}")
print(f"X_val_s shape: {X_val_s.shape}, y_val shape: {y_val.shape}")
print(f"X_test_s shape: {X_test_s.shape}, y_test shape: {y_test.shape}")



# import pandas as pd
#
# def print_label_distribution(y_train, y_val, y_test, label_encoder):
#     # 각 데이터셋의 라벨 분포 계산
#     train_counts = pd.Series(y_train).value_counts().sort_index()
#     val_counts = pd.Series(y_val).value_counts().sort_index()
#     test_counts = pd.Series(y_test).value_counts().sort_index()
#
#     # 라벨 인코더로 숫자 라벨을 문자열 라벨로 변환
#     label_names = label_encoder.inverse_transform(train_counts.index)
#
#     # DataFrame 생성
#     distribution_df = pd.DataFrame({
#         "Label": label_names,
#         "Train Count": train_counts.values,
#         "Validation Count": val_counts.values,
#         "Test Count": test_counts.values
#     })
#
#     # 결과 출력
#     print(distribution_df.sort_values("Label"))
#
# # 호출
# print_label_distribution(y_train, y_val, y_test, label_encoder)

import matplotlib.pyplot as plt

def plot_sample_images(X_data, y_data, label_encoder, num_samples=5):
    """
    데이터에서 이미지를 샘플링하여 plot
    """
    # 5개 샘플링
    sampled_indices = np.random.choice(len(X_data), num_samples, replace=False)
    sampled_images = X_data[sampled_indices]
    sampled_labels = y_data[sampled_indices]

    # Plot 설정
    plt.figure(figsize=(15, 5))
    for i, (image, label) in enumerate(zip(sampled_images, sampled_labels)):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(image)
        plt.title(f"Label: {label_encoder.inverse_transform([label])[0]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# 샘플링 후 시각화
plot_sample_images(X_train, y_train, label_encoder)



# import numpy as np
# import os
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
# from PIL import Image
#
# # 경로 설정
# base_dir = "TinyImageNet"
# train_dir = os.path.join(base_dir, "train")
# test_dir = os.path.join(base_dir, "val")
#
# # 이미지 크기 설정
# image_size = (64, 64)
#
# def load_combined_data(train_dir, test_dir, image_size):
#     X_data = []
#     y_data = []
#     label_names = []
#
#     # Train 데이터 로드
#     for label_folder in os.listdir(train_dir):
#         label_path = os.path.join(train_dir, label_folder, "images")
#         if not os.path.isdir(label_path):
#             continue
#
#         label_names.append(label_folder)
#         for image_file in os.listdir(label_path):
#             if image_file.endswith(".JPEG"):
#                 image_path = os.path.join(label_path, image_file)
#                 image = Image.open(image_path).resize(image_size).convert("RGB")
#                 X_data.append(np.array(image))
#                 y_data.append(label_folder)
#
#     # Test 데이터 로드
#     test_annotations_path = os.path.join(test_dir, "val_annotations.txt")
#     test_images_path = os.path.join(test_dir, "images")
#
#     # val_annotations.txt 읽기
#     with open(test_annotations_path, "r") as f:
#         test_annotations = f.readlines()
#
#     test_label_map = {}
#     for line in test_annotations:
#         parts = line.strip().split("\t")
#         test_label_map[parts[0]] = parts[1]
#
#     # Val 데이터 로드
#     for image_file in os.listdir(test_images_path):
#         if image_file.endswith(".JPEG"):
#             image_path = os.path.join(test_images_path, image_file)
#             label = test_label_map[image_file]
#             image = Image.open(image_path).resize(image_size).convert("RGB")
#             X_data.append(np.array(image))
#             y_data.append(label)
#
#     return np.array(X_data), np.array(y_data), label_names
#
# # Combined 데이터 로드
# X_data, y_data, label_names = load_combined_data(train_dir, test_dir, image_size)
#
# # Label Encoding
# label_encoder = LabelEncoder()
# label_encoder.fit(label_names)
#
# y_encoded = label_encoder.transform(y_data)
#
# # 데이터 셋 나누기 (80000:20000:10000)
# X_train, X_temp, y_train, y_temp = train_test_split(
#     X_data, y_encoded, train_size=80000, stratify=y_encoded, random_state=42
# )
# X_val, X_test, y_val, y_test = train_test_split(
#     X_temp, y_temp, train_size=20000, stratify=y_temp, random_state=42
# )
#
# # 데이터 전처리 : 스케일링
# X_train_s = X_train.astype('float32') / 255.0
# X_val_s = X_val.astype('float32') / 255.0
# X_test_s = X_test.astype('float32') / 255.0
#
# # 데이터 저장 경로
# save_dir = "TinyImageNet/Processed_data"
# os.makedirs(save_dir, exist_ok=True)
#
# # 데이터 저장
# def save_data(save_dir, X_train_s, y_train, X_val_s, y_val, X_test_s, y_test):
#     np.save(os.path.join(save_dir, "X_train_s.npy"), X_train_s)
#     np.save(os.path.join(save_dir, "y_train_encoded.npy"), y_train)
#     np.save(os.path.join(save_dir, "X_val_s.npy"), X_val_s)
#     np.save(os.path.join(save_dir, "y_val_encoded.npy"), y_val)
#     np.save(os.path.join(save_dir, "X_test_s.npy"), X_test_s)
#     np.save(os.path.join(save_dir, "y_test_encoded.npy"), y_test)
#     print("Data saved successfully!")
#
# save_data(save_dir, X_train_s, y_train, X_val_s, y_val, X_test_s, y_test)
#
# # 데이터 확인
# print(f"X_train_s shape: {X_train_s.shape}, y_train shape: {y_train.shape}")
# print(f"X_val_s shape: {X_val_s.shape}, y_val shape: {y_val.shape}")
# print(f"X_test_s shape: {X_test_s.shape}, y_test shape: {y_test.shape}")



