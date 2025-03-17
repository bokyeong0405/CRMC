import numpy as np
import pandas as pd
import os
import tensorflow as tf
# 모델 생성
from tensorflow.keras import layers, models
from evaluation_util import *
from sklearn.model_selection import KFold
from tqdm import tqdm
import matplotlib.pyplot as plt

# 데이터 불러오기
def load_data(save_dir):
    X_train_s = np.load(os.path.join(save_dir, "X_train_s.npy"))
    y_train_encoded = np.load(os.path.join(save_dir, "y_train_encoded.npy"))
    X_val_s = np.load(os.path.join(save_dir, "X_val_s.npy"))
    y_val_encoded = np.load(os.path.join(save_dir, "y_val_encoded.npy"))
    X_test_s = np.load(os.path.join(save_dir, "X_test_s.npy"))
    y_test_encoded = np.load(os.path.join(save_dir, "y_test_encoded.npy"))
    print("Data loaded successfully!")
    return X_train_s, y_train_encoded, X_val_s, y_val_encoded, X_test_s, y_test_encoded

# 정규화된 데이터 불러오기
save_dir = "TinyImageNet/Processed_data"
X_train_s, y_train_encoded, X_val_s, y_val_encoded, X_test_s, y_test_encoded = load_data(save_dir)

# 데이터 확인
print(f"X_train_s shape: {X_train_s.shape}, y_train_encoded shape: {y_train_encoded.shape}")
print(f"X_val_s shape: {X_val_s.shape}, y_val_encoded shape: {y_val_encoded.shape}")
print(f"X_test_s shape: {X_test_s.shape}, y_test_encoded shape: {y_test_encoded.shape}")

# 데이터 증강 정의 (수평 뒤집기, 회전, 확대/축소)
horizontal_flip = tf.keras.Sequential([layers.RandomFlip("horizontal")])  # 수평 뒤집기
rotation = tf.keras.Sequential([layers.RandomRotation(0.1)])  # ±10% 각도 회전
zoom = tf.keras.Sequential([layers.RandomZoom(0.2)])  # 20% 확대/축소

# 데이터셋에 증강
def preprocess_data(X, y, training=False):
    if training:
        original_dataset = tf.data.Dataset.from_tensor_slices((X, y))  # 원본 데이터
        flipped_dataset = original_dataset.map(
            lambda x, y: (horizontal_flip(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )  # 수평 뒤집기 적용
        rotated_dataset = original_dataset.map(
            lambda x, y: (rotation(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )  # 회전 적용
        zoomed_dataset = original_dataset.map(
            lambda x, y: (zoom(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )  # 확대/축소 적용

        # 원본 + 증강 데이터 합치기
        dataset = original_dataset.concatenate(flipped_dataset)
        dataset = dataset.concatenate(rotated_dataset)
        dataset = dataset.concatenate(zoomed_dataset)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
    return dataset


# batch size 정의
BATCH_SIZE = 64
EPOCHs = 30

train_ds = (
    preprocess_data(X_train_s, y_train_encoded, training=True)
    .shuffle(buffer_size=160000)
    .batch(BATCH_SIZE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

val_ds = (
    preprocess_data(X_val_s, y_val_encoded, training=False)
    .batch(BATCH_SIZE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

test_ds = (
    preprocess_data(X_test_s, y_test_encoded, training=False)
    .batch(BATCH_SIZE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# # train_ds의 전체 크기 확인
# total_samples = sum(1 for _ in train_ds.unbatch())  # unbatch로 각 샘플 순회하여 총 개수 계산
# print(f"Total samples in train_ds: {total_samples}")
#
# exit()

NUM_SAMPLES = 320000

# decay steps 계산
decay_steps = int(0.8 * EPOCHs * (NUM_SAMPLES / BATCH_SIZE))

# 옵티마이저, 로스 정의
# lr = 1e-4
# optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
# 학습률 스케줄 설정
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=decay_steps,
    decay_rate=0.96,
    staircase=False  # 계단식 감소
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # label 정수형

# # 학습률 변화 시각화
# steps = range(0, EPOCHs * int(NUM_SAMPLES / BATCH_SIZE))
# lrs = [lr_schedule(step).numpy() for step in steps]
#
# plt.plot(steps, lrs)
# plt.xlabel("Training Steps")
# plt.ylabel("Learning Rate")
# plt.title("Exponential Decay Schedule")
# plt.show()
#
# exit()

# # SGD with Momentum
# optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
#
# # Loss 정의
# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

class BasicBlock(tf.keras.layers.Layer):
 # PyTorch 코드에서 mul = 1
 mul = 1

 def __init__(self, in_planes, out_planes, stride=1):
  super(BasicBlock, self).__init__()

  # 첫 번째 Conv + BN + ReLU
  self.conv1 = layers.Conv2D(out_planes, kernel_size=3, strides=stride, padding='same', use_bias=False)
  self.bn1 = layers.BatchNormalization()

  # 두 번째 Conv + BN
  self.conv2 = layers.Conv2D(out_planes, kernel_size=3, strides=1, padding='same', use_bias=False)
  self.bn2 = layers.BatchNormalization()

  self.shortcut = models.Sequential()
  if stride != 1 or in_planes != out_planes:
   self.shortcut.add(layers.Conv2D(out_planes, kernel_size=1, strides=stride, use_bias=False))
   self.shortcut.add(layers.BatchNormalization())

 def call(self, x, training=False):
  out = self.conv1(x)
  out = self.bn1(out, training=training)
  out = tf.nn.relu(out)

  out = self.conv2(out)
  out = self.bn2(out, training=training)

  shortcut = self.shortcut(x, training=training)
  out = tf.nn.relu(out + shortcut)
  return out

class ResNet(tf.keras.Model):
 def __init__(self, block, num_blocks, num_classes=200):
  super(ResNet, self).__init__()
  self.in_planes = 64

  # Conv1 + BN + ReLU + MaxPool
  self.conv1 = layers.Conv2D(self.in_planes, kernel_size=3, strides=1, padding='same')
  self.bn1 = layers.BatchNormalization()
  self.relu = layers.ReLU()
  # self.maxpool1 = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')

  # make_layer를 통해 layer1~layer4 생성
  self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
  self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
  self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
  self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

  self.avgpool = layers.GlobalAveragePooling2D()
  # self.dropout = layers.Dropout(0.25)
  self.fc = layers.Dense(num_classes)

 def _make_layer(self, block, out_planes, num_blocks, stride):
  strides = [stride] + [1] * (num_blocks - 1)
  layer_list = []
  for s in strides:
   layer_list.append(block(self.in_planes, out_planes, s))
   self.in_planes = block.mul * out_planes
  return models.Sequential(layer_list)

 def call(self, x, training=False):
  # stem
  x = self.conv1(x)
  x = self.bn1(x, training=training)
  x = self.relu(x)
  # x = self.maxpool1(x)

  # 4개의 residual stage
  x = self.layer1(x, training=training)
  x = self.layer2(x, training=training)
  x = self.layer3(x, training=training)
  x = self.layer4(x, training=training)

  x = self.avgpool(x)  # N x 512
  # x = self.dropout(x)
  x = self.fc(x)  # N x num_classes
  return x

def ResNet34(num_classes=200):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    # return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

device = "/gpu:0" if len(tf.config.list_physical_devices('GPU')) > 0 else "/cpu:0"
#
# # 모델 생성
# model = ResNet34(num_classes=200)
# model.build((None, 64, 64, 3))
# model.summary()
#
#
#
# # Metrics for loss and accuracy
# train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
# val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
# val_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
#
#
#
# with tf.device(device):
#     for epoch in range(EPOCHs):
#         # Reset metrics at the start of each epoch
#         train_loss_metric.reset_states()
#         train_accuracy_metric.reset_states()
#         val_loss_metric.reset_states()
#         val_accuracy_metric.reset_states()
#
#         # Training Loop
#         train_iterator = tqdm(train_ds, desc=f"Training epoch {epoch + 1}")
#         for data, label in train_iterator:
#             with tf.GradientTape() as tape:
#                 preds = model(data, training=True)
#                 loss = loss_fn(label, preds)
#             grads = tape.gradient(loss, model.trainable_variables)
#             optimizer.apply_gradients(zip(grads, model.trainable_variables))
#
#             # Update metrics
#             train_loss_metric.update_state(loss)
#             train_accuracy_metric.update_state(label, preds)
#
#             # Log progress
#             train_iterator.set_postfix(
#                 loss=train_loss_metric.result().numpy(),
#                 accuracy=train_accuracy_metric.result().numpy()
#             )
#
#         # Validation Loop
#         val_iterator = tqdm(val_ds, desc=f"Validation epoch {epoch + 1}")
#         for data, label in val_iterator:
#             preds = model(data, training=False)
#             loss = loss_fn(label, preds)
#
#             # Update metrics
#             val_loss_metric.update_state(loss)
#             val_accuracy_metric.update_state(label, preds)
#
#             # Log progress
#             val_iterator.set_postfix(
#                 val_loss=val_loss_metric.result().numpy(),
#                 val_accuracy=val_accuracy_metric.result().numpy()
#             )
#
#         # Epoch summary
#         print(
#             f"Epoch {epoch + 1}: "
#             f"Train Loss = {train_loss_metric.result().numpy():.4f}, "
#             f"Train Accuracy = {train_accuracy_metric.result().numpy():.4f}, "
#             f"Val Loss = {val_loss_metric.result().numpy():.4f}, "
#             f"Val Accuracy = {val_accuracy_metric.result().numpy():.4f}"
#         )
#
# # 가중치 저장
# model.save_weights("ResNet_ImageNet_tf")

## 모델 평가

new_model = ResNet34(num_classes=200)
new_model.build((None, 64, 64, 3))  # 파라미터 초기화
new_model.load_weights("ResNet_ImageNet_tf")  # 가중치 로드

test_ds = (
    tf.data.Dataset.from_tensor_slices((X_test_s, y_test_encoded))
    .batch(BATCH_SIZE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# 테스트 정확도 계산
test_accuracy = 0.0
count = 0
for data, label in test_ds:
    preds = new_model(data, training=False)
    preds = preds.numpy()  # 텐서에서 numpy로 변환
    # print("######## preds #########")
    # print(preds)
    label = label.numpy()  # 라벨도 numpy 배열로 변환
    # print("######## label #########")
    # print(label)
    batch_acc = evaluation_for_SC(preds, label)
    test_accuracy += batch_acc
    count += 1
test_accuracy = test_accuracy / count
print(f"Test Accuracy after loading weights: {test_accuracy:.4f}")

exit()

def split_resnet(model, splitting_points, input_shape=(64, 64, 3)):
    """
    splitting_points: 정수 리스트. [0, 2]와 같이 주어지면 해당 인덱스 위치 기준으로 모델을 세 부분으로 나눔.
    인덱스와 실제 레이어 인덱스 매핑:
      0 -> after layer index 3 (max_pooling2d 뒤)
      1 -> after layer index 4 (sequential_3 뒤)
      2 -> after layer index 5 (sequential_8 뒤)
      3 -> after layer index 6 (sequential_15 뒤)
      4 -> after layer index 7 (sequential_19 뒤)
    """

    # 레이어 인덱스 매핑
    layer_map = {0: 3, 1: 4, 2: 5, 3: 6, 4: 7}

    # splitting_points 정렬
    splitting_points = sorted(splitting_points)

    # 경계 인덱스 생성
    # 첫 서브모델 시작은 0
    boundaries = [0]
    for p in splitting_points:
        boundaries.append(layer_map[p] + 1)
    boundaries.append(len(model.layers))

    submodels = []
    current_input_shape = input_shape

    # 첫 submodel은 원래 입력 shape 사용
    # 이후 submodel은 이전 submodel의 출력 shape 기반으로 설정하기 위해 dummy forward 필요
    prev_output = None

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        layer_slice = model.layers[start:end]

        # 새로운 submodel 정의
        # 첫 번째 submodel은 original input shape
        if i == 0:
            # 첫 submodel 입력
            inputs = tf.keras.Input(shape=current_input_shape)
            x = inputs
            for lyr in layer_slice:
                x = lyr(x)
        else:
            # 이후 submodel은 이전 submodel의 출력 shape를 알아야 함
            # prev_output 형태에서 Input을 생성
            inputs = tf.keras.Input(shape=prev_output.shape[1:])
            x = inputs
            for lyr in layer_slice:
                x = lyr(x)

        submodel = tf.keras.Model(inputs, x)
        submodels.append(submodel)

        # dummy forward 통해 shape확인
        dummy_input = tf.zeros((1,) + current_input_shape)
        dummy_output = submodel(dummy_input)
        prev_output = dummy_output
        current_input_shape = tuple(dummy_output.shape[1:].as_list())

    return submodels


def assign_weights_to_submodels(original_model, submodels):
    """
    original_model: 전체 모델 (이미 weights load됨)
    submodels: split_resnet으로 만든 submodel 리스트
    각 submodel에 맞는 weights를 original_model에서 가져와 할당
    """
    # original_model 레이어를 순회하며 submodel 레이어와 이름 일치 시킴
    # 여기서는 submodel은 original_model의 레이어를 slice한 것이므로 순서 같음.
    # 단순히 submodel 레이어에 original 레이어의 get_weights() -> set_weights() 수행
    # 주의: submodel을 slicing한 레이어 순서와 original_model 순서가 동일하다고 가정.
    # split_resnet 함수에서 order 유지했으므로 문제없음.
    original_layers = original_model.layers

    # submodel별로 레이어 인덱스를 추출하고 가중치 적용
    start_idx = 0
    for sm in submodels:
        # submodel은 InputLayer를 포함하므로 실제 가중치가 있는 레이어는 len(sm.layers)-1개
        real_layers_count = len(sm.layers) - 1

        # original_model에서 real_layers_count만큼 레이어를 가져와야 함
        end_idx = start_idx + real_layers_count

        # sm.layers[1:]와 original_layers[start_idx:end_idx]를 매칭
        for sm_lyr, orig_lyr in zip(sm.layers[1:], original_layers[start_idx:end_idx]):
            sm_lyr.set_weights(orig_lyr.get_weights())

        start_idx = end_idx

original_model = ResNet34(num_classes=200)
original_model.build((None, 64, 64, 3))  # 파라미터 초기화
original_model.load_weights("ResNet_ImageNet_tf")  # 가중치 로드

# 하나의 splitting point 예: 0
submodels = split_resnet(original_model, [0])
assign_weights_to_submodels(original_model, submodels)

# 이제 submodels[0], submodels[1]가 분리된 모델이며, 각자의 weights가 로드됨.

# 두 개의 splitting point 예: 0, 2
submodels_multi = split_resnet(original_model, [0, 2])
assign_weights_to_submodels(original_model, submodels_multi)

test_accuracy = 0.0
count = 0

# submodel inference test
for data, label in test_ds:
    # 첫 번째 submodel로 중간 결과(특징맵) 얻기
    with tf.device(device):
        intermediate_output = submodels[0](data, training=False)  # 첫 번째 submodel 추론

    # 두 번째 submodel에 중간 출력 입력하여 최종 예측값 얻기
    with tf.device(device):
        preds = submodels[1](intermediate_output, training=False)  # 두 번째 submodel 추론

    # preds와 label을 numpy 형태로 변환
    preds = preds.numpy()
    label = label.numpy()

    # 정확도 계산
    batch_acc = evaluation_for_SC(preds, label)
    test_accuracy += batch_acc
    count += 1

final_accuracy = test_accuracy / count
print(f"Test Accuracy using split submodels: {final_accuracy:.4f}")



# 두 개의 splitting point 예: 0, 2
submodels_multi = split_resnet(original_model, [0, 2])
assign_weights_to_submodels(original_model, submodels_multi)

print("submodels", submodels)
print("submodels_multi", submodels_multi)

print("submodels")
submodels[0].summary()
submodels[1].summary()

print("submodels_multi")
submodels_multi[0].summary()
submodels_multi[1].summary()
submodels_multi[2].summary()

test_accuracy = 0.0
count = 0

# submodel inference test
for data, label in test_ds:
    # 첫 번째 submodel로 중간 결과(특징맵) 얻기
    with tf.device(device):
        intermediate_output = submodels[0](data, training=False)  # 첫 번째 submodel 추론

    # 두 번째 submodel에 중간 출력 입력하여 최종 예측값 얻기
    with tf.device(device):
        preds = submodels[1](intermediate_output, training=False)  # 두 번째 submodel 추론

    # preds와 label을 numpy 형태로 변환
    preds = preds.numpy()
    label = label.numpy()

    # 정확도 계산
    batch_acc = evaluation_for_SC(preds, label)
    test_accuracy += batch_acc
    count += 1

final_accuracy = test_accuracy / count
print(f"Test Accuracy using split submodels: {final_accuracy:.4f}")

# submodel inference test
for data, label in test_ds:
    # 첫 번째 submodel로 중간 결과(특징맵) 얻기
    with tf.device(device):
        intermediate_output_1 = submodels_multi[0](data, training=False)  # 첫 번째 submodel 추론

    # 두 번째 submodel에 중간 출력 입력하여 최종 예측값 얻기
    with tf.device(device):
        intermediate_output_2 = submodels_multi[1](intermediate_output_1, training=False)  # 두 번째 submodel 추론

    # 세 번째 submodel에 중간 출력 입력하여 최종 예측값 얻기
    with tf.device(device):
        preds = submodels_multi[2](intermediate_output_2, training=False)  # 두 번째 submodel 추론


    # preds와 label을 numpy 형태로 변환
    preds = preds.numpy()
    label = label.numpy()

    # 정확도 계산
    batch_acc = evaluation_for_SC(preds, label)
    test_accuracy += batch_acc
    count += 1

final_accuracy = test_accuracy / count
print(f"Test Accuracy using split submodels_multi: {final_accuracy:.4f}")