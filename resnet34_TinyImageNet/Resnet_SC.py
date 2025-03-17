import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tqdm import tqdm
from evaluation_util import *

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
    def __init__(self, block, num_blocks, num_classes=10):
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
        x = self.fc(x)  # N x num_classes
        return x


def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


# -----------------------------
# CIFAR-10 데이터 로드 및 전처리
# -----------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = y_train.squeeze()  # (50000, 1) -> (50000,)
y_test = y_test.squeeze()    # (10000, 1) -> (10000,)

# 정규화 & augmentation
def preprocess_train(image, label):
    image = tf.cast(image, tf.float32)
    # 랜덤 크롭: 원본 32x32 -> padding 후 32x32 크롭
    image = tf.image.resize_with_pad(image, 40, 40)  # 4픽셀 패딩 효과
    image = tf.image.random_crop(image, [32,32,3])
    image = tf.image.random_flip_left_right(image)
    image = image / 255.0
    # CIFAR mean,std 적용 가능하지만 여기서는 단순 normalize만 시연
    return image, label

def preprocess_test(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

batch_size = 32
train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(50000)
            .map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))

test_ds = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
           .map(preprocess_test, num_parallel_calls=tf.data.AUTOTUNE)
           .batch(batch_size)
           .prefetch(tf.data.AUTOTUNE))
print()

device = "/gpu:0" if len(tf.config.list_physical_devices('GPU')) > 0 else "/cpu:0"
# 모델 생성
model = ResNet34(num_classes=10)
model.build((None, 32, 32, 3))
model.summary()

# 옵티마이저, 로스 정의
lr = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

epochs = 30

with tf.device(device):
    for epoch in range(epochs):
        # training loop
        iterator = tqdm(train_ds)
        for data, label in iterator:
            with tf.GradientTape() as tape:
                preds = model(data, training=True)
                loss = loss_fn(label, preds)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            iterator.set_description(f"epoch:{epoch+1} loss:{loss.numpy():.4f}")

# 가중치 저장
model.save_weights("ResNet_tf_v2")

new_model = ResNet34(num_classes=10)
new_model.build((None, 32, 32, 3))  # 파라미터 초기화
new_model.load_weights("ResNet_tf_v2")  # 가중치 로드

# 테스트 정확도 계산
test_accuracy = 0.0
count = 0
for data, label in test_ds:
    preds = new_model(data, training=False)
    preds = preds.numpy()  # 텐서에서 numpy로 변환
    label = label.numpy()  # 라벨도 numpy 배열로 변환
    batch_acc = evaluation_for_SC(preds, label)
    test_accuracy += batch_acc
    count += 1

test_accuracy = test_accuracy / count
print(f"Test Accuracy after loading weights: {test_accuracy:.4f}")
# new_model.summary()
# Test Accuracy after loading weights: 0.8128

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

# 예제 사용:
# 원본 모델 로드
# (ResNet34 정의가 되어있다고 가정)
original_model = ResNet34(num_classes=10)
original_model.build((None, 32, 32, 3))
original_model.load_weights("ResNet_tf_v2")

# 하나의 splitting point 예: 0
submodels = split_resnet(original_model, [0])
assign_weights_to_submodels(original_model, submodels)

# 이제 submodels[0], submodels[1]가 분리된 모델이며, 각자의 weights가 로드됨.

# 두 개의 splitting point 예: 0, 2
submodels_multi = split_resnet(original_model, [0, 2])
assign_weights_to_submodels(original_model, submodels_multi)

# print("submodels", submodels)
# print("submodels_multi", submodels_multi)
#
# print("submodels")
# submodels[0].summary()
# submodels[1].summary()
#
# print("submodels_multi")
# submodels_multi[0].summary()
# submodels_multi[1].summary()
# submodels_multi[2].summary()

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