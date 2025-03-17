import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tqdm import tqdm
from evaluation_util import *
import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def burst_err(data, error_rate=0.1):
    result = np.ones_like(data)  # 입력 데이터와 똑같은 크기의 배열 생성
    burst_period = 365

    error_flag = False
    count = 0
    for idx, i in np.ndenumerate(result):
        if count == 0:
            if np.random.rand(1) <= error_rate:
                error_flag = True
            else:
                error_flag = False
        if error_flag:
            result[idx] = 0
        count += 1
        if count == burst_period:
            count = 0
    return result


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
        self.conv1 = layers.Conv2D(self.in_planes, kernel_size=7, strides=2, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.maxpool1 = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')

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
        x = self.maxpool1(x)

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

device = "/gpu:0" if len(tf.config.list_physical_devices('GPU')) > 0 else "/cpu:0"

def split_resnet(model, splitting_points, input_shape=(32, 32, 3)):
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

# 가능한 splitting points
possible_points = [0,1,2,3,4]

# 조합 구하기 (단일 split, 2개 split)
one_point_combinations = [(p,) for p in possible_points]
two_points_combinations = list(itertools.combinations(possible_points, 2))

error_rates = [0.1, 0.2, 0.3, 0.4, 0.5]

original_model = ResNet34(num_classes=10)
original_model.build((None, 32, 32, 3))  # 파라미터 초기화
original_model.load_weights("ResNet_tf")  # 가중치 로드

def test_split_accuracy_1d(original_model, splitting_points, test_ds, error_rate):
    # 단일 split 또는 다중 split에 대해 1D error_rate 적용 (단일 split용)
    # 여기서는 두 개 split에 대한 1D 기능은 제외하므로 이 함수는 단일 split에만 사용
    submodels = split_resnet(original_model, splitting_points)
    assign_weights_to_submodels(original_model, submodels)
    num_sub = len(submodels)

    test_accuracy = 0.0
    count = 0
    for data, label in test_ds:
        with tf.device(device):
            output = data
            for i in range(num_sub):
                output = submodels[i](output, training=False)
                if i < num_sub - 1:
                    output_np = output.numpy()
                    mask = burst_err(output_np, error_rate=error_rate)
                    output = output_np * mask
        preds = output
        if not isinstance(preds, np.ndarray):
            preds = preds.numpy()
        label = label.numpy()
        batch_acc = evaluation_for_SC(preds, label)
        test_accuracy += batch_acc
        count += 1

    return test_accuracy / count

# 단일 split point 결과 측정 및 저장
one_point_results = {}
for sp in one_point_combinations:
    acc_dict = {}
    data_to_save = []
    for er in error_rates:
        acc = test_split_accuracy_1d(original_model, sp, test_ds, er)
        acc_dict[er] = acc
        data_to_save.append([er, acc])
    one_point_results[sp] = acc_dict
    np.save(f"acc_{'_'.join(map(str,sp))}.npy", np.array(data_to_save))



print("One-point split results:")
for sp, acc_dict in one_point_results.items():
    for er, acc in acc_dict.items():
        print(f"Split {sp}, error_rate {er}, accuracy {acc:.4f}")

    data_loaded = np.load(f"acc_{'_'.join(map(str, sp))}.npy")
    # data_loaded[:,0] = error_rate, data_loaded[:,1] = accuracy
    plt.figure()
    plt.plot(data_loaded[:, 0], data_loaded[:, 1], marker='o')
    plt.title(f"Split {sp} - 1D Error Rate vs Accuracy")
    plt.xlabel("Error Rate")
    plt.ylabel("Accuracy")
    plt.grid(True)
    # plt.savefig(f"plot_1d_{'_'.join(map(str, sp))}.png")
    plt.show()

error_rates_x = [0.1, 0.2, 0.3, 0.4, 0.5]
error_rates_y = [0.1, 0.2, 0.3, 0.4, 0.5]

def test_split_accuracy_2d(original_model, splitting_points, test_ds, error_rates_x, error_rates_y):
    # 두 개 split 포인트이면 submodel은 3개
    # 첫 split 후 er_x 적용, 두 번째 split 후 er_y 적용
    submodels = split_resnet(original_model, splitting_points)
    assign_weights_to_submodels(original_model, submodels)
    num_sub = len(submodels)
    Z = np.zeros((len(error_rates_x), len(error_rates_y)))
    for i, er_x in enumerate(error_rates_x):
        for j, er_y in enumerate(error_rates_y):
            test_accuracy = 0.0
            count = 0
            for data, label in test_ds:
                output = data
                output = submodels[0](output, training=False)
                out_np = output.numpy()
                mask_x = burst_err(out_np, error_rate=er_x)
                output = out_np * mask_x

                output = submodels[1](output, training=False)
                out_np = output.numpy()
                mask_y = burst_err(out_np, error_rate=er_y)
                output = out_np * mask_y

                # 마지막 submodel
                if num_sub == 3:
                    output = submodels[2](output, training=False)

                preds = output
                if not isinstance(preds, np.ndarray):
                    preds = preds.numpy()
                label = label.numpy()
                batch_acc = evaluation_for_SC(preds, label)
                test_accuracy += batch_acc
                count += 1
            Z[i,j] = test_accuracy / count
    return Z

# 두 개 split 지점에 대해 2D 실험 (예시로 (0,2)만 사용)
for sp in two_points_combinations: # 필요하면 다른 조합 추가 가능
    Z = test_split_accuracy_2d(original_model, sp, test_ds, error_rates_x, error_rates_y)
    np.save(f"./acc/acc_{'_'.join(map(str,sp))}.npy", Z)
    # Z[i,j]: er_x = error_rates_x[i], er_y = error_rates_y[j]

    ERx, ERy = np.meshgrid(error_rates_x, error_rates_y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(ERx, ERy, Z, cmap='viridis')
    ax.set_xlabel('Error Rate X')
    ax.set_ylabel('Error Rate Y')
    ax.set_zlabel('Accuracy')
    ax.set_title(f"3D Plot for Split {sp}")
    # plt.savefig(f"plot_3d_{'_'.join(map(str, sp))}.png")
    plt.show()