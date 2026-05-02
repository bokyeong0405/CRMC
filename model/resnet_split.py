"""ResNet34 split-computing library.

Ported from ``resnet34_TinyImageNet/Resnet_SC.py`` (which mixed model definition,
training, evaluation, and split logic in one script). Only the model classes and
the splitting helpers are kept here so they can be imported by the device server
and the orchestrator without triggering any side effects.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


# Split point index → index of the last layer that belongs to the *previous*
# submodel. Boundary is `layer_map[p] + 1`.
#   0 → after layer1 (first residual stage)
#   1 → after layer2
#   2 → after layer3
#   3 → after layer4
#   4 → after avgpool (i.e. only fc remains in the next submodel)
LAYER_MAP = {0: 3, 1: 4, 2: 5, 3: 6, 4: 7}


class BasicBlock(tf.keras.layers.Layer):
    mul = 1

    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.conv1 = layers.Conv2D(out_planes, kernel_size=3, strides=stride, padding="same", use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(out_planes, kernel_size=3, strides=1, padding="same", use_bias=False)
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
        return tf.nn.relu(out + shortcut)


class ResNet(tf.keras.Model):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64

        self.conv1 = layers.Conv2D(self.in_planes, kernel_size=3, strides=1, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()

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
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        x = self.fc(x)
        return x


def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def load_full_model(weight_path=None, num_classes=10, input_shape=(32, 32, 3)):
    """Build ResNet34 and (optionally) load weights.

    If ``weight_path`` is ``None`` or the file does not exist, the model keeps
    its random initialisation — useful for end-to-end plumbing tests before the
    real ``.h5`` file is mounted.
    """
    model = ResNet34(num_classes=num_classes)
    model.build((None,) + tuple(input_shape))
    if weight_path:
        try:
            model.load_weights(weight_path)
        except (tf.errors.NotFoundError, OSError):
            # Caller is responsible for logging this fallback.
            pass
    return model


def split_resnet(model, splitting_points, input_shape):
    """Slice ``model`` into ``len(splitting_points) + 1`` submodels.

    ``splitting_points`` is a list of ints in [0, 4]; see ``LAYER_MAP`` for the
    boundary semantics. The return is a list of fresh ``tf.keras.Model``
    instances that share *no weights* with ``model`` until
    ``assign_weights_to_submodels`` is called.
    """
    splitting_points = sorted(splitting_points)

    boundaries = [0]
    for p in splitting_points:
        boundaries.append(LAYER_MAP[p] + 1)
    boundaries.append(len(model.layers))

    submodels = []
    current_input_shape = tuple(input_shape)
    prev_output = None

    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        layer_slice = model.layers[start:end]

        if i == 0:
            inputs = tf.keras.Input(shape=current_input_shape)
        else:
            inputs = tf.keras.Input(shape=prev_output.shape[1:])
        x = inputs
        for lyr in layer_slice:
            x = lyr(x)

        submodel = tf.keras.Model(inputs, x)
        submodels.append(submodel)

        dummy = tf.zeros((1,) + current_input_shape)
        prev_output = submodel(dummy)
        current_input_shape = tuple(prev_output.shape[1:].as_list())

    return submodels


def assign_weights_to_submodels(original_model, submodels):
    """Copy weights from ``original_model.layers`` into the matching submodel
    layers in order. Assumes ``split_resnet`` preserved the original layer
    ordering (it does)."""
    original_layers = original_model.layers
    start_idx = 0
    for sm in submodels:
        # Submodel always has an InputLayer prepended.
        real_layers_count = len(sm.layers) - 1
        end_idx = start_idx + real_layers_count
        for sm_lyr, orig_lyr in zip(sm.layers[1:], original_layers[start_idx:end_idx]):
            sm_lyr.set_weights(orig_lyr.get_weights())
        start_idx = end_idx


def build_submodels(weight_path, splitting_points, num_classes=10, input_shape=(32, 32, 3)):
    """One-shot helper: load the full model + split + assign weights.

    Returns ``(full_model, submodels)``.
    """
    full_model = load_full_model(weight_path, num_classes=num_classes, input_shape=input_shape)
    submodels = split_resnet(full_model, splitting_points, input_shape=input_shape)
    assign_weights_to_submodels(full_model, submodels)
    return full_model, submodels
