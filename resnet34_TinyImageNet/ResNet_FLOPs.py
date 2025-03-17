import tensorflow as tf
from tensorflow.keras import layers, models, Input

def ResNet34(input_shape=(32, 32, 3), num_classes=10):
    inputs = Input(shape=input_shape)

    # Initial Layers
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False, name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn_conv1')(x)
    x = layers.ReLU(name='relu_conv1')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='max_pool')(x)

    # Layer 1
    for i in range(3):
        name_suffix = f'layer1_block{i+1}'
        x_prev = x
        x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False, name=f'{name_suffix}_conv1')(x)
        x = layers.BatchNormalization(name=f'{name_suffix}_bn1')(x)
        x = layers.ReLU(name=f'{name_suffix}_relu1')(x)
        x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False, name=f'{name_suffix}_conv2')(x)
        x = layers.BatchNormalization(name=f'{name_suffix}_bn2')(x)
        if i == 0:
            shortcut = layers.Conv2D(64, kernel_size=1, strides=1, padding='same', use_bias=False, name=f'{name_suffix}_shortcut_conv')(x_prev)
            shortcut = layers.BatchNormalization(name=f'{name_suffix}_shortcut_bn')(shortcut)
        else:
            shortcut = x_prev  # Identity
        x = layers.Add(name=f'{name_suffix}_add')([x, shortcut])
        x = layers.ReLU(name=f'{name_suffix}_relu2')(x)

    # Layer 2
    for i in range(4):
        name_suffix = f'layer2_block{i+1}'
        strides = 2 if i == 0 else 1
        x_prev = x
        x = layers.Conv2D(128, kernel_size=3, strides=strides, padding='same', use_bias=False, name=f'{name_suffix}_conv1')(x)
        x = layers.BatchNormalization(name=f'{name_suffix}_bn1')(x)
        x = layers.ReLU(name=f'{name_suffix}_relu1')(x)
        x = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', use_bias=False, name=f'{name_suffix}_conv2')(x)
        x = layers.BatchNormalization(name=f'{name_suffix}_bn2')(x)
        if i == 0:
            shortcut = layers.Conv2D(128, kernel_size=1, strides=strides, padding='same', use_bias=False, name=f'{name_suffix}_shortcut_conv')(x_prev)
            shortcut = layers.BatchNormalization(name=f'{name_suffix}_shortcut_bn')(shortcut)
        else:
            shortcut = x_prev  # Identity
        x = layers.Add(name=f'{name_suffix}_add')([x, shortcut])
        x = layers.ReLU(name=f'{name_suffix}_relu2')(x)

    # Layer 3
    for i in range(6):
        name_suffix = f'layer3_block{i+1}'
        strides = 2 if i == 0 else 1
        x_prev = x
        x = layers.Conv2D(256, kernel_size=3, strides=strides, padding='same', use_bias=False, name=f'{name_suffix}_conv1')(x)
        x = layers.BatchNormalization(name=f'{name_suffix}_bn1')(x)
        x = layers.ReLU(name=f'{name_suffix}_relu1')(x)
        x = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', use_bias=False, name=f'{name_suffix}_conv2')(x)
        x = layers.BatchNormalization(name=f'{name_suffix}_bn2')(x)
        if i == 0:
            shortcut = layers.Conv2D(256, kernel_size=1, strides=strides, padding='same', use_bias=False, name=f'{name_suffix}_shortcut_conv')(x_prev)
            shortcut = layers.BatchNormalization(name=f'{name_suffix}_shortcut_bn')(shortcut)
        else:
            shortcut = x_prev  # Identity
        x = layers.Add(name=f'{name_suffix}_add')([x, shortcut])
        x = layers.ReLU(name=f'{name_suffix}_relu2')(x)

    # Layer 4
    for i in range(3):
        name_suffix = f'layer4_block{i+1}'
        strides = 2 if i == 0 else 1
        x_prev = x
        x = layers.Conv2D(512, kernel_size=3, strides=strides, padding='same', use_bias=False, name=f'{name_suffix}_conv1')(x)
        x = layers.BatchNormalization(name=f'{name_suffix}_bn1')(x)
        x = layers.ReLU(name=f'{name_suffix}_relu1')(x)
        x = layers.Conv2D(512, kernel_size=3, strides=1, padding='same', use_bias=False, name=f'{name_suffix}_conv2')(x)
        x = layers.BatchNormalization(name=f'{name_suffix}_bn2')(x)
        if i == 0:
            shortcut = layers.Conv2D(512, kernel_size=1, strides=strides, padding='same', use_bias=False, name=f'{name_suffix}_shortcut_conv')(x_prev)
            shortcut = layers.BatchNormalization(name=f'{name_suffix}_shortcut_bn')(shortcut)
        else:
            shortcut = x_prev  # Identity
        x = layers.Add(name=f'{name_suffix}_add')([x, shortcut])
        x = layers.ReLU(name=f'{name_suffix}_relu2')(x)

    # Final Layers
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    outputs = layers.Dense(num_classes, name='fc')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='ResNet34')
    return model

def get_num_elements(shape):
    """Helper function to compute the total number of elements from a TensorShape."""
    num = 1
    for dim in shape:
        if dim is not None:
            num *= dim
    return num

def compute_flops(layer, input_shape, output_shape):
    """
    Compute FLOPs for a given layer based on its type and input/output shapes.

    Args:
        layer: The Keras layer instance.
        input_shape: TensorShape of the input to the layer.
        output_shape: TensorShape of the output from the layer.

    Returns:
        flops: Number of floating point operations for the layer.
    """
    flops = 0

    if isinstance(layer, layers.Conv2D):
        # FLOPs for Conv2D: 2 * kernel_height * kernel_width * in_channels * out_channels * output_height * output_width
        kernel_height, kernel_width = layer.kernel_size
        in_channels = input_shape[-1]
        out_channels = layer.filters
        output_height, output_width = output_shape[1], output_shape[2]
        flops = 2 * kernel_height * kernel_width * in_channels * out_channels * output_height * output_width

    elif isinstance(layer, layers.Dense):
        # FLOPs for Dense: 2 * in_features * out_features
        in_features = input_shape[-1]
        out_features = layer.units
        flops = 2 * in_features * out_features

    elif isinstance(layer, layers.BatchNormalization):
        # FLOPs for BatchNormalization: 2 * number of features (scale and shift)
        flops = 2 * output_shape[-1]

    elif isinstance(layer, layers.ReLU) or isinstance(layer, layers.Activation):
        # FLOPs for ReLU: number of elements
        flops = get_num_elements(output_shape[1:])  # Exclude batch dimension

    elif isinstance(layer, layers.MaxPooling2D):
        # FLOPs for MaxPooling: (kernel_height * kernel_width - 1) * output_elements
        pool_height, pool_width = layer.pool_size
        output_elements = get_num_elements(output_shape[1:])  # Exclude batch dimension
        flops = (pool_height * pool_width - 1) * output_elements

    elif isinstance(layer, layers.GlobalAveragePooling2D):
        # FLOPs for GlobalAveragePooling: input_height * input_width * channels
        input_height, input_width, channels = input_shape[1], input_shape[2], input_shape[3]
        flops = input_height * input_width * channels

    elif isinstance(layer, layers.Add):
        # FLOPs for Add: number of elements
        flops = get_num_elements(output_shape[1:])  # Exclude batch dimension

    # Add more layer types if needed

    return flops

def get_model_flops(model, input_shape=(1, 32, 32, 3)):
    """
    Compute and display FLOPs for each layer and the total model.

    Args:
        model: The Keras model instance.
        input_shape: Tuple representing the input shape (including batch size).

    Returns:
        flops_list: List of tuples containing layer names and their FLOPs.
        total_flops: Total FLOPs for the model.
    """
    flops_list = []
    tensor_shape_map = {}

    # Initialize input tensors' shapes
    for input_tensor in model.inputs:
        tensor_shape_map[input_tensor.ref()] = tf.TensorShape(input_shape)

    # Iterate over layers in the order they are defined in the model
    for layer in model.layers:
        # Get input tensors
        inputs = layer.input  # Could be a list or a single tensor

        if isinstance(inputs, list):
            input_shapes = [tensor_shape_map.get(t.ref(), None) for t in inputs]
            if None in input_shapes:
                print(f"Warning: Missing input shape for layer {layer.name}. Skipping FLOP computation for this layer.")
                continue
        else:
            input_shapes = [tensor_shape_map.get(inputs.ref(), None)]
            if input_shapes[0] is None:
                print(f"Warning: Missing input shape for layer {layer.name}. Skipping FLOP computation for this layer.")
                continue

        # Compute output shape
        if len(input_shapes) == 1:
            input_shape_layer = input_shapes[0]
        else:
            input_shape_layer = input_shapes

        # Ensure the layer is built
        if not layer.built:
            try:
                layer.build(input_shape_layer)
            except Exception as e:
                print(f"Error building layer {layer.name}: {e}")
                continue

        # Compute output shape
        try:
            output_shape = layer.compute_output_shape(input_shape_layer)
        except Exception as e:
            print(f"Error computing output shape for layer {layer.name}: {e}")
            continue

        # Compute FLOPs
        layer_flops = compute_flops(layer, input_shape_layer, output_shape)
        flops_list.append((layer.name, layer_flops))

        # Map output tensors to shape
        if isinstance(layer.output, list):
            for o in layer.output:
                tensor_shape_map[o.ref()] = output_shape
        else:
            tensor_shape_map[layer.output.ref()] = output_shape

    # Aggregate and print FLOPs
    total_flops = 0
    print("\nLayer-wise FLOPs (in MFLOPs):")
    for layer_name, flops in flops_list:
        flops_m = flops / 1e6
        print(f"{layer_name} : {flops_m:.2f} MFLOPs")
        total_flops += flops
    total_flops_m = total_flops / 1e6
    print("==============================================")
    print(f"Total FLOPs: {total_flops_m:.2f} MFLOPs")
    return flops_list, total_flops

if __name__ == "__main__":
    # Instantiate and build the model
    model = ResNet34(input_shape=(32, 32, 3), num_classes=10)
    model.build((None, 32, 32, 3))
    model.summary()

    # Compute and print FLOPs
    flops, total_flops = get_model_flops(model)

# Layer-wise FLOPs (in MFLOPs):
# conv2d : 0.08 MFLOPs
# batch_normalization : 0.00 MFLOPs
# re_lu : 0.00 MFLOPs
# max_pooling2d : 0.00 MFLOPs
#
# sequential_3 : 5.32 MFLOPs
# 1
# sequential_8 : 21.28 MFLOPs
# 2
# sequential_15 : 53.83 MFLOPs
# 3
# sequential_19 : 131.22 MFLOPs
# 4
# global_average_pooling2d : 0.00 MFLOPs
# dense : 0.01 MFLOPs
# ==============================================
# Total FLOPs: 211.74 MFLOPs
