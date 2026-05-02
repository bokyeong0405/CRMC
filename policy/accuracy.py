"""ResNet34 split-computing accuracy predictor.

Ported from ``resnet34_TinyImageNet/ResNet_predict_accuracy.py`` with two
practical changes:

1. ``data_dir`` is read from the env var ``CRMC_ACC_DATA_DIR`` (defaults to
   ``/data/acc/`` inside containers), instead of the hard-coded Windows path.
2. If the lookup ``.npy`` file is missing, ``predict_accuracy_res`` returns
   ``None`` rather than raising. Callers should treat ``None`` as "accuracy
   data unavailable, skip the threshold filter" — this lets the system run
   end-to-end before the offline calibration files are mounted.
"""

import logging
import os

import numpy as np
import scipy.linalg
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = os.environ.get("CRMC_ACC_DATA_DIR", "/data/acc/")

# Mirrors the original ``mapping_dict``: end-layer-in-original-numbering →
# split-point-id used in the calibration filenames (acc_{p}.npy /
# acc_{p1}_{p2}.npy). 7 is the sentinel for "no split / single device".
_SP_FILENAME_MAP = {2: 0, 7: 1, 14: 2, 18: 3, 31: 4, 34: 7, 0: 7}


def _map_split_points(split_points):
    return [_SP_FILENAME_MAP[v] for v in split_points]


def predict_accuracy_res(split_points, pdr_values, data_dir=None):
    """Return predicted accuracy in [0, 1], or ``None`` if calibration data is
    missing.

    ``split_points`` and ``pdr_values`` follow the original GSPDA convention:
    the first entry corresponds to the source-device "no compute" hop and is
    dropped before the lookup.
    """
    data_dir = data_dir or DEFAULT_DATA_DIR

    if not isinstance(split_points, (list, tuple)):
        raise ValueError("split_points should be a list or tuple of split points.")

    if len(split_points) >= 2 and split_points[0] == 0:
        split_points = split_points[1:]
        pdr_values = pdr_values[1:]

    mapped = _map_split_points(split_points)

    if not isinstance(pdr_values, (list, tuple)):
        pdr_values = [pdr_values]

    # "No split" — whole model on one device, accuracy is the baseline.
    if mapped[0] == 7:
        return 0.5052

    if len(mapped) == 1:
        file_path = os.path.join(data_dir, f"acc_{mapped[0]}.npy")
        if not os.path.exists(file_path):
            logger.warning("Accuracy lookup file missing: %s", file_path)
            return None

        data_loaded = np.load(file_path)
        x = np.arange(10, 60, 10)
        y = data_loaded[:, 1]

        poly = PolynomialFeatures(degree=4, include_bias=False)
        poly_features = poly.fit_transform(x.reshape(-1, 1))
        model = LinearRegression()
        model.fit(poly_features, y)

        pdr_input = np.array([[pdr_values[0]]])
        return float(model.predict(poly.transform(pdr_input))[0])

    if len(mapped) == 2:
        if len(pdr_values) != 2:
            raise ValueError("For two split points, pdr_values must have two entries.")

        file_path = os.path.join(data_dir, f"acc_{mapped[0]}_{mapped[1]}.npy")
        if not os.path.exists(file_path):
            logger.warning("Accuracy lookup file missing: %s", file_path)
            return None

        data_loaded = np.load(file_path)
        xs = np.arange(10, 60, 10)
        ys = np.arange(10, 60, 10)
        xx, yy = np.meshgrid(xs, ys)
        zz = data_loaded

        x_flat = np.array(xx).flatten()
        y_flat = np.array(yy).flatten()
        z_flat = np.array(zz).flatten()

        data = np.stack([x_flat, y_flat, z_flat], 1)
        A = np.c_[
            np.ones(data.shape[0]),
            data[:, :2],
            np.prod(data[:, :2], axis=1),
            data[:, :2] ** 2,
        ]
        coef, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])

        p1, p2 = pdr_values
        return float(np.dot([1, p1, p2, p1 * p2, p1 ** 2, p2 ** 2], coef))

    raise ValueError("Only 1 or 2 split points are supported.")
