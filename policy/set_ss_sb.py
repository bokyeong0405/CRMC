"""Path → (split_points, BER_list) decoder for the accuracy predictor.

Ported from ``resnet34_TinyImageNet/set_ss_sb_resnet.py``.
"""


def set_ss_sb(short_path, end, D_BER, SS):
    sp = []
    b = []

    if len(short_path) != 1:
        for i in range(len(short_path) - 1):
            sp.append(end[short_path[i] % SS])
            i_dev = short_path[i] // SS
            j_dev = short_path[i + 1] // SS
            if i_dev == 0:
                b.append(D_BER[0] if j_dev == 1 else D_BER[2])
            elif i_dev == 1:
                b.append(D_BER[1] if j_dev == 2 else D_BER[0])
            elif i_dev == 2:
                b.append(D_BER[1] if j_dev == 1 else D_BER[2])
    else:
        sp.append(34)
        b.append(0)

    return sp, b
