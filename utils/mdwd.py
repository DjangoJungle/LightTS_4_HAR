import pywt
import numpy as np


def mdwd(data, wavelet='db1', level=3):
    """
    多级离散小波分解（MDWD）
    :param data: 输入数据（numpy数组）
    :param wavelet: 小波类型（默认使用'db1'）
    :param level: 分解层数（默认3层）
    :return: 分解后的系数列表
    """
    coeffs = pywt.wavedec(data, wavelet, level=level)
    return coeffs
