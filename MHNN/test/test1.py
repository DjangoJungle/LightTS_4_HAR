import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy import signal


# 生成带噪音的信号
def generate_signal(noise_level=0.5):
    t = np.linspace(0, 1, 400)
    # 原始信号为正弦波 + 正弦波的叠加
    signal = np.sin(2 * np.pi * 30 * t) + np.sin(2 * np.pi * 60 * t)
    # 加入噪音
    noise = noise_level * np.random.randn(len(t))
    return t, signal + noise


# 小波变换去噪函数
def denoise_signal(noisy_signal, wavelet='db4', level=4, threshold=None):
    # 使用离散小波变换分解信号
    coeffs = pywt.wavedec(noisy_signal, wavelet, level=level)

    # 如果没有设定阈值，则自动根据中位数设定
    if threshold is None:
        threshold = np.median(np.abs(coeffs[-level])) / 0.6745

    # 阈值处理（软阈值处理）
    denoised_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]

    # 使用逆小波变换重建信号
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)

    return denoised_signal


# 可视化原始信号、噪声信号及去噪后的信号
def plot_signals(t, original_signal, noisy_signal, denoised_signal):
    plt.figure(figsize=(10, 7))

    plt.subplot(3, 1, 1)
    plt.plot(t, original_signal)
    plt.title('Original Signal')

    plt.subplot(3, 1, 2)
    plt.plot(t, noisy_signal)
    plt.title('Noisy Signal')

    plt.subplot(3, 1, 3)
    plt.plot(t, denoised_signal)
    plt.title('Denoised Signal using Wavelet Transform')

    plt.tight_layout()
    plt.show()


# 生成信号
t, noisy_signal = generate_signal(noise_level=0.5)
original_signal = np.sin(2 * np.pi * 30 * t) + np.sin(2 * np.pi * 60 * t)

# 使用小波变换进行去噪
denoised_signal = denoise_signal(noisy_signal)

# 可视化结果
plot_signals(t, original_signal, noisy_signal, denoised_signal)
