import numpy as np


def get_range_curve(adc_data, sampling_rate):
    """
    对ADC数据执行FFT以获取Range曲线

    参数:
    adc_data (array): 包含ADC数据的数组，维度未知
    sampling_rate (float): ADC采样率

    返回值:
    range_curve (array): Range曲线，包含频谱的幅度值
    """

    # 获取ADC数据的长度并将其变形为一维数组
    data_len = np.prod(adc_data.shape)
    adc_data = adc_data.flatten()

    # 计算FFT和相应的频率数组
    fft = np.fft.fft(adc_data)
    freqs = np.fft.fftfreq(data_len) * sampling_rate

    # 计算Range曲线（频谱的幅度值）
    range_curve = np.abs(fft / data_len)

    # 只返回前一半（正频率部分）以避免重复计算
    half_range_curve = range_curve[:data_len // 2]
    half_freqs = freqs[:data_len // 2]

    return half_range_curve, half_freqs