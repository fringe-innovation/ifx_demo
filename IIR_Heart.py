# IIR 巴特沃斯带通滤波器通带截止频率0.8Hz,阻带截止频率2 Hz 过滤心跳信号
import numpy as np
from scipy.signal import butter, sosfilt


def iir_heart(n: int, phase: np.ndarray):
    fs = 20  # 采样率为20Hz，即0.05秒的采样间隔
    f1 = 0.8 / (fs/2)  # 归一化通带截止频率
    f2 = 2 / (fs/2)  # 归一化阻带截止频率
    # 设计巴特沃斯IIR滤波器
    sos = butter(n, [f1, f2], btype='bandpass', output='sos')
    res = sosfilt(sos, phase)
    return res



