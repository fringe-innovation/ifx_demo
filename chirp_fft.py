import numpy as np

def chirp1dfft(numsample, numchirp, numframe, data):
    win = np.hamming(numsample)
    range = np.zeros((numsample, numchirp * numframe))
    for i in range(numchirp * numframe):
        temp = data[:, i] * win  # 将data矩阵第i列与汉明窗win逐元素相乘
        temp = np.fft.fft(temp, numsample)  # 对temp向量做FFT变换
        range[:, i] = temp  # 存储FFT变换结果到range矩阵的第i列