import numpy as np

def chirp1dfft(numsample, numchirp, numframe, data):
    win = np.hamming(numsample)
    r = np.zeros((numsample, numchirp * numframe))
    for i in range(numchirp * numframe):
        temp = data[:, i] * win  # 将data矩阵第i列与汉明窗win逐元素相乘
        temp = np.fft.fft(temp, numsample)  # 对temp向量做FFT变换
        r[:, i] = temp  # 存储FFT变换结果到range矩阵的第i列

        a = 0.625  # 加权因子
        b = np.zeros(numsample)
        temp = np.zeros((numsample, numchirp * numframe))
        for i in range(numframe):
            h = range[:, i]  # 取出range矩阵的第i列
            b = a * b + (1 - a) * h  # 更新b向量
            temp[:, i] = h - b  # 计算并存储当前帧与平均值之间的差值