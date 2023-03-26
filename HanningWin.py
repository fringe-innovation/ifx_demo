import numpy as np


def hanning_win(data_in):
    wincofeci = [0.0800, 0.0894, 0.1173, 0.1624, 0.2231, 0.2967, 0.3802, 0.4703, 0.5633, 0.6553, 0.7426, 0.8216, 0.8890,
                 0.9422, 0.9789, 0.9976]
    scale_heart_wfm = 300000
    win_len = len(wincofeci)
    data_len = len(data_in)
    res = np.zeros(data_len)
    for i in range(win_len):
        temp_data_head = np.real(data_in[i]) * scale_heart_wfm * wincofeci[i] + np.imag(data_in[i]) * 0
        temp_data_end = np.real(data_in[data_len-i-1]) * scale_heart_wfm * wincofeci[i] + np.imag(data_in[data_len-i-1]) * 0
        res[i] = temp_data_head
        res[data_len-i-1] = temp_data_end
    for i in (win_len, data_len-win_len):
        res[i] = data_in[i].real * scale_heart_wfm
