import numpy as np


def peakheart(data_in, breath_index):
    heart_start_freq_index = 20  # 开始索引
    heart_end_freq_index = 60  # 搜索的结束索引
    # data_lan = len(data_in)
    p_peak_values = np.zeros(128)  # 索引对应峰值
    p_peak_index = np.zeros(128)  # 索引
    max_num_peaks_spectrum = 4
    convert_hz_bpm = 60.0
    freq_increment_hz = 0.0195
    breathing_harmonic_num = 2
    breathing_harmonic_thresh_bpm = 4.0
    num_peaks = 0

    for i in range(heart_start_freq_index, heart_end_freq_index):
        if data_in[i] > data_in[i - 1] and data_in[i] > data_in[i + 1]:
            p_peak_index[num_peaks] = i
            p_peak_values[num_peaks] = data_in[i]
            num_peaks += 1
    if num_peaks < max_num_peaks_spectrum:
        index_num_peaks = num_peaks
    else:
        index_num_peaks = max_num_peaks_spectrum

    p_peak_index_sorted = np.zeros(index_num_peaks)
    if index_num_peaks != 0:
        for i in range(index_num_peaks):
            idx = np.argmax(p_peak_values)
            p_peak_index_sorted[i] = idx
            p_peak_values[idx] = 0
        max_index_breath_spect = p_peak_index[int(p_peak_index_sorted[0])]
    else:
        max_index_breath_spect = np.argmax(data_in[heart_start_freq_index:heart_end_freq_index])

    diff_index = np.abs(max_index_breath_spect - breathing_harmonic_num * breath_index)
    if (diff_index * freq_increment_hz * convert_hz_bpm) < breathing_harmonic_thresh_bpm:
        max_index_breath_spect = p_peak_index[int(p_peak_index_sorted[2])]

    res = convert_hz_bpm * (max_index_breath_spect - 1) * freq_increment_hz
    return res
