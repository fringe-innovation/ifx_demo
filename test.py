import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import constants
from ifxAvian import Avian
from collections import deque

from fft_spectrum import fft_spectrum
from Peakcure import peakcure
from Diffphase import diffphase
from IIR_Heart import iir_heart
from IIR_Breath import iir_breath
from scipy.signal import argrelextrema

import time
class HumanPresenceAndDFFTAlgo:

    def __init__(self, config: Avian.DeviceConfig):
        self.num_samples_per_chirp = config.num_samples_per_chirp
        self.num_chirps_per_frame = config.num_chirps_per_frame

        # compute Blackman-Harris Window matrix over chirp samples(range)
        self.range_window = signal.blackmanharris(self.num_samples_per_chirp).reshape(1, self.num_samples_per_chirp)

        bandwidth_hz = abs(config.end_frequency_Hz - config.start_frequency_Hz)
        fft_size = self.num_samples_per_chirp * 2
        self.range_bin_length = constants.c / (2 * bandwidth_hz * fft_size / self.num_samples_per_chirp)

    def human_presence_and_dfft(self, data_in):  # sourcery skip: inline-immediately-returned-variable
        # data: single chirp data for single antenna

        # calculate range fft spectrum of the frame
        range_fft = fft_spectrum(data_in, self.range_window)

        # Average absolute FFT values over number of chirps
        fft_spec_abs = abs(range_fft)
        fft_norm = np.divide(fft_spec_abs.sum(axis=0), self.num_chirps_per_frame)

        skip = 25
        max_index = np.argmax(fft_norm[skip:])
        dist = self.range_bin_length * (max_index + skip)

        return range_fft, dist


def parse_program_arguments(description, def_nframes, def_frate):
    # Parse all program attributes
    # description:   describes program
    # def_nframes:   default number of frames
    # def_frate:     default frame rate in Hz

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-n', '--nframes', type=int,
                        default=def_nframes,
                        help=f"number of frames, default {str(def_nframes)}")
    parser.add_argument('-f', '--frate', type=int,
                        default=def_frate,
                        help=f"frame rate in Hz, default {str(def_frate)}")
    return parser.parse_args()


def db(x):
    return 20 * np.log10(np.abs(x))
def f(x):
    return 120 + 40 * x

# 定义一个函数，输入是一个心脏跳动曲线数组，输出是一个峰值数组
def detect_peaks(ecg):
    # 峰值阈值，只有超过这个值的点才被认为是峰值，可以根据实际情况调整
    threshold = 70
    return [
        (i, ecg[i])
        for i in range(1, ecg.size - 1)
        if ecg[i] > threshold and ecg[i] > ecg[i - 1] and ecg[i] > ecg[i + 1]
    ]

# 输入是一个峰值的索引和值的元组，输出是一个心率的值
def calculate_heart_rate(peak):
    # 假设每个点对应的时间间隔为0.1秒，你可以根据实际情况修改
    time_interval = 0.05
    # 从峰值元组中获取索引
    index = peak[0]
    return 60 / (index * time_interval)

# 输入是一个心率曲线数组，输出是一个平均心率的值
def sliding_window(hr):
    # 窗口大小
    window_size = 80
    # 输出数组，长度为(hr.size - window_size) // step + 1
    output = np.zeros(len(hr) - window_size)
    # 循环遍历每个窗口
    for i in range(len(hr) - window_size ):
    # 计算窗口内的平均值，并存储在输出数组中
        output[i] = np.mean(hr[i:i+window_size])
    # 返回输出数组中的最后一个元素，即最新的平均心率
    return output[-1]

if __name__ == '__main__':
    args = parse_program_arguments(
        '''Derives presence and peeking information from Radar Data''',
        def_nframes=256,
        def_frate=20)

    print(f"Radar SDK Version: {Avian.get_version()}")

    config = Avian.DeviceConfig(
        sample_rate_Hz=2e6,  # ADC sample rate of 2MHz
        rx_mask=6,  # RX antenna 1 activated
        tx_mask=1,  # TX antenna 1 activated
        tx_power_level=31,  # TX power level of 31
        if_gain_dB=33,  # 33dB if gain
        start_frequency_Hz=58e9,  # start frequency: 58.0 GHz
        end_frequency_Hz=63.5e9,  # end frequency: 63.5 GHz
        num_samples_per_chirp=256,  # 256 samples per chirp
        num_chirps_per_frame=1,  # 32 chirps per frame
        chirp_repetition_time_s=0.000150,  # Chirp repetition time (or pulse repetition time) of 150us
        frame_repetition_time_s=1 / args.frate,  # Frame repetition time default 0.05s (frame rate of 20Hz)
        hp_cutoff_Hz=20000,
        mimo_mode="off")  # MIMO disabled

    # connect to an Avian radar sensor
    with Avian.Device() as device:
        device.set_config(config)
        algo = HumanPresenceAndDFFTAlgo(config)

        num_frame = args.nframes
        num_chirp = config.num_chirps_per_frame
        num_sample = config.num_samples_per_chirp
        rate = 0
        np_time = 0
        heart_bit = 0
        heart = []
        breath = []
        data = []
        for _ in range(8000):
            frame = device.get_next_frame()
            frame = frame[0, 0, :]
            data.append(frame)

            data = np.array(data)
            mean_centering = np.zeros([num_frame, num_sample])
            avg = np.sum(data[:, :], axis=1) / num_sample
            for i in range(num_sample):
                mean_centering[:, i] = data[:, i] - avg
            dfft_data, dist = algo.human_presence_and_dfft(mean_centering)

            dfft_data = np.transpose(dfft_data)
            X, Y = np.meshgrid(np.linspace(0, 60, num_frame * num_chirp), np.arange(num_sample - 150))
            #print(dist)
            if 0.35 < dist < 1.1:
                # 无人计数置零
                np_time = 0
                # rang-bin相位提取及解纠缠
                rang_bin, phase, phase_unwrap = peakcure(dfft_data)
                # 相位差分
                diff_phase = diffphase(phase_unwrap)
                # 滑动平均滤波
                phase_remove = np.convolve(diff_phase, 5, 'same')
                # 过滤呼吸信号
                breath_wave = iir_breath(4, phase_remove)
                # 过滤心跳信号
                heart_wave = iir_heart(8, phase_remove)

                heart_fre = np.abs(np.fft.fftshift(np.fft.fft(heart_wave)))
                breath_fre = np.abs(np.fft.fft(breath_wave)) ** 2

                array_heart = np.array(heart_fre)
                array_breath = np.array(heart_fre)
                heart_peaks = argrelextrema(array_heart, np.greater, order=30)
                breath_peaks = argrelextrema(array_breath, np.greater, order=90)

                heart_peaks = heart_peaks[0][array_heart[heart_peaks] > f(dist)]
                heart_count = len(heart_peaks)
                heart.append(heart_count)

                breath_peaks = breath_peaks[0][array_breath[breath_peaks] >f(dist)]
                breath_count = len(breath_peaks)
                breath.append(breath_count)
                rate = rate + 1
                times = np.linspace(0, 60, num_frame)
                if len(heart) >= 1200:
                    heart_bit = sum(heart)
                    breath_bit = sum(breath)
                    heart.pop(0)
                    breath.pop(0)
                    if rate % 80 == 0:
                        print(breath_bit)
                        print(heart_bit)
                        rate = 1200  
            else:
                np_time = np_time + 1
                if np_time > 30:
                    print("no person")
                    np_time = 0  
            
            """ times = np.linspace(0, 60, num_frame)
                plt.figure(1)
                plt.subplot(2, 1, 1)
                plt.plot(times, breath_fre)
                plt.title('Respiratory waveform')
                plt.xlabel('t/s')
                plt.ylabel('dB')
                plt.subplot(2, 1, 2)
                plt.plot(times, heart_fre)
                plt.title('Heart waveform')
                plt.xlabel('t/s')
                plt.ylabel('dB') 
                plt.show()  """   
            data = []
            """if len(heart) >= 1200:
                    heart_bit = sum(heart)
                    heart.pop(0)
                    if rate % 40 == 0:
                      print(heart_bit)
                      rate = 1200 """
            

            """ for i in range(peak_count):
                c += 1   
                heart_bit = heart_bit + c
              rate += 1
              heart = heart + heart_bit
              heart_bit = 0  """


                                      #if len(heart) >= 1200:
                                      #  heart_bit = sum(heart)
                                      #  heart.pop(0)
                                      #  if rate % 40 == 0:
                                      #    print(heart_bit)
                                      #    rate = 1200     

"""                 for c, _ in enumerate(range(peak_count), start=1):
                    heart_bit = heart_bit + c
                rate += 1
                heart1 = heart1 + heart_bit
                heart_bit = 0
                if rate >= 400:
                  print(heart1*3)
                  rate = 0
                  heart1 = 0 """

"""peaks = detect_peaks(heart_fre)
                heart.extend(calculate_heart_rate(peak) for peak in peaks)
                #  用滑动窗口算法来计算平均心率，并输出
                if rate >= 80:
                    avg_hr = sliding_window(heart)
                    print(avg_hr)
                    rate = 0  """