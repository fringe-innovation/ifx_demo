import argparse
import pprint
import numpy as np
import math
from scipy import signal
from scipy import constants
from ifxAvian import Avian

from fft_spectrum import fft_spectrum
from DBF import DBF
from doppler import DopplerAlgo
from Peakcure import peakcure
from Diffphase import diffphase
from IIR_Heart import iir_heart
from IIR_Breath import iir_breath
from scipy.signal import argrelextrema
from presence_detection import PresenceAntiPeekingAlgo


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
    def human_angle(self, data_in):
        
        return 


def num_rx_antennas_from_config(config):
    rx_mask = config.rx_mask

    # popcount for rx_mask
    c = 0
    for i in range(32):
        if rx_mask & (1 << i):
            c += 1
    return c

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



#用于单次计算给定窗口长度的均值滤波
def ava_filter(x, filt_length):
    N = len(x)
    res = []
    for i in range(N):
        if i <= filt_length // 2 or i >= N - (filt_length // 2):
            temp = x[i]
        else:
            sum = 0
            for j in range(filt_length):
                sum += x[i - filt_length // 2 + j]
            temp = sum * 1.0 / filt_length
        res.append(temp)
    return res

#函数denoise用于指定次数调用ava_filter函数，进行降噪处理
def denoise( x, n, filt_length):
    for _ in range(n):
        res = ava_filter(x, filt_length)
        x = res
    return res

def f(x):
    return 76 + 19 * x


if __name__ == '__main__':
    args = parse_program_arguments(
        '''Derives presence and peeking information from Radar Data''',
        def_nframes=256,
        def_frate=20)
    num_beams = 27         # number of beams
    max_angle_degrees = 40  # maximum angle, angle ranges from -40 to +40 degrees
    print(f"Radar SDK Version: {Avian.get_version()}")

    config = Avian.DeviceConfig(
        sample_rate_Hz=2e6,  # ADC sample rate of 2MHz
        rx_mask=5,  # RX antenna 3 activated
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
        rate = 0
        np_time = 0
        heart_bit = 0
        heart = []
        breath = []
        data = []
        yaw = []
        device.set_config(config)
        algo = HumanPresenceAndDFFTAlgo(config)
        # get metrics and print them
        metrics = device.metrics_from_config(config)
        pprint.pprint(metrics)
        # get maximum range
        max_range_m = metrics.max_range_m
        # Create frame handle
        num_rx_antennas = num_rx_antennas_from_config(config)
        # Create objects for Range-Doppler, DBF, and plotting.
        doppler = DopplerAlgo(config, num_rx_antennas)
        dbf = DBF(num_rx_antennas, num_beams=num_beams,
                  max_angle_degrees=max_angle_degrees)
        presence = PresenceAntiPeekingAlgo(config.num_samples_per_chirp, config.num_chirps_per_frame)
        num_frame = args.nframes
        num_chirp = config.num_chirps_per_frame
        num_sample = config.num_samples_per_chirp
        
        for _ in range(8000):
            frame = device.get_next_frame()
            rd_spectrum = np.zeros(
                (config.num_samples_per_chirp, 2*config.num_chirps_per_frame, num_rx_antennas), dtype=complex)

            beam_range_energy = np.zeros(
                (config.num_samples_per_chirp, num_beams))
            for i_ant in range(num_rx_antennas):  # For each antenna
                # Current RX antenna (num_samples_per_chirp x num_chirps_per_frame)
                data_p = frame[0, :, :]
                mat = frame[i_ant, :, :]
                presence_status, peeking_status = presence.presence(data_p)

                # Compute Doppler spectrum
                dfft_dbfs = doppler.compute_doppler_map(mat, i_ant)
                rd_spectrum[:, :, i_ant] = dfft_dbfs

            # Compute Range-Angle map
            rd_beam_formed = dbf.run(rd_spectrum)
            for i_beam in range(num_beams):
                doppler_i = rd_beam_formed[:, :, i_beam]
                beam_range_energy[:, i_beam] += np.linalg.norm(
                    doppler_i, axis=1) / np.sqrt(num_beams)
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
            

            # Maximum energy in Range-Angle map
            max_energy = np.max(beam_range_energy)
            scale = 150
            beam_range_energy = scale*(beam_range_energy/max_energy - 1)
            # Find dominant angle of target
            _, idx = np.unravel_index(
                beam_range_energy.argmax(), beam_range_energy.shape)
            angle=math.acos(idx/30)*180/3.14
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
                heart_peaks = argrelextrema(array_heart, np.greater, order=15)
                breath_peaks = argrelextrema(array_breath, np.greater, order=20)

                heart_peaks = heart_peaks[0][array_heart[heart_peaks] > f(dist)]
                heart_count = len(heart_peaks)
                heart.append(heart_count)

                breath_peaks = breath_peaks[0][array_breath[breath_peaks] >(f(dist)+20)]
                breath_count = len(breath_peaks)
                breath.append(breath_count)
                rate = rate + 1
                times = np.linspace(0, 60, num_frame)

                if rate >= 800:
                    heart_bit = sum(heart)
                    breath_bit = sum(breath)
                    heart.pop(0)
                    breath.pop(0)
                    if rate % 40 == 0:
                        print(breath_bit)
                        print(heart_bit)
                        rate = 800  
            else:
                np_time = np_time + 1
                if np_time > 30:
                    print("no person")
                    print(angle-30)
                    np_time = 0  
            data = []
