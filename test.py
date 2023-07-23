import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import constants
from ifxAvian import Avian

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

        skip = 20
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


if __name__ == '__main__':
    args = parse_program_arguments(
        '''Derives presence and peeking information from Radar Data''',
        def_nframes=256,
        def_frate=20)

    print(f"Radar SDK Version: {Avian.get_version()}")

    config = Avian.DeviceConfig(
        sample_rate_Hz=2e6,  # ADC sample rate of 2MHz
        rx_mask=2,  # RX antenna 1 activated
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
        heart_bit = 0
        data = []
        for frame_number in range(args.nframes):
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
            #ax = plt.figure(1).add_subplot(projection='3d')
            #ax.plot_wireframe(X, Y, db(dfft_data[:num_sample - 150, :]))
            #ax.set_title('1dfft')
            #ax.set_xlabel('t/s')
            #ax.set_ylabel('range(m)')
            #plt.show()
            if 0.3 < dist < 0.9:
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
                times = np.linspace(0, 60, num_frame)
                y = argrelextrema(heart_wave, np.greater, order=8)
                c = 0
                for i in range(20):
                    for i in range(len(y[0]) - 1):
                        c += 1   
                
                    c += heart_bit
                heart_bit = c % 4
                print(heart_bit)
                plt.figure(1)
                plt.subplot(2, 1, 1)
                plt.plot(times, breath_wave)
                plt.title('Respiratory waveform')
                plt.xlabel('t/s')
                plt.ylabel('dB')
                plt.subplot(2, 1, 2)
                plt.plot(times, heart_wave)
                plt.title('Heart waveform')
                plt.xlabel('t/s')
                plt.ylabel('dB') 
                plt.show()
            data = []
                    




"""  elif dist < 0.3:
                print('被遮挡')
            else:
                print('当前位置无人') """
