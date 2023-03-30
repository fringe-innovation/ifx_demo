import argparse
import numpy as np
from scipy import signal
from scipy import constants
from collections import deque
from ifxAvian import Avian

from fft_spectrum import fft_spectrum
from Peakcure import peakcure
from Diffphase import diffphase
from IIR_Heart import iir_heart
from IIR_Breath import iir_breath
from PeakBreath import peakbreath
from PeakHeart import peakheart


class HumanPresenceAndDFFTAlgo:

    def __init__(self, config: Avian.DeviceConfig):
        self.num_samples_per_chirp = config.num_samples_per_chirp
        self.num_chirps_per_frame = config.num_chirps_per_frame

        # compute Blackman-Harris Window matrix over chirp samples(range)
        self.range_window = signal.blackmanharris(self.num_samples_per_chirp).reshape(1, self.num_samples_per_chirp)

        bandwidth_hz = abs(config.end_frequency_Hz - config.start_frequency_Hz)
        fft_size = self.num_samples_per_chirp * 2
        self.range_bin_length = constants.c / (2 * bandwidth_hz * fft_size / self.num_samples_per_chirp)

        # Algorithm Parameters
        self.detect_start_sample = self.num_samples_per_chirp // 8
        self.detect_end_sample = self.num_samples_per_chirp // 2

        self.threshold_presence = 0.1

        self.alpha_slow = 0.001
        self.alpha_med = 0.05
        self.alpha_fast = 0.6

        self.slow_avg = None
        self.fast_avg = None

        # Initialize state
        self.presence_status = False
        self.first_run = True

    def human_presence_and_dfft(self, data_in):  # sourcery skip: inline-immediately-returned-variable
        # data: single chirp data for single antenna

        # copy values into local variables to keep names short
        alpha_slow = self.alpha_slow
        alpha_med = self.alpha_med
        alpha_fast = self.alpha_fast

        # calculate range fft spectrum of the frame
        range_fft = fft_spectrum(data_in, self.range_window)

        # Average absolute FFT values over number of chirps
        fft_spec_abs = abs(range_fft)
        fft_norm = np.divide(fft_spec_abs.sum(axis=0), self.num_chirps_per_frame)

        skip = 8
        max_index = np.argmax(fft_norm[skip:])
        dist = self.range_bin_length * (max_index + skip)

        # Presence sensing
        if self.first_run:  # initialize averages
            self.slow_avg = fft_norm
            self.fast_avg = fft_norm
            self.first_run = False

        alpha_used = alpha_med if self.presence_status == False else alpha_slow
        self.slow_avg = self.slow_avg * (1 - alpha_used) + fft_norm * alpha_used
        self.fast_avg = self.fast_avg * (1 - alpha_fast) + fft_norm * alpha_fast
        data = self.fast_avg - self.slow_avg

        self.presence_status = np.max(data[self.detect_start_sample:self.detect_end_sample]) > self.threshold_presence

        return self.presence_status, range_fft


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


# sourcery skip: for-index-underscore
if __name__ == '__main__':

    args = parse_program_arguments(
        '''Derives presence and peeking information from Radar Data''',
        def_nframes=300,
        def_frate=20)

    print(f"Radar SDK Version: {Avian.get_version()}")

    config = Avian.DeviceConfig(
        sample_rate_Hz=2e6,  # ADC sample rate of 2MHz
        rx_mask=1,  # RX antenna 1 activated
        tx_mask=1,  # TX antenna 1 activated
        tx_power_level=31,  # TX power level of 31
        if_gain_dB=33,  # 33dB if gain
        start_frequency_Hz=58e9,  # start frequency: 58.0 GHz
        end_frequency_Hz=63.5e9,  # end frequency: 63.5 GHz
        num_samples_per_chirp=256,  # 256 samples per chirp
        num_chirps_per_frame=1,  # 32 chirps per frame
        chirp_repetition_time_s=0.000150,  # Chirp repetition time (or pulse repetition time) of 150us
        frame_repetition_time_s=1 / args.frate,  # Frame repetition time default 0.005s (frame rate of 200Hz)
        mimo_mode="off")  # MIMO disabled

    # connect to an Avian radar sensor
    with Avian.Device() as device:

        # metrics = device.metrics_from_config(config)

        # set device config
        device.set_config(config)
        algo = HumanPresenceAndDFFTAlgo(config)
        q = deque()
        while True:
            frame = device.get_next_frame()
            frame = frame[0, 0, :]

            q.append(frame)
            if len(q) == args.nframes:
                data = np.array(q)
                presence, dfft_data = algo.human_presence_and_dfft(data)
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

                # breath_fre = np.abs(np.fft.fftshift(np.fft.fft(breath_wave)))
                heart_fre = np.abs(np.fft.fftshift(np.fft.fft(heart_wave)))

                breath_fre = np.abs(np.fft.fft(breath_wave)) ** 2

                breath_rate, maxIndexBreathSpect = peakbreath(breath_fre)
                heart_rate = peakheart(heart_fre, maxIndexBreathSpect)

                print(f"呼吸频率：{breath_rate}, 心跳频率：{heart_rate}")
                q.pop()
