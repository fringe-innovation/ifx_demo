# ===========================================================================
# Copyright (C) 2022 Infineon Technologies AG
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ===========================================================================

import pprint
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy import constants

from ifxAvian import Avian
from fft_spectrum import *
from DBF import DBF
from doppler import DopplerAlgo
from presence_detection import PresenceAntiPeekingAlgo
from collections import namedtuple

from Peakcure import peakcure
from Diffphase import diffphase
from IIR_Heart import iir_heart
from IIR_Breath import iir_breath

def num_rx_antennas_from_config(config):
    rx_mask = config.rx_mask

    # popcount for rx_mask
    c = 0
    for i in range(32):
        if rx_mask & (1 << i):
            c += 1
    return c


class LivePlot:
    def __init__(self, max_angle_degrees: float, max_range_m: float):
        # max_angle_degrees: maximum supported speed
        # max_range_m:   maximum supported range
        self.h = None
        self.max_angle_degrees = max_angle_degrees
        self.max_range_m = max_range_m

        self._fig, self._ax = plt.subplots(nrows=1, ncols=1)

        self._fig.canvas.manager.set_window_title("Range-Angle-Map using DBF")
        self._fig.canvas.mpl_connect('close_event', self.close)
        self._is_window_open = True

    def _draw_first_time(self, data: np.ndarray):
        # First time draw

        minmin = -60
        maxmax = 0

        self.h = self._ax.imshow(
            data,
            vmin=minmin, vmax=maxmax,
            cmap='viridis',
            extent=(-self.max_angle_degrees,
                    self.max_angle_degrees,
                    0,
                    self.max_range_m),
            origin='lower')

        self._ax.set_xlabel("angle (degrees)")
        self._ax.set_ylabel("distance (m)")
        self._ax.set_aspect("auto")

        self._fig.subplots_adjust(right=0.8)
        cbar_ax = self._fig.add_axes([0.85, 0.0, 0.03, 1])    

        cbar = self._fig.colorbar(self.h, cax=cbar_ax)
        cbar.ax.set_ylabel("magnitude (a.u.)")

    def _draw_next_time(self, data: np.ndarray):
        # Update data for each antenna

        self.h.set_data(data)

    def draw(self, data: np.ndarray, title: str):
        if self._is_window_open:
            if self.h:
                self._draw_next_time(data)
            else:
                self._draw_first_time(data)
            self._ax.set_title(title)

            # Needed for Matplotlib ver: 3.4.0 and 3.4.1 helps with capture closing event
            plt.draw()
            plt.pause(1e-3)

    def close(self, event=None):
        if not self.is_closed():
            self._is_window_open = False
            plt.close(self._fig)
            plt.close('all')
            print('Application closed!')

    def is_closed(self):
        return not self._is_window_open



def presence(self, mat):
    # copy values into local variables to keep names short
    alpha_slow = self.alpha_slow
    alpha_med = self.alpha_med
    alpha_fast = self.alpha_fast

    # Compute range FFT
    range_fft = fft_spectrum(mat, self.window)

    # Average absolute FFT values over number of chirps
    fft_spec_abs = abs(range_fft)
    fft_norm = np.divide(fft_spec_abs.sum(axis=0), self.num_chirps_per_frame)

    # Presence sensing
    if self.first_run: # initialize averages
        self.slow_avg = fft_norm
        self.fast_avg = fft_norm
        self.slow_peek_avg = fft_norm
        self.first_run = False

    alpha_used = alpha_med if self.presence_status == False else alpha_slow
    self.slow_avg = self.slow_avg*(1-alpha_used) + fft_norm*alpha_used
    self.fast_avg = self.fast_avg*(1-alpha_fast) + fft_norm*alpha_fast
    data = self.fast_avg-self.slow_avg

    self.presence_status = np.max(data[self.detect_start_sample:self.detect_end_sample]) > self.threshold_presence

    # Peeking sensing
    if self.peeking_status == False:
        alpha_used = self.alpha_med
    else:
        alpha_used = self.alpha_slow

    self.slow_peek_avg = self.slow_peek_avg*(1-alpha_used) + fft_norm*alpha_used
    data_peek = self.fast_avg-self.slow_peek_avg

    self.peeking_status = np.max(data_peek[self.peek_start_sample:self.peek_end_sample]) > self.threshold_peeking

    return namedtuple("state", ["presence", "peeking"])(self.presence_status, self.peeking_status)



class DistanceAlgo:
    # Algorithm for computing distance

    def __init__(self, config):
        # Inits all needed common values
        # config: dictionary with configuration for device used by set_config() as input
        
        # derive the number of Chirps, Samples per chirp from frame shape
        self._chirps_per_frame = config.num_chirps_per_frame
        self._samples_per_chirp = config.num_samples_per_chirp

        # compute Blackman-Harris Window matrix over chirp samples(range)
        self._range_window = signal.blackmanharris(
            self._samples_per_chirp).reshape(1, self._samples_per_chirp)

        bandwidth_hz = abs(config.end_frequency_Hz - config.start_frequency_Hz)

        fft_size = self._samples_per_chirp * 2
        self._range_bin_length = (constants.c) / (2 * bandwidth_hz * fft_size / self._samples_per_chirp)

    def compute_distance(self, chirp_data):
        # Computes distance using chirp data
        # chirp_data: single antenna chirp data

        # Step 1 - calculate fft spectrum for the frame
        range_fft = fft_spectrum(chirp_data, self._range_window)

        # Step 2 - Convert to absolute spectrum
        range_fft_abs = abs(range_fft)

        # Step 3 - Coherent integration of all chirps
        dat = np.divide(range_fft_abs.sum(axis=0), self._chirps_per_frame)

        # Step 4 - Peak search and distance calculation
        skip = 8
        max = np.argmax(dat[skip:])
        return (self._range_bin_length * (max + skip))



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



# -------------------------------------------------
# Main logic
# -------------------------------------------------
if __name__ == '__main__':
    num_beams = 27         # number of beams
    max_angle_degrees = 45  # maximum angle, angle ranges from -40 to +40 degrees

    config = Avian.DeviceConfig(
        sample_rate_Hz=1_000_000,       # 1MHZ
        rx_mask=5,                      # activate RX1 and RX3
        tx_mask=1,                      # activate TX1
        if_gain_dB=33,                  # gain of 33dB
        tx_power_level=31,              # TX power level of 31
        start_frequency_Hz=60e9,        # 60GHz
        end_frequency_Hz=61.5e9,        # 61.5GHz
        num_chirps_per_frame=128,       # 128 chirps per frame
        num_samples_per_chirp=64,       # 64 samples per chirp
        chirp_repetition_time_s=0.0005,  # 0.5ms
        frame_repetition_time_s=0.15,   # 0.15s, frame_Rate = 6.667Hz
        mimo_mode='off'                 # MIMO disabled
    )

    with Avian.Device() as device:
        # set configuration
        device.set_config(config)
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
        distanceExample = DistanceAlgo(config)
        plot = LivePlot(max_angle_degrees, max_range_m)

        #while not plot.is_closed():
        for frame_number in range(100):
            # frame has dimension num_rx_antennas x num_samples_per_chirp x num_chirps_per_frame
            frame = device.get_next_frame()
            status = np.zeros(num_rx_antennas, dtype=bool)
            rd_spectrum = np.zeros(
                (config.num_samples_per_chirp, 2*config.num_chirps_per_frame, num_rx_antennas), dtype=complex)

            beam_range_energy = np.zeros(
                (config.num_samples_per_chirp, num_beams))
            for i_ant in range(num_rx_antennas):  # For each antenna
                # Current RX antenna (num_samples_per_chirp x num_chirps_per_frame)
                data = frame[0, :, :]
                mat = frame[i_ant, :, :]
                presence_status, peeking_status = presence.presence(data)

                # Compute Doppler spectrum
                dfft_dbfs = doppler.compute_doppler_map(mat, i_ant)
                rd_spectrum[:, :, i_ant] = dfft_dbfs

            # Compute Range-Angle map
            rd_beam_formed = dbf.run(rd_spectrum)
            for i_beam in range(num_beams):
                doppler_i = rd_beam_formed[:, :, i_beam]
                beam_range_energy[:, i_beam] += np.linalg.norm(
                    doppler_i, axis=1) / np.sqrt(num_beams)

            # Maximum energy in Range-Angle map
            max_energy = np.max(beam_range_energy)

            scale = 150
            beam_range_energy = scale*(beam_range_energy/max_energy - 1)

            # Find dominant angle of target
            _, idx = np.unravel_index(
                beam_range_energy.argmax(), beam_range_energy.shape)
            angle_degrees = np.linspace(-max_angle_degrees,
                                        max_angle_degrees, num_beams)[idx]
            distance = distanceExample.compute_distance(data)

            # And plot...
            #plot.draw(beam_range_energy, f"Range-Angle map using DBF, angle={angle_degrees:+02.0f} degrees")
            print(f" Presence: {presence_status}")
            #print("Distance:" + format(distance, "^05.3f") + "m")
            print(f"Angle: {angle_degrees} degrees")
        plot.close()


