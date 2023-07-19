# ===========================================================================
# Copyright (C) 2021-2022 Infineon Technologies AG
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

import argparse
import numpy as np

from ifxAvian import Avian
from scipy import signal
from scipy import constants
import matplotlib.pyplot as plt

from internal.fft_spectrum import *
# -------------------------------------------------
# Computation
# -------------------------------------------------
class DistanceFFT_Algo:
    # Algorithm for computation of distance fft from raw data

    def __init__(self, config : Avian.DeviceConfig):
        # Common values initiation
        # cfg: dictionary with configuration for device used by set_config() as input

        self._numchirps = config.num_chirps_per_frame
        chirpsamples = config.num_samples_per_chirp

        # compute Blackman-Harris Window matrix over chirp samples(range)
        self._range_window = signal.blackmanharris(chirpsamples).reshape(1, chirpsamples)

        start_frequency_Hz = config.start_frequency_Hz
        end_frequency_Hz = config.end_frequency_Hz
        bandwidth_hz = abs(end_frequency_Hz-start_frequency_Hz)
        fft_size = chirpsamples * 2
        self._range_bin_length = (constants.c) / (2 * bandwidth_hz * fft_size / chirpsamples)

    def compute_distance(self, data):
        # Computes a distance for one chirp of data
        # data: single chirp data for single antenna

        # Step 1 - calculate range fft spectrum of the frame
        range_fft = fft_spectrum(data, self._range_window)

        # Step 2 - convert to absolute spectrum
        fft_spec_abs = abs(range_fft)

        # Step 3 - coherent integration of all chirps
        data = np.divide(fft_spec_abs.sum(axis=0), self._numchirps)

        # Step 4 - peak search and distance calculation
        skip = 8
        max = np.argmax(data[skip:])

        dist = self._range_bin_length * (max + skip)
        return dist, data

