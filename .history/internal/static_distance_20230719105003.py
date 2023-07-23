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

from fft_spectrum import *


# -------------------------------------------------
# Computation
# -------------------------------------------------
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
