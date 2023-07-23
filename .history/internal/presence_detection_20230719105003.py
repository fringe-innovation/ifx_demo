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
from ifxAvian import Avian
from fft_spectrum import *
import numpy as np
from scipy import signal
from collections import namedtuple

class PresenceAntiPeekingAlgo:
    def __init__(self, num_samples_per_chirp, num_chirps_per_frame):
        """Presence and Anti-Peeking Algorithm
        
        This is a simple use case of an a presence detection and
        anti-peeking demo.
        
        Parameters:
        num_samples_per_chirp: Number of samples per chirp
        """
        self.num_samples_per_chirp = num_samples_per_chirp
        self.num_chirps_per_frame = num_chirps_per_frame

        # Algorithm Parameters
        self.detect_start_sample = num_samples_per_chirp//8
        self.detect_end_sample = num_samples_per_chirp
        self.peek_start_sample = num_samples_per_chirp//2
        self.peek_end_sample = num_samples_per_chirp

        self.threshold_presence = 0.0007
        self.threshold_peeking = 0.0006

        self.alpha_slow = 0.001
        self.alpha_med = 0.05
        self.alpha_fast = 0.6

        # Initialize state
        self.presence_status = False
        self.peeking_status = False
        self.first_run = True
        
        # Use Blackmann-Harris as window function
        self.window = signal.blackmanharris(num_samples_per_chirp).reshape(1,num_samples_per_chirp)

    def presence(self, mat):
        """Run the presence and anti-peeking algorithm on the current frame.
        
        Parameters:
            - mat: Radar data for one antenna as returned by Frame.get_mat_from_antenna
        
        Returns:
            - Tuple consisting of the state for presence detection and peeking.
              The first bool indicates if a target was detected. The second bool
              indicates if peeking was detected.
        """
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

        if self.presence_status == False:
            alpha_used = alpha_med
        else:
            alpha_used = alpha_slow

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
