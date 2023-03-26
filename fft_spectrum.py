# ===========================================================================
# Copyright (C) 2021 Infineon Technologies AG
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

import numpy as np

def fft_spectrum(mat, range_window):
    # Calculate fft spectrum
    # mat:          chirp data
    # range_window: window applied on input data before fft

    # received data 'mat' is in matrix form for a single receive antenna
    # each row contains 'chirpsamples' samples for a single chirp
    # total number of rows = 'numchirps'

    # -------------------------------------------------
    # Step 1 - remove DC bias from samples
    # -------------------------------------------------
    [numchirps, chirpsamples] = np.shape(mat)

    # helpful in zero padding for high resolution FFT.
    # compute row (chirp) averages
    avgs = np.average(mat, 1).reshape(numchirps, 1)

    #de-bias values
    mat = mat - avgs
    # -------------------------------------------------
    # Step 2 - Windowing the Data
    # -------------------------------------------------
    mat = np.multiply(mat, range_window)

    # -------------------------------------------------
    # Step 3 - add zero padding here
    # -------------------------------------------------
    zp1 = np.pad(mat, ((0, 0), (0, chirpsamples)), 'constant')

    # -------------------------------------------------
    # Step 4 - Compute FFT for distance information
    # -------------------------------------------------
    range_fft = np.fft.fft(zp1)/chirpsamples

    #ignore the redundant info in negative spectrum
    #compensate energy by doubling magnitude
    range_fft = 2*range_fft[:, range(int(chirpsamples))]

    return range_fft
        