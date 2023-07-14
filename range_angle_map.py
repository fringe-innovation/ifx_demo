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
import matplotlib.pyplot as plt
import numpy as np

from ifxAvian import Avian
from fft_spectrum import *
from DBF import DBF
from doppler import DopplerAlgo


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



