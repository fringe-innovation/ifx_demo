from ifxAvian import Avian

import numpy as np
# from scipy import signal
# import pprint


# from examples.internal.fft_spectrum import *
from DBF import DBF
from doppler import DopplerAlgo
from presence_detection import PresenceAntiPeekingAlgo
from range_angle_map import num_rx_antennas_from_config


if __name__ == "__main__":

    num_beams = 27         # number of beams
    max_angle_degrees = 45  # maximum angle, angle ranges from -45 to +45 degrees

    print(f"Radar SDK Version: {Avian.get_version_full()}")

    config = Avian.DeviceConfig(
        sample_rate_Hz=2e6,                   # ADC sample rate of 1MHz
        rx_mask=5,                            # RX antenna 1 activated
        tx_mask=1,                            # TX antenna 1 and 3 activated
        tx_power_level=31,                    # TX power level of 31
        if_gain_dB=33,                        # 33dB if gain
        start_frequency_Hz=58e9,              # start frequency: 58 GHz
        end_frequency_Hz=63.5e9,              # end frequency: 63.5 GHz
        num_samples_per_chirp=128,            # 128 samples per chirp
        num_chirps_per_frame=64,              # 64 chirps per frame
        # Chirp repetition time (or pulse repetition time) of 150us
        chirp_repetition_time_s=0.000150,
        # Frame repetition time default 0.1s (frame rate of 10Hz)
        frame_repetition_time_s=0.1,
        mimo_mode="off")                      # MIMO disabled

    with Avian.Device() as device:
        # set device config for presence sensing
        device.set_config(config)

        # get metrics and print them
        metrics = device.metrics_from_config(config)
        # pprint.pprint(metrics)

        # get maximum range
        max_range_m = metrics.max_range_m

        # Create frame handle
        num_rx_antennas = num_rx_antennas_from_config(config)

        # Create objects for Range-Doppler, DBF, and plotting.
        doppler = DopplerAlgo(config, num_rx_antennas)
        dbf = DBF(num_rx_antennas, num_beams=num_beams,
                max_angle_degrees=max_angle_degrees)
        presence = PresenceAntiPeekingAlgo(
            config.num_samples_per_chirp, config.num_chirps_per_frame)

        while True:
            frame_data = device.get_next_frame()

            status = np.zeros(num_rx_antennas, dtype=bool)

            rd_spectrum = np.zeros(
                (config.num_samples_per_chirp, 2*config.num_chirps_per_frame, num_rx_antennas), dtype=complex)

            beam_range_energy = np.zeros(
                (config.num_samples_per_chirp, num_beams))

            for i_ant in range(num_rx_antennas):    # For each antenna:
                data = frame_data[i_ant, :, :]
                presence_status, peeking_status = presence.presence(data)
                status[i_ant] = presence_status     ###

                # Compute Doppler spectrum
                dfft_dbfs = doppler.compute_doppler_map(data, i_ant)
                rd_spectrum[:, :, i_ant] = dfft_dbfs
    
            if np.any(status):

                # Compute Range-Angle map
                rd_beam_formed = dbf.run(rd_spectrum)
                for i_beam in range(num_beams):
                    doppler_i = rd_beam_formed[:, :, i_beam]
                    beam_range_energy[:, i_beam] += np.linalg.norm(
                        doppler_i, axis=1) / np.sqrt(num_beams)

                # Maximum energy in Range-Angle map
                max_energy = np.max(beam_range_energy)

                # Rescale map to better capture the peak The rescaling is done in a
                # way such that the maximum always has the same value, independent
                # on the original input peak. A proper peak search can greatly
                # improve this algorithm.
                scale = 150
                beam_range_energy = scale*(beam_range_energy/max_energy - 1)

                # Find dominant angle of target
                _, idx = np.unravel_index(
                    beam_range_energy.argmax(), beam_range_energy.shape)
                angle_degrees = np.linspace(-max_angle_degrees,
                                            max_angle_degrees, num_beams)[idx]
                print(f"Angle: {angle_degrees} degrees")

            else:
                print("No presence detected")
