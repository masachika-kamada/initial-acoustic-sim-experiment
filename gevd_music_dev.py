import argparse

import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra

from lib.doa import MUSIC, GevdMUSIC
from src.file_io import load_config, load_signal_from_wav, write_signal_to_wav
from src.visualization_tools import plot_music_spectrum


class RoomConfig:
    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.room_dim = self.config["general"]["room_dim"]
        self.snr = self.config["general"]["snr"]
        self.fs = self.config["general"]["fs"]
        self.mic_positions = self.create_mic_positions()

    def create_mic_positions(self):
        mic_params = self.config["general"]["mic_positions"]
        return pra.circular_2D_array(
            center=mic_params["center"],
            M=mic_params["M"],
            phi0=mic_params["phi0"],
            radius=mic_params["radius"]
        )

    def generate_room_acoustics(self):
        corners = np.array([[0, 0], [0, self.room_dim[1]], self.room_dim, [self.room_dim[0], 0]]).T  # [x,y]
        room = pra.Room.from_corners(corners, fs=self.fs, max_order=0)
        room_noise = pra.Room.from_corners(corners, fs=self.fs, max_order=0)

        for source in self.config["source"]:
            signal = load_signal_from_wav(source["file_path"], self.fs)
            room.add_source(source["position"], signal=signal[self.fs * source["start_time"]:])

        for noise in self.config["noise"]:
            signal = load_signal_from_wav(noise["file_path"], self.fs)
            room.add_source(noise["position"], signal=signal[self.fs * noise["start_time"]:])
            room_noise.add_source(noise["position"], signal=signal[self.fs * noise["start_time"]:])

        for r in [room, room_noise]:
            mic_array = pra.MicrophoneArray(self.mic_positions, self.fs)
            r.add_microphone_array(mic_array)
            r.simulate(snr=self.snr)

        room.plot()
        plt.show()
        start = int(self.fs * self.config["general"]["start_time"])
        end = int(self.fs * self.config["general"]["end_time"])
        return room.mic_array.signals[:, start:end], room_noise.mic_array.signals[:, start:end]


def perform_fft_on_frames(signal, nfft, hop_size):
    num_frames = (signal.shape[1] - nfft) // hop_size + 1
    X = np.empty((signal.shape[0], nfft // 2 + 1, num_frames), dtype=complex)
    for t in range(num_frames):
        frame = signal[:, t * hop_size:t * hop_size + nfft]
        X[:, :, t] = np.fft.rfft(frame, n=nfft)
    return X


def main(room_config, config_dir):
    signal, signal_noise = room_config.generate_room_acoustics()
    write_signal_to_wav(signal, f"{config_dir}/simulation.wav", room_config.fs)

    nfft = 512
    hop_size = nfft // 2
    X = perform_fft_on_frames(signal, nfft, hop_size)
    X_noise = perform_fft_on_frames(signal_noise, nfft, hop_size)

    doa = GevdMUSIC(
    # doa = MUSIC(
        L=room_config.mic_positions,
        fs=room_config.fs,
        nfft=nfft,
        c=343.0,
        mode="near",
        azimuth=np.linspace(-np.pi, np.pi, 360),
        source_noise_thresh=20,
        X_noise=X_noise,
        output_dir=config_dir
    )
    doa.locate_sources(X, freq_range=[300, 3500],
                       display=False,
                       save=True,
                       auto_identify=True,
                       use_noise=True)
    plot_music_spectrum(doa, output_dir=config_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate room acoustics based on YAML config.")
    parser.add_argument("--config_dir", type=str, required=True, help="Directory containing the config.yaml file")
    args = parser.parse_args()

    config_dir = f"experiments/{args.config_dir}"
    room_config = RoomConfig(f"{config_dir}/config.yaml")

    main(room_config, config_dir)
