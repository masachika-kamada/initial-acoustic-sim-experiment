import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
from lib.doa import MUSIC, GevdMUSIC
from src.file_io import load_config, load_signal_from_wav, write_signal_to_wav
from src.visualization_tools import plot_music_spectrum


def create_mic_positions(mic_params):
    return pra.circular_2D_array(
        center=mic_params["center"],
        M=mic_params["M"],
        phi0=mic_params["phi0"],
        radius=mic_params["radius"]
    )


def generate_room_acoustics(config, output_dir):
    room_dim = config["general"]["room_dim"]
    fs = config["general"]["fs"]
    snr = config["general"]["snr"]
    corners = np.array([[0, 0], [0, room_dim[1]], room_dim, [room_dim[0], 0]]).T  # [x, y]
    room = pra.Room.from_corners(corners, fs=fs, max_order=0)
    room_noise = pra.Room.from_corners(corners, fs=fs, max_order=0)
    mic_positions = create_mic_positions(config["general"]["mic_positions"])

    for source in config["source"]:
        signal = load_signal_from_wav(source["file_path"], fs)
        room.add_source(source["position"], signal=signal[fs * source["start_time"] :])

    for noise in config["noise"]:
        signal = load_signal_from_wav(noise["file_path"], fs)
        room.add_source(noise["position"], signal=signal[fs * noise["start_time"] :])
        room_noise.add_source(noise["position"], signal=signal[fs * noise["start_time"] :])

    for r in [room, room_noise]:
        mic_array = pra.MicrophoneArray(mic_positions, fs)
        r.add_microphone_array(mic_array)
        r.simulate(snr=snr)

    room.plot()
    plt.savefig(f"{output_dir}/room.png")
    plt.close()
    start = int(fs * config["general"]["start_time"])
    end = int(fs * config["general"]["end_time"])
    return room.mic_array.signals[:, start:end], room_noise.mic_array.signals[:, start:end], mic_positions


def create_doa_object(method, source_noise_thresh, mic_positions, fs, nfft, X_noise, output_dir):
    common_params = {
        "L": mic_positions,
        "fs": fs,
        "nfft": nfft,
        "c": 343.0,
        "mode": "far",
        "azimuth": np.linspace(-np.pi, np.pi, 360),
        "source_noise_thresh": source_noise_thresh,
        "output_dir": output_dir,
    }
    if method == "MUSIC":
        doa = MUSIC(**common_params)
    elif method == "GEVD-MUSIC":
        doa = GevdMUSIC(**common_params, X_noise=X_noise)
    else:
        raise ValueError(f"Unknown method: {method}")
    return doa


def perform_fft_on_frames(signal, nfft, hop_size):
    num_frames = (signal.shape[1] - nfft) // hop_size + 1
    X = np.empty((signal.shape[0], nfft // 2 + 1, num_frames), dtype=complex)
    for t in range(num_frames):
        frame = signal[:, t * hop_size : t * hop_size + nfft]
        X[:, :, t] = np.fft.rfft(frame, n=nfft)
    return X


def main(config, output_dir):
    signal, signal_noise, mic_positions = generate_room_acoustics(config, output_dir)
    write_signal_to_wav(signal, f"{output_dir}/simulation.wav", config["general"]["fs"])

    nfft = config["general"]["nfft"]
    hop_size = config["general"]["hop_size"]
    freq_range = config["general"]["freq_range"]

    X = perform_fft_on_frames(signal, nfft, hop_size)
    X_noise = perform_fft_on_frames(signal_noise, nfft, hop_size)

    doa_config = config["doa"]
    doa = create_doa_object(
        method=doa_config["method"],
        source_noise_thresh=doa_config["source_noise_thresh"],
        mic_positions=mic_positions,
        fs=config["general"]["fs"],
        nfft=nfft,
        X_noise=X_noise,
        output_dir=output_dir,
    )
    doa.locate_sources(X, freq_range=freq_range, auto_identify=True, use_noise=True)
    plot_music_spectrum(doa, output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate room acoustics based on YAML config.")
    parser.add_argument("--config_dir", type=str, required=True, help="Directory containing the config.yaml file")
    args = parser.parse_args()

    config_dir = f"experiments/{args.config_dir}"
    config = load_config(f"{config_dir}/config.yaml")
    output_dir = f"{config_dir}/output"
    os.makedirs(output_dir, exist_ok=True)

    main(config, output_dir)
