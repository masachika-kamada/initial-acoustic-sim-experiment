import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
from lib.doa import MUSIC, GevdMUSIC, GsvdMUSIC
from src.file_io import load_config, load_signal_from_wav, write_signal_to_wav
from src.visualization_tools import plot_music_spectrum


def create_mic_positions(mic_params):
    return pra.circular_2D_array(
        center=mic_params["center"],
        M=mic_params["M"],
        phi0=mic_params["phi0"],
        radius=mic_params["radius"]
    )


def create_materials(config):
    absorption = config["absorption"]
    m = config["material"]
    max_order = config["max_order"]
    if m is None:
        m = pra.Material(energy_absorption=absorption)
    else:
        m = pra.Material(m)
    return m, max_order


def generate_room_acoustics(config, output_dir):
    room_dim = config["room"]["room_dim"]
    fs = config["room"]["fs"]
    snr = config["room"]["snr"]
    corners = np.array([[0, 0], [0, room_dim[1]], room_dim, [room_dim[0], 0]]).T  # [x, y]

    rooms = []
    m, max_order = create_materials(config["room"]["source"])
    rooms.append(pra.Room.from_corners(corners, fs=fs, max_order=max_order, materials=m))
    noise_config = config.get("noise", [])
    if len(noise_config) > 0:
        m, max_order = create_materials(config["room"]["noise"])
        rooms.append(pra.Room.from_corners(corners, fs=fs, max_order=max_order, materials=m))

    for source in config["source"]:
        signal = load_signal_from_wav(source["file_path"], fs)
        rooms[0].add_source(source["position"], signal=signal[fs * source["start_time"] :])

    for noise in noise_config:
        signal = load_signal_from_wav(noise["file_path"], fs)
        rooms[0].add_source(noise["position"], signal=signal[fs * noise["start_time"] :])
        rooms[1].add_source(noise["position"], signal=signal[fs * noise["start_time"] :])

    mic_positions = create_mic_positions(config["mic_positions"])
    for r in rooms:
        mic_array = pra.MicrophoneArray(mic_positions, fs)
        r.add_microphone_array(mic_array)
        r.simulate(snr=snr)
    rooms[0].plot()
    plt.savefig(f"{output_dir}/room.png")
    plt.close()

    start = int(fs * config["processing"]["start_time"])
    end = int(fs * config["processing"]["end_time"])
    signal = rooms[0].mic_array.signals[:, start:end]
    signal_noise = rooms[1].mic_array.signals[:, start:end] if len(noise_config) > 0 else None
    return signal, signal_noise, mic_positions


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
    elif method == "GSVD-MUSIC":
        doa = GsvdMUSIC(**common_params, X_noise=X_noise)
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
    pra_config = config["pra"]
    signal, signal_noise, mic_positions = generate_room_acoustics(pra_config, output_dir)
    write_signal_to_wav(signal, f"{output_dir}/simulation.wav", pra_config["room"]["fs"])

    fft_config = config["fft"]
    window_size = fft_config["window_size"]
    hop_size = fft_config["hop_size"]
    X = perform_fft_on_frames(signal, window_size, hop_size)
    X_noise = perform_fft_on_frames(signal_noise, window_size, hop_size) if signal_noise is not None else None

    doa_config = config["doa"]
    doa = create_doa_object(
        method=doa_config["method"],
        source_noise_thresh=doa_config["source_noise_thresh"],
        mic_positions=mic_positions,
        fs=pra_config["room"]["fs"],
        nfft=window_size,
        X_noise=X_noise,
        output_dir=output_dir,
    )
    doa.locate_sources(X, freq_range=doa_config["freq_range"], auto_identify=True)
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
