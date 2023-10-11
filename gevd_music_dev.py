import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
from src.file_io import load_signal_from_wav, write_signal_to_wav
from lib.doa import MUSIC, GevdMUSIC


def plot_music_spectrum(doa):
    estimated_angles = doa.grid.azimuth
    music_spectrum = doa.grid.values
    music_spectrum -= np.min(music_spectrum)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1, projection="polar")
    plt.polar(estimated_angles, music_spectrum)
    plt.title("MUSIC Spectrum (Polar Coordinates)")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(np.rad2deg(estimated_angles), music_spectrum)
    plt.title("MUSIC Spectrum (Cartesian Coordinates)")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Magnitude")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def generate_room_acoustics(wav_file_path, fs, sources_positions, mic_positions):
    room_dim = [4, 4]
    corners = np.array([[0, 0], [0, room_dim[1]], room_dim, [room_dim[0], 0]]).T  # [x,y]
    room = pra.Room.from_corners(corners, fs=fs, max_order=0)

    signal = load_signal_from_wav(wav_file_path, room.fs)
    samples_per_source = len(signal) // len(sources_positions)
    for i, pos in enumerate(sources_positions):
        room.add_source(pos, signal=signal[samples_per_source * i:samples_per_source * (i + 1)])

    voice = "data/raw/sample/arctic_a0001.wav"
    signal = load_signal_from_wav(voice, room.fs)
    # signalを2倍にする
    signal = np.concatenate([signal, signal])[:samples_per_source]
    room.add_source([1, 2.5], signal=signal)

    mic_array = pra.MicrophoneArray(mic_positions, room.fs)
    room.add_microphone_array(mic_array)
    room.simulate(snr=10)
    room.plot()
    return room.mic_array.signals


def generate_room_acoustics2(fs, mic_positions):
    room_dim = [4, 4]
    corners = np.array([[0, 0], [0, room_dim[1]], room_dim, [room_dim[0], 0]]).T  # [x,y]
    room = pra.Room.from_corners(corners, fs=fs, max_order=0)

    voice = "data/raw/sample/arctic_a0001.wav"
    signal = load_signal_from_wav(voice, room.fs)
    room.add_source([1, 2.5], signal=signal)

    mic_array = pra.MicrophoneArray(mic_positions, room.fs)
    room.add_microphone_array(mic_array)
    room.simulate(snr=10)
    return room.mic_array.signals


def generate_room_acoustics3(wav_file_path, fs, sources_positions, mic_positions):
    room_dim = [4, 4]
    corners = np.array([[0, 0], [0, room_dim[1]], room_dim, [room_dim[0], 0]]).T  # [x,y]
    room = pra.Room.from_corners(corners, fs=fs, max_order=0)

    signal = load_signal_from_wav(wav_file_path, room.fs)
    samples_per_source = len(signal) // len(sources_positions)
    for i, pos in enumerate(sources_positions):
        room.add_source(pos, signal=signal[samples_per_source * i:samples_per_source * (i + 1)])

    mic_array = pra.MicrophoneArray(mic_positions, room.fs)
    room.add_microphone_array(mic_array)
    room.simulate(snr=10)
    return room.mic_array.signals


def perform_fft_on_frames(signal, nfft, hop_size):
    num_frames = (signal.shape[1] - nfft) // hop_size + 1
    X = np.empty((signal.shape[0], nfft // 2 + 1, num_frames), dtype=complex)
    for t in range(num_frames):
        frame = signal[:, t * hop_size:t * hop_size + nfft]
        X[:, :, t] = np.fft.rfft(frame, n=nfft)
    return X


def main():
    wav_file_path = "data/processed/propeller/p2000_2/dst.wav"
    fs = 16000
    sources_positions = np.array([[0.5, 0.5], [2, 3.2], [3.5, 2]])
    mic_positions = pra.circular_2D_array(center=[2.,2.], M=8, phi0=0, radius=0.1)
    signal = generate_room_acoustics(wav_file_path, fs, sources_positions, mic_positions)
    signal2 = generate_room_acoustics2(fs, mic_positions)
    signal3 = generate_room_acoustics3(wav_file_path, fs, sources_positions, mic_positions)
    write_signal_to_wav(signal, "data/simulation/gevd/room2.wav", fs)

    nfft = 512
    hop_size = nfft // 2
    X = perform_fft_on_frames(signal, nfft, hop_size)
    X_noise = perform_fft_on_frames(signal3, nfft, hop_size)

    doa = GevdMUSIC(
    # doa = MUSIC(
        L=mic_positions,
        fs=fs,
        nfft=nfft,
        c=343.0,
        mode="near",
        azimuth=np.linspace(-np.pi, np.pi, 360),
        signal_noise_thresh=4,
        X_noise=X_noise,
        num_src=1
    )
    # doa.locate_sources(X, freq_range=[300, 3500], display=True, auto_identify=True)
    doa.locate_sources(X, freq_range=[300, 3500], display=True)
    plot_music_spectrum(doa)


if __name__ == "__main__":
    main()
