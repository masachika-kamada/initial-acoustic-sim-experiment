import pyroomacoustics as pra
from pyroomacoustics.doa import MUSIC
import numpy as np
import matplotlib.pyplot as plt
from src.file_io import load_signal_from_wav, write_signal_to_wav


room_dim = [4, 4]
corners = np.array([[0, 0], [0, room_dim[1]], room_dim, [room_dim[0], 0]]).T  # [x,y]

# 2次元の部屋を作成
room = pra.Room.from_corners(
    corners,
    fs=16000,
    max_order=0,
)

# 音源の位置を定義
sources_positions = np.array([[0.5, 0.5]])

# 音源を部屋に追加
signal = load_signal_from_wav("data/processed/propeller/p2000_2/dst.wav", room.fs)
samples_per_source = len(signal) // len(sources_positions)
for i, pos in enumerate(sources_positions):
    room.add_source(pos, signal=signal[samples_per_source * i:samples_per_source * (i + 1)])

# マイクロホンアレイの位置を計算
mic_positions = pra.circular_2D_array(center=[2.,2.], M=8, phi0=0, radius=0.1)

# マイクロホンアレイを部屋に追加
mic_array = pra.MicrophoneArray(mic_positions, room.fs)
room.add_microphone_array(mic_array)

room.simulate(snr=10)
simulated_signals = room.mic_array.signals

# FFTの長さとホップサイズを定義
nfft = 512
hop_size = nfft // 2

# シミュレートされた信号をフレームに分割し、FFTを計算
num_frames = (simulated_signals.shape[1] - nfft) // hop_size + 1
X = np.empty((simulated_signals.shape[0], nfft // 2 + 1, num_frames), dtype=complex)

for t in range(num_frames):
    frame = simulated_signals[:, t*hop_size:t*hop_size+nfft]
    X[:, :, t] = np.fft.rfft(frame, n=nfft)

num_mics = X.shape[0]  # マイクの数

plt.figure(figsize=(16, 4 * num_mics))

for mic_index in range(num_mics):
    # np.fft.rfft によるスペクトログラム
    plt.subplot(num_mics, 2, 2*mic_index + 1)
    plt.title(f"Spectrogram for Microphone {mic_index + 1} (np.fft.rfft)")
    plt.imshow(20 * np.log10(np.abs(X[mic_index, :, :]) + 1e-6), aspect="auto", cmap="inferno", origin="lower")
    plt.colorbar().set_label("Intensity [dB]")
    plt.xlabel("Time Frame")
    plt.ylabel("Frequency Bin")

    # plt.specgram によるスペクトログラム
    plt.subplot(num_mics, 2, 2*mic_index + 2)
    plt.title(f"Spectrogram for Microphone {mic_index + 1} (plt.specgram)")
    spectrum, freqs, t, im = plt.specgram(simulated_signals[mic_index, :], NFFT=512, noverlap=int(512 / 16 * 15), Fs=16000, cmap="inferno")
    plt.colorbar(im).set_label("Intensity [dB]")
    plt.xlabel("Time [sec]")
    plt.ylabel("Frequency [Hz]")

plt.tight_layout()
plt.savefig("spectrogram.png")
