import os
import wave

import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile
from scipy.signal import resample_poly

from src.audio_processing import scale_signal


def convert_to_wav(audio_file_path):
    # ファイル拡張子を抽出（".mp3", ".m4a", ".ogg"など）
    file_ext = os.path.splitext(audio_file_path)[1][1:]

    audio = AudioSegment.from_file(audio_file_path, file_ext)

    wav_file_path = os.path.splitext(audio_file_path)[0] + ".wav"
    audio.export(wav_file_path, format="wav")
    return wav_file_path


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def write_signal_to_wav(signal, file_path, sample_rate):
    ensure_dir(file_path)
    signal = scale_signal(signal)

    # チャンネル数をチェック
    if len(signal.shape) == 1:
        channels = 1
    else:
        channels = signal.shape[0]
        signal = signal.T.flatten()

    with wave.open(file_path, "w") as wave_out:
        wave_out.setnchannels(channels)
        wave_out.setsampwidth(2)
        wave_out.setframerate(sample_rate)
        wave_out.writeframes(signal.tobytes())


def write_signal_to_npz(signal, file_path, sample_rate):
    ensure_dir(file_path)
    np.savez(file_path, signal=signal, sample_rate=sample_rate)


def load_signal_from_npz(file_path):
    data = np.load(file_path)
    return data["signal"], data["sample_rate"]


# パスから音源を読み込む関数
def load_signal_from_wav(file_path, expected_fs):
    fs, signal = wavfile.read(file_path)
    if fs != expected_fs:
        # リサンプリングするためのアップサンプリングとダウンサンプリングの比率を計算
        up = expected_fs
        down = fs
        # リサンプリング
        signal = signal.astype(float)
        signal = resample_poly(signal, up, down)
    return signal
