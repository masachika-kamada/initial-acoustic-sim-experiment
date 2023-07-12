from pydub import AudioSegment
import os
import numpy as np
import wave
from src.audio_processing import scale_signal


def convert_to_wav(audio_file_path):
    # ファイル拡張子を抽出（'.mp3', '.m4a', '.ogg'など）
    file_ext = os.path.splitext(audio_file_path)[1][1:]

    audio = AudioSegment.from_file(audio_file_path, file_ext)

    wav_file_path = os.path.splitext(audio_file_path)[0] + ".wav"
    audio.export(wav_file_path, format="wav")
    return wav_file_path


def write_signal_to_wav(signal, file_name, sample_rate):
    signal = scale_signal(signal)

    # チャンネル数をチェック
    if len(signal.shape) == 1:
        channels = 1
    else:
        channels = signal.shape[0]
        signal = signal.T.flatten()

    with wave.open(file_name, "w") as wave_out:
        wave_out.setnchannels(channels)
        wave_out.setsampwidth(2)
        wave_out.setframerate(sample_rate)
        wave_out.writeframes(signal.tobytes())


def write_signal_to_npz(signal, file_name, sample_rate):
    np.savez(file_name, signal=signal, sample_rate=sample_rate)


def load_signal_from_npz(file_name):
    data = np.load(file_name)
    return data['signal'], data['sample_rate']
