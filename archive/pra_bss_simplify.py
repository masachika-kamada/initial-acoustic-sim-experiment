import time
import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
from mir_eval.separation import bss_eval_sources
from scipy.io import wavfile


class BSSHandler:
    def __init__(self, wav_files, bss_type, room_dim=[8, 9], source=np.array([1, 4.5]), locations=[[2.5, 3], [2.5, 6]], L=2048):
        self.wav_files = wav_files
        self.bss_type = bss_type
        self.room_dim = room_dim
        self.source = source
        self.locations = locations
        self.L = L
        self.hop = L // 2
        self.win_a = pra.hann(L)
        self.win_s = pra.transform.stft.compute_synthesis_window(self.win_a, self.hop)
        self.SDR = []
        self.SIR = []
        self.room = self.prepare_room()
        self.signals = self.get_signals()
        self.add_sources_to_room()
        self.room.add_microphone_array(pra.MicrophoneArray(np.array([[6.5, 4.47], [6.5, 4.49], [6.5, 4.51]]).T, fs=self.room.fs))
        self.room.compute_rir()
        # 音源別に録音
        self.separate_recordings = self.record_sources()
        # 全音源を混合
        self.mics_signals = np.sum(self.separate_recordings, axis=0)

    def prepare_room(self):
        return pra.ShoeBox(self.room_dim, fs=16000, max_order=15, absorption=0.35, sigma2_awgn=1e-8)

    def get_signals(self):
        return [
            np.concatenate([wavfile.read(f)[1].astype(np.float32) for f in source_files])
            for source_files in self.wav_files
        ]

    def add_sources_to_room(self):
        delays = [1.0, 0.0]
        for sig, d, loc in zip(self.signals, delays, self.locations):
            self.room.add_source(loc, signal=np.zeros_like(sig), delay=d)

    def record_sources(self):
        separate_recordings = []
        for source, signal in zip(self.room.sources, self.signals):
            # print(source.signal, source.signal.shape)
            source.signal[:] = signal
            # print(source.signal, source.signal.shape)
            self.room.simulate()
            # print(self.room.mic_array.signals, self.room.mic_array.signals.shape)
            separate_recordings.append(self.room.mic_array.signals)
            source.signal[:] = 0.0
        return np.array(separate_recordings)

    def convergence_callback(self, Y):
        ref = np.moveaxis(self.separate_recordings, 1, 2)
        y = pra.transform.stft.synthesis(Y, self.L, self.hop, win=self.win_s)
        y = y[self.L - self.hop :, :].T
        m = np.minimum(y.shape[1], ref.shape[1])
        sdr, sir, sar, perm = bss_eval_sources(ref[:, :m, 0], y[:, :m])
        self.SDR.append(sdr)
        self.SIR.append(sir)

    def perform_bss(self):
        X = pra.transform.stft.analysis(self.mics_signals.T, self.L, self.hop, win=self.win_a)
        if self.bss_type == "fastmnmf2":
            Y = pra.bss.fastmnmf2(X, n_iter=30, n_components=2, n_src=2, callback=self.convergence_callback)
        return Y

    def evaluate_bss(self, Y):
        y = pra.transform.stft.synthesis(Y, self.L, self.hop, win=self.win_s)
        y = y[self.L - self.hop :, :].T
        ref = np.moveaxis(self.separate_recordings, 1, 2)
        m = np.minimum(y.shape[1], ref.shape[1])
        sdr, sir, sar, perm = bss_eval_sources(ref[:, :m, 0], y[:, :m])
        print("SDR:", sdr)
        print("SIR:", sir)
        return y, perm

    def plot_results(self):
        print(self.SDR)
        print(self.SIR)

    def save_results(self, y):
        wavfile.write("bss_iva_mix.wav", self.room.fs, pra.normalize(self.mics_signals[0, :], bits=16).astype(np.int16))
        for i, sig in enumerate(y):
            wavfile.write(f"bss_iva_source{i+1}.wav", self.room.fs, pra.normalize(sig, bits=16).astype(np.int16))


if __name__ == "__main__":
    wav_files = [
        [
            "data/raw/sample/cmu_arctic_us_axb_a0004.wav",
            "data/raw/sample/cmu_arctic_us_axb_a0005.wav",
            "data/raw/sample/cmu_arctic_us_axb_a0006.wav",
        ],
        [
            "data/raw/sample/cmu_arctic_us_aew_a0001.wav",
            "data/raw/sample/cmu_arctic_us_aew_a0002.wav",
            "data/raw/sample/cmu_arctic_us_aew_a0003.wav",
        ],
    ]

    ## START BSS
    bss_type = "fastmnmf2"

    # Create BSSHandler instance
    bss_handler = BSSHandler(wav_files, bss_type)
    # Perform BSS
    Y = bss_handler.perform_bss()
    # Evaluate BSS performance
    y, perm = bss_handler.evaluate_bss(Y)
    # Plot results
    bss_handler.plot_results()
    # Save results
    bss_handler.save_results(y)
