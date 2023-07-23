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
        self.separate_recordings = self.record_sources()
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
            source.signal[:] = signal
            self.room.simulate()
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
        ref = np.moveaxis(self.separate_recordings, 1, 2)
        X = pra.transform.stft.analysis(self.mics_signals.T, self.L, self.hop, win=self.win_a)
        t_begin = time.perf_counter()

        if self.bss_type == "auxiva":
            # Run AuxIVA
            Y = pra.bss.auxiva(X, n_iter=50, proj_back=True, callback=self.convergence_callback)
        elif self.bss_type == "ilrma":
            # Run ILRMA
            Y = pra.bss.ilrma(X, n_iter=50, n_components=2, proj_back=True, callback=self.convergence_callback)
        elif self.bss_type == "fastmnmf":
            # Run FastMNMF
            Y = pra.bss.fastmnmf(X, n_iter=50, n_components=2, n_src=2, callback=self.convergence_callback)
        elif self.bss_type == "fastmnmf2":
            # Run FastMNMF2
            Y = pra.bss.fastmnmf2(X, n_iter=50, n_components=2, n_src=2, callback=self.convergence_callback)
        elif self.bss_type == "sparseauxiva":
            # Estimate set of active frequency bins
            ratio = 0.35
            average = np.abs(np.mean(np.mean(X, axis=2), axis=0))
            k = np.int_(average.shape[0] * ratio)
            S = np.sort(np.argpartition(average, -k)[-k:])
            # Run SparseAuxIva
            Y = pra.bss.sparseauxiva(X, S, n_iter=50, proj_back=True, callback=self.convergence_callback)

        t_end = time.perf_counter()
        print(f"Time for BSS: {(t_end - t_begin):.2f} s")
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

    def plot_results(self, ref, y, perm):
        plt.figure()
        a = np.array(self.SDR)
        b = np.array(self.SIR)
        print(self.SDR)
        print(self.SIR)
        plt.plot(np.arange(a.shape[0]) * 10, a[:, 0], label="SDR Source 0", c="r", marker="*")
        plt.plot(np.arange(a.shape[0]) * 10, a[:, 1], label="SDR Source 1", c="r", marker="o")
        plt.plot(np.arange(b.shape[0]) * 10, b[:, 0], label="SIR Source 0", c="b", marker="*")
        plt.plot(np.arange(b.shape[0]) * 10, b[:, 1], label="SIR Source 1", c="b", marker="o")
        plt.legend()

        plt.xlabel('Iteration')  # x軸は繰り返しの回数を表します。
        plt.ylabel('dB')         # y軸はSDRとSIRの値を表します。dB (デシベル) で表示されます。

        plt.tight_layout(pad=0.5)
        plt.show()

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

    choices = ["ilrma", "auxiva", "sparseauxiva", "fastmnmf", "fastmnmf2"]
    ## START BSS
    bss_type = choices[4]
    # マイク数を音源より増やす場合、fastmnmf と fastmnmf2 以外はエラー

    # Create BSSHandler instance
    bss_handler = BSSHandler(wav_files, bss_type)
    # bss_handler.room.plot()
    # plt.show()
    # Perform BSS
    Y = bss_handler.perform_bss()
    # Evaluate BSS performance
    y, perm = bss_handler.evaluate_bss(Y)
    # Get reference signal
    ref = np.moveaxis(bss_handler.separate_recordings, 1, 2)
    # Plot results
    bss_handler.plot_results(ref, y, perm)
    # Save results
    bss_handler.save_results(y)


        # plt.figure()
        # plt.subplot(2, 2, 1)
        # plt.specgram(ref[0, :, 0], NFFT=1024, Fs=self.room.fs, cmap="inferno")
        # plt.title("Source 0 (clean)")

        # plt.subplot(2, 2, 2)
        # plt.specgram(ref[1, :, 0], NFFT=1024, Fs=self.room.fs, cmap="inferno")
        # plt.title("Source 1 (clean)")

        # plt.subplot(2, 2, 3)
        # plt.specgram(y[perm[0], :], NFFT=1024, Fs=self.room.fs, cmap="inferno")
        # plt.title("Source 0 (separated)")

        # plt.subplot(2, 2, 4)
        # plt.specgram(y[perm[1], :], NFFT=1024, Fs=self.room.fs, cmap="inferno")
        # plt.title("Source 1 (separated)")

        # plt.tight_layout(pad=0.5)
