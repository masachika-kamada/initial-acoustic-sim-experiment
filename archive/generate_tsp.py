import numpy as np
import soundfile as sf


def get_tsp(order=18):
    N = 2**order + 2**(order-1) # length of signal
    J = 2**order # effective length
    shift = int((N-J)/2)
    TSP = np.exp(-1j * (2*np.pi) * J * (np.arange(int(N/2)) / N)**2)
    TSP = TSP * np.exp(-1j * (2*np.pi) * (shift/N) * np.arange(int(N/2)))
    iTSP = 1 / TSP
    tsp = np.fft.irfft(TSP)
    itsp = np.fft.irfft(iTSP)
    return tsp / np.max(np.abs(tsp)), itsp / np.max(np.abs(itsp))


if __name__ == "__main__":
    order = 18
    tsp,itsp = get_tsp(order)
    sampwidth = 3
    fs = 48000

    tsp_file_path = 'tsp_custom_48k.wav'
    itsp_file_path = 'itsp_custom_48k.wav'
    sf.write(tsp_file_path, tsp, fs)
    sf.write(itsp_file_path, itsp, fs)
