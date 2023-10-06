import numpy as np
import matplotlib.pyplot as plt
import scipy


class MusicDoaEstimator:
    def __init__(self, N, M, p, fc, mic_spacing, speed_of_sound, seed=0):
        self.N = N
        self.M = M
        self.p = p
        self.fc = fc
        self.speed_of_sound = speed_of_sound
        self.mic_spacing = mic_spacing
        self.seed = seed
        np.random.seed(self.seed)

    def generate_signal(self):
        s = np.zeros((self.N, self.p), dtype=complex)  # initialize as complex
        fs = 16000  # sampling frequency

        for t in range(self.p):
            t_val = t / fs  # calculate time value
            amp = np.random.multivariate_normal(mean=np.zeros(self.N), cov=np.diag(np.ones(self.N)))  # generate random amplitude
            s[:, t] = np.exp(1j * 2 * np.pi * self.fc * t_val) * amp

        return s

    def compute_steering_vector(self, theta):
        steering_vector = np.exp(-1j * 2 * np.pi * self.fc * self.mic_spacing * (np.cos(theta) / self.speed_of_sound) * np.arange(self.M))
        return steering_vector.reshape((self.M, 1))

    def compute_music_spectrum(self, theta, noise_eigvecs):
        steering_vector = self.compute_steering_vector(theta)
        vector_norm_squared = np.abs(steering_vector.conj().T @ steering_vector)[0, 0]
        noise_projection = steering_vector.conj().T @ noise_eigvecs @ noise_eigvecs.conj().T @ steering_vector
        return vector_norm_squared / noise_projection[0, 0]

    def estimate_doa(self, X, show_plots=True):
        Rxx = X @ X.conj().T / self.p

        eigvals, eigvecs = np.linalg.eig(Rxx)  # eigenvalues and eigenvectors
        eigvals = eigvals.real
        eignorms = np.abs(eigvals)

        idx = eignorms.argsort()[::-1]  # sort in descending order of eigenvalues
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        noise_eigvecs = eigvecs[:, self.N:]

        theta_vals = np.arange(0, 181, 1)
        music_spectrum_vals = np.array([self.compute_music_spectrum(val * np.pi / 180.0, noise_eigvecs) for val in theta_vals]).real

        # parameters for peak finding
        height = np.max(music_spectrum_vals) * 0.1  # peaks are at least 10% of max
        distance = 10  # peaks are at least 10 degrees apart

        peak_indices, _ = scipy.signal.find_peaks(music_spectrum_vals, height=height, distance=distance)

        if show_plots:
            self.show_eigenvalues(eigvals)
            self.show_music_spectrum(theta_vals, music_spectrum_vals)

        return theta_vals[peak_indices]

    def show_eigenvalues(self, eigvals):
        fig, ax = plt.subplots(figsize=(18, 6))
        ax.scatter(np.arange(self.N), eigvals[:self.N], label="N EigVals from Source")
        ax.scatter(np.arange(self.N, self.M), eigvals[self.N:], label="M-N EigVals from Noise")
        plt.legend()
        plt.show()

    def show_music_spectrum(self, theta_vals, music_spectrum_vals):
        fig, ax = plt.subplots(figsize=(18, 6))
        plt.plot(np.abs(theta_vals), music_spectrum_vals)
        plt.xticks(np.arange(0, 181, 10))
        plt.grid()
        plt.show()


def main():
    estimator = MusicDoaEstimator(N=3, M=10, p=100, fc=10000, mic_spacing=0.01, speed_of_sound=343)

    s = estimator.generate_signal()

    # DOAs (Direction of Arrivals) in radians
    doas = np.array([50, 85, 145]) * np.pi / 180

    # Steering matrix
    A = np.zeros((estimator.M, estimator.N), dtype=complex)
    for i in range(estimator.N):
        A[:, i] = estimator.compute_steering_vector(doas[i])[:, 0]

    n = np.random.multivariate_normal(mean=np.zeros(estimator.M), cov=np.diag(np.ones(estimator.M)), size=estimator.p).T
    X = A @ s + n

    estimated_doas = estimator.estimate_doa(X)
    print(estimated_doas)


if __name__ == "__main__":
    main()
