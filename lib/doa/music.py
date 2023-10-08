# Author: Eric Bezzam
# Date: July 15, 2016

import numpy as np

from .doa import DOA


class MUSIC(DOA):
    """
    Class to apply MUltiple SIgnal Classication (MUSIC) direction-of-arrival
    (DoA) for a particular microphone array.

    .. note:: Run locate_source() to apply the MUSIC algorithm.

    Parameters
    ----------
    L: numpy array
        Microphone array positions. Each column should correspond to the
        cartesian coordinates of a single microphone.
    fs: float
        Sampling frequency.
    nfft: int
        FFT length.
    c: float
        Speed of sound. Default: 343 m/s
    num_src: int
        Number of sources to detect. Default: 1
    mode: str
        'far' or 'near' for far-field or near-field detection
        respectively. Default: 'far'
    r: numpy array
        Candidate distances from the origin. Default: np.ones(1)
    azimuth: numpy array
        Candidate azimuth angles (in radians) with respect to x-axis.
        Default: np.linspace(-180.,180.,30)*np.pi/180
    colatitude: numpy array
        Candidate elevation angles (in radians) with respect to z-axis.
        Default is x-y plane search: np.pi/2*np.ones(1)
    frequency_normalization: bool
        If True, the MUSIC pseudo-spectra are normalized before averaging across the frequency axis, default:False
    """

    def __init__(
        self,
        L,
        fs,
        nfft,
        c=343.0,
        num_src=1,
        mode="far",
        r=None,
        azimuth=None,
        colatitude=None,
        frequency_normalization=False,
        signal_noise_thresh=None,
        **kwargs
    ):

        DOA.__init__(
            self,
            L=L,
            fs=fs,
            nfft=nfft,
            c=c,
            num_src=num_src,
            mode=mode,
            r=r,
            azimuth=azimuth,
            colatitude=colatitude,
            **kwargs
        )

        self.Pssl = None
        self.frequency_normalization = frequency_normalization
        self.signal_noise_thresh = signal_noise_thresh

    def _process(self, X, display, auto_identify):
        """
        Perform MUSIC for given frame in order to estimate steered response
        spectrum.
        """
        # compute steered response
        self.Pssl = np.zeros((self.num_freq, self.grid.n_points))
        C_hat = self._compute_correlation_matricesvec(X)
        # subspace decomposition
        eigvecs_s = self._extract_signal_subspace(C_hat[None, ...],
                                                  display=display,
                                                  auto_identify=auto_identify)
        # compute spatial spectrum
        identity = np.zeros((self.num_freq, self.M, self.M))
        identity[:, list(np.arange(self.M)), list(np.arange(self.M))] = 1
        cross = identity - np.matmul(eigvecs_s, np.transpose(np.conjugate(eigvecs_s), (0, 1, 3, 2)))
        self.Pssl = self._compute_spatial_spectrumvec(cross)
        if self.frequency_normalization:
            self._apply_frequency_normalization()
        self.grid.set_values(np.squeeze(np.sum(self.Pssl, axis=1) / self.num_freq))

    def _apply_frequency_normalization(self):
        """
        Normalize the MUSIC pseudo-spectrum per frequency bin
        """
        self.Pssl = self.Pssl / np.max(self.Pssl, axis=0, keepdims=True)

    def plot_individual_spectrum(self):
        """
        Plot the steered response for each frequency.
        """

        # check if matplotlib imported
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            import warnings

            warnings.warn("Matplotlib is required for plotting")
            return

        # only for 2D
        if self.grid.dim == 3:
            pass
        else:
            import warnings

            warnings.warn("Only for 2D.")
            return

        # plot
        for k in range(self.num_freq):

            freq = float(self.freq_bins[k]) / self.nfft * self.fs
            azimuth = self.grid.azimuth * 180 / np.pi

            plt.plot(azimuth, self.Pssl[k, 0 : len(azimuth)])

            plt.ylabel("Magnitude")
            plt.xlabel("Azimuth [degrees]")
            plt.xlim(min(azimuth), max(azimuth))
            plt.title("Steering Response Spectrum - " + str(freq) + " Hz")
            plt.grid(True)

    def _compute_spatial_spectrumvec(self, cross):
        mod_vec = np.transpose(
            np.array(self.mode_vec[self.freq_bins, :, :]), axes=[2, 0, 1]
        )
        # timeframe, frequ, no idea
        denom = np.matmul(
            np.conjugate(mod_vec[..., None, :]), np.matmul(cross, mod_vec[..., None])
        )
        return 1.0 / abs(denom[..., 0, 0])

    def _compute_spatial_spectrum(self, cross, k):

        P = np.zeros(self.grid.n_points)

        for n in range(self.grid.n_points):
            Dc = np.array(self.mode_vec[k, :, n], ndmin=2).T
            Dc_H = np.conjugate(np.array(self.mode_vec[k, :, n], ndmin=2))
            denom = np.linalg.multi_dot([Dc_H, cross, Dc])
            P[n] = 1 / abs(denom)

        return P

    # non-vectorized version
    def _compute_correlation_matrices(self, X):
        C_hat = np.zeros([self.num_freq, self.M, self.M], dtype=complex)
        for i in range(self.num_freq):
            k = self.freq_bins[i]
            for s in range(self.num_snap):
                C_hat[i, :, :] = C_hat[i, :, :] + np.outer(
                    X[:, k, s], np.conjugate(X[:, k, s])
                )
        return C_hat / self.num_snap

    # vectorized version
    def _compute_correlation_matricesvec(self, X):
        # change X such that time frames, frequency microphones is the result
        X = np.transpose(X, axes=[2, 1, 0])
        # select frequency bins
        X = X[..., list(self.freq_bins), :]
        # Compute PSD and average over time frame
        C_hat = np.matmul(X[..., None], np.conjugate(X[..., None, :]))
        # Average over time-frames
        C_hat = np.mean(C_hat, axis=0)
        return C_hat

    # vectorized versino
    def _extract_signal_subspace(self, R, display, auto_identify):
        # Step 1: Eigenvalue decomposition
        # Eigenvalues and eigenvectors are returned in ascending order; no need to sort.
        eigvals, eigvecs = np.linalg.eigh(R)

        # Step 2: Display if flag is True
        if display is True:
            self._display_eigvals(eigvals)

        # Step 3: Auto-identify signal and noise if flag is True
        if auto_identify:
            self.num_src = self._auto_identify(eigvals)

        # Step 4: Extract signal subspace
        # eigvecs_n = eigvecs[..., :-self.num_src]
        eigvecs_s = eigvecs[..., -self.num_src:]
        # eigvals_n = eigvals[..., :-self.num_src]
        # eigvals_s = eigvals[..., -self.num_src:]

        return eigvecs_s


    def _display_eigvals(self, eigvals):
        import matplotlib.pyplot as plt

        # Visualize the order of eigenvalue magnitudes
        sorted_indices = np.argsort(eigvals[0])
        cmap = plt.get_cmap("viridis")
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        cax1 = ax1.matshow(sorted_indices, cmap=cmap, aspect="auto")
        fig1.colorbar(cax1, label="Rank of Eigenvalue (not normalized)")
        ax1.set_title("Eigenvalue Ranks for Each Row")
        ax1.set_xlabel("Eigenvalue Index")
        ax1.set_ylabel("Row Index")

        # Visualize the magnitude of eigenvalues
        fig2, axes2 = plt.subplots(1, 8, figsize=(15, 8), sharey=True)
        for i in range(8):
            axes2[i].plot(eigvals[..., i], label=f"Eigenvalue {i+1}", marker="o", linestyle="")
            axes2[i].set_title(f"Eigenvalue {i+1}")
            axes2[i].set_xlabel("Row Index")
            axes2[i].set_ylim([np.min(eigvals), np.max(eigvals)])

        axes2[0].set_ylabel("Eigenvalue Magnitude")
        plt.suptitle("Distribution of Eigenvalue Magnitudes Across Rows")

        plt.show()


    def _auto_identify(self, eigvals):
        """
        Automatically identify the number of sources based on the eigenvalues
        of the correlation matrix.
        """
        eigvals_max = np.max(eigvals[0], axis=0)
        # Compute the eigenvalue ratio between consecutive elements
        eigvals_ratio = eigvals_max[1:] / eigvals_max[:-1]
        # Find the index where the ratio exceeds the threshold or return the last index
        index = np.argmax(eigvals_ratio > self.signal_noise_thresh)
        num_sources = len(eigvals_ratio) - index if index else len(eigvals_ratio)
        return num_sources
