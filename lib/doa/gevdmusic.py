import numpy as np
import scipy

from .music import *


class GevdMUSIC(MUSIC):
    """
    Class to apply the Generalized Eigenvalue Decomposition (GEVD) based MUSIC
    (GEVD-MUSIC) direction-of-arrival (DoA) for a particular microphone array,
    extending the capabilities of the original MUSIC algorithm.

    .. note:: Run locate_source() to apply the GEVD-MUSIC algorithm.

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
    source_noise_thresh: float
        Threshold for automatically identifying the number of sources. Default: 100
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
        source_noise_thresh=100,
        X_noise = None,
        output_dir=".",
        **kwargs
    ):

        MUSIC.__init__(
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
            frequency_normalization=frequency_normalization,
            source_noise_thresh=source_noise_thresh,
            output_dir=output_dir,
            **kwargs
        )

        self.X_noise = X_noise

    def _process(self, X, display, save, auto_identify, use_noise):
        # compute steered response
        self.spatial_spectrum = np.zeros((self.num_freq, self.grid.n_points))
        # Compute source and noise correlation matrices
        R = self._compute_correlation_matricesvec(X)
        K = self._compute_correlation_matricesvec(self.X_noise)
        # subspace decomposition
        eigvecs_s, eigvecs_n = self._extract_subspaces(R, K, display=display, save=save,
                                                       auto_identify=auto_identify)
        # compute spatial spectrum
        self.spatial_spectrum = self._compute_spatial_spectrum(eigvecs_s, eigvecs_n, use_noise)

        if self.frequency_normalization:
            self._apply_frequency_normalization()
        self.grid.set_values(np.squeeze(np.sum(self.spatial_spectrum, axis=1) / self.num_freq))

    def _extract_subspaces(self, R, K, display, save, auto_identify):
        # Initialize
        eigvals = np.empty(R.shape[:2], dtype=complex)
        eigvecs = np.empty(R.shape, dtype=complex)

        # Step 1: Eigenvalue decomposition
        for i in range(self.num_freq):
            eigvals[i], eigvecs[i] = scipy.linalg.eigh(R[i], K[i])
        # If eigenvalues are complex, take the real part
        if np.iscomplexobj(eigvals):
            eigvals = np.real(eigvals)

        # Step 2: Display if flag is True
        if display or save:
            self._plot_eigvals(eigvals, display, save)

        # Step 3: Auto-identify source and noise if flag is True
        if auto_identify:
            self.num_src = self._auto_identify(eigvals, save)

        # Step 4: Extract subspace
        eigvecs_n = eigvecs[..., :-self.num_src]
        eigvecs_s = eigvecs[..., -self.num_src:]

        return eigvecs_s, eigvecs_n
