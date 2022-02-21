"""Submodule containing classes for time-based data."""

from scipy.signal import welch, get_window
import numpy as np
import matplotlib.pyplot as plt
import allantools

from .freq_data import OscillatorNoise


class CounterData:
    """
    Counter data, i.e. a time series of frequency data.

    Parameters
    ----------
    freqs : list_like
        measured frequencies in Hz
    duration : float
        duration of counter measurement the measurement in s
    divide_by : int (optional, default 1)
        if a prescaler was used, CounterData will automatically scale the resulting
        spectral densities.

    Attributes
    ----------
    freqs : 1darray
        measured frequencies in Hz
    mean_frequency : float
        mean frequency of the measurement in Hz
    duration : float
        duration of the counter meausurement in s
    n_samples : int
        number of measurements
    sample_rate : float
        sampling rate in Hz
    divide_by : int
        If a prescaler was used, provide the divide-by factor. Used for calculation of
        oscillator noise, c.p. `to_oscillator_noise` method.
    """

    def __init__(self, freqs, duration, divide_by=1, **kwargs):
        del kwargs  # unused but helpfull when loading data from files
        self.divide_by = divide_by
        self.freqs = freqs
        self.mean_frequency = np.mean(self.freqs)
        self.duration = duration
        self.n_samples = len(self.freqs)
        self.sample_rate = int(self.n_samples / self.duration)

    def to_oscillator_noise(self, method="welch", window="hann", **kwargs):
        """
        Create a OscillatorNoise object using the Welch method.

        Parameters
        ----------
        method : {"welch", "lpsd"}, optional
            The method used for calculating the oscillator noise. Defaults to Welch
            method.
        window : str or tuple or array_like, optional
            Desired window to use. If `window` is a string or tuple, it is  passed to
            `scipy.signal.get_window` to generate the window values, which are DFT-even
            by default. See `scipy.signal.get_window` for a list of windows and required
            parameters. If `window` is array_like it will be used directly as the window
            and its length must be nperseg. Defaults  to a Hann window.
        **kwargs :
            Arguments will be passed to the function used for calculating the oscillator
            noise. Note that `scaling` and `return_onesided` are always set
            automatically for consistency.

        Returns
        -------
        OscillatorNoise
        """

        assert method in ["welch", "lpsd"]
        if method == "welch":
            f, Pxx = welch(
                self.freqs,
                self.sample_rate,
                window=window,
                return_onesided=True,
                scaling="density",
                **kwargs
            )
        elif method == "lpsd":
            f, Pxx = lpsd(
                self.freqs, self.sample_rate, window=window, scaling="density", **kwargs
            )
        return OscillatorNoise(
            f, Pxx, representation="psd_freq", n_sided=1, divide_by=self.divide_by
        )

    def adev(self, scaling=1):
        """
        Calculates the Allan deviation of the data.

        Parameters
        ----------
        scaling : float (optional)
            normalization factor, i.e. the oscillator frequency ν_0

        Returns
        -------
        taus, adev, adeverror : list
            The taus for which the Allan deviation has been calculated, the adev at
            these taus and their statistical error.
        """
        freqs = np.array(self.freqs) * scaling
        tau_max = np.log10(len(self.freqs))
        taus = np.logspace(0, tau_max) / self.sample_rate
        (taus, adev, adeverror, _) = allantools.adev(
            freqs, data_type="freq", rate=self.sample_rate, taus=taus
        )
        return taus, adev, adeverror

    def plot_time_record(self, ax=None):
        """
        Plots the time record of the data.

        Parameters
        ----------
        ax : Axis (optional)
            If axis is provided, they will be used for the plot. if not provided, a new
            plot will automatically be created.

        Returns
        -------
        fig, ax : Figure and Axis
            The Figure and Axis handles of the plot that was used.
        """
        t = np.linspace(0, self.duration, num=self.n_samples)
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        ax.plot(
            t,
            self.freqs,
            label="Mean frequency: ({:3f}+/-{:3f}) MHz".format(
                self.mean_frequency * 1e-6, np.std(self.freqs) * 1e-6
            ),
        )
        ax.set_xlabel("time t (s)")
        ax.set_ylabel("frequency deviation (Hz)")
        ax.legend()
        plt.grid(b="on", which="minor", axis="both")
        plt.box(on="on")
        return fig, ax

    def plot_adev(self, ax=None, **kwargs):
        """
        Plots the Allan deviation of the data.

        Parameters
        ----------
        ax : Axis (optional)
            If axis is provided, they will be used for the plot. if not provided, a new
            plot will automatically be created.
        **kwargs:
            keyworded arguments passed to `adev()`.

        Returns
        -------
        fig, ax : Figure and Axis
            The Figure and Axis handles of the plot that was used.
        """
        taus, adev, adeverror = self.adev(**kwargs)
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.errorbar(taus, adev, yerr=adeverror)
        ax.set_xlabel("Averaging time t (s)")
        ax.set_ylabel(r"Allan deviation $\sigma_y(t)$")
        plt.grid(b="on", which="minor", axis="both")
        plt.box(on="on")
        return fig, ax


def lpsd(
    x,
    fs=1.0,
    window="hann",
    fmin=None,
    fmax=None,
    Jdes=1000,
    Kdes=100,
    Kmin=1,
    xi=0.5,
    scaling="density",
):
    """
    Compute the LPSD power spectrum estimation with a logarithmic frequency axis.

    Parameters
    ----------
    x : array_like
        time series to be transformed. "We assume to have a long stream x(n),
        n=0, ..., N-1 of equally spaced input data sampled with frequency fs. Typical
        values for N range from 10^4 to >10^6" [1]

    fs : float
        Sampling frequency of the `x` time series. Defaults to 1.0.

    window : str
        Desired window to use. If `window` is a string or tuple, it is passed to
        `scipy.signal.get_window` to generate the window values, which are DFT-even by
        default. See `scipy.signal.get_window` for a list of windows and required
        parameters. Defaults to a Hann window. "Choose a window function w(j, l) to
        reduce spectral leakage within the estimate. ... The computations of the window
        function will be performed when the segment lengths L(j) have been determined."
        [1]

    fmin, fmax : float, optional
        Lowest and highest frequency to estimate. Defaults to `fs / len(x)` and the
        Nyquist frequency `fs / 2`, respectively. "... we propose not to use the first
        few frequency bins. The first frequency bin that yields unbiased spectral
        estimates depends on the window function used. The bin is given by the effective
        half-width of the window transfer function." [1].

    Jdes : int, optional
        Desired number of Fourier frequencies. Defaults to 1000. "A typical value for J
        is 1000" [1]

    Kdes : int, optional
        Desired number of averages. Defaults to 100.

    Kmin : int, optional
        Minimum number of averages. Defaults to 1.

    xi : float, optional
        Fractional overlap between segments (0 <= xi < 1). Defaults to 0.5. "The
         amount of overlap is a trade-off between computational effort and flatness of
        the data weighting." [1]. See Figures 5 and 6 [1].

    scaling : {'density', 'spectrum'}, optional
        Selects between computing the power spectral density ('density') where `Pxx` has
        units of V**2/Hz and computing the power spectrum ('spectrum') where `Pxx` has
        units of V**2, if `x` is measured in V and `fs` is measured in Hz. Defaults to
        'density'.

    Returns
    -------
    f : 1-d array
        Vector of frequencies corresponding to Pxx
    Pxx : 1d-array
        Vector of (uncalibrated) power spectrum estimates

    Notes
    -----
    The implementation follows references [1] and [2] quite closely; in particular, the
    variable names used in the program generally correspond to the variables in the
    paper; and the corresponding equation numbers are indicated in the comments.

    References
    ----------
      [1] Michael Tröbs and Gerhard Heinzel, "Improved spectrum estimation  from
      digitized time series on a logarithmic frequency axis" in Measurement, vol 39
      (2006), pp 120-129.
        * http://dx.doi.org/10.1016/j.measurement.2005.10.010
        * http://pubman.mpdl.mpg.de/pubman/item/escidoc:150688:1

      [2] Michael Tröbs and Gerhard Heinzel, Corrigendum to "Improved spectrum
      estimation from digitized time series on a logarithmic frequency axis."
    """

    # Based on https://github.com/rudolfbyker/lpsd
    # FIXME: Replace with an import, once a package is available.

    assert scaling in ["density", "spectrum"]

    N = len(x)  # Table 1
    jj = np.arange(Jdes, dtype=int)  # Table 1

    if not fmin:
        fmin = fs / N  # Lowest frequency possible
    if not fmax:
        fmax = fs / 2  # Nyquist rate

    g = np.log(fmax) - np.log(fmin)  # (12)
    f = fmin * np.exp(jj * g / (Jdes - 1))  # (13)
    rp = fmin * np.exp(jj * g / (Jdes - 1)) * (np.exp(g / (Jdes - 1)) - 1)  # (15)

    # r' now contains the 'desired resolutions' for each frequency bin, given the rule
    # that we want the resolution to be equal to the difference in frequency between
    # adjacent bins. Below we adjust this to account for the minimum and desired number
    # of averages.

    ravg = (fs / N) * (1 + (1 - xi) * (Kdes - 1))  # (16)
    rmin = (fs / N) * (1 + (1 - xi) * (Kmin - 1))  # (17)

    case1 = rp >= ravg  # (18)
    case2 = np.logical_and(rp < ravg, np.sqrt(ravg * rp) > rmin)  # (18)
    case3 = np.logical_not(np.logical_or(case1, case2))  # (18)

    rpp = np.zeros(Jdes)

    rpp[case1] = rp[case1]  # (18)
    rpp[case2] = np.sqrt(ravg * rp[case2])  # (18)
    rpp[case3] = rmin  # (18)

    # r'' contains adjusted frequency resolutions, accounting for the finite length of
    # the data, the constraint of the minimum number of averages, and the desired number
    # of averages.  We now round r'' to the nearest bin of the DFT to get our final
    # resolutions r.
    L = np.around(fs / rpp).astype(int)  # segment lengths (19)
    r = fs / L  # actual resolution (20)
    m = f / r  # Fourier Tranform bin number (7)

    # Allocate space for some results
    Pxx = np.empty(Jdes)
    S1 = np.empty(Jdes)
    S2 = np.empty(Jdes)

    # Loop over frequencies. For each frequency, we basically conduct Welch's method
    # with the fourier transform length chosen differently for each frequency.
    # TODO: Try to eliminate the for loop completely, since it is unpythonic and slow.
    # Maybe write doctests first...
    for jj in range(len(f)):

        # Calculate the number of segments
        D = int(np.around((1 - xi) * L[jj]))  # (2)
        K = int(np.floor((N - L[jj]) / D + 1))  # (3)

        # reshape the time series so each column is one segment  <-- FIXME: This is not
        # clear.
        a = np.arange(L[jj])
        b = D * np.arange(K)
        ii = a[:, np.newaxis] + b  # Selection matrix
        data = x[ii]  # x(l+kD(j)) in (5)

        # Remove the mean of each segment.
        data -= np.mean(data, axis=0)  # (4) & (5)

        # Compute the discrete Fourier transform
        w = get_window(window, L[jj])  # (5)
        sinusoid = np.exp(
            -2j * np.pi * np.arange(L[jj])[:, np.newaxis] * m[jj] / L[jj]
        )  # (6)
        data = data * (sinusoid * w[:, np.newaxis])  # (5,6)

        # Average the squared magnitudes
        Pxx[jj] = np.mean(np.abs(np.sum(data, axis=0)) ** 2)  # (8)

        # Calculate some properties of the window function which will be used during
        # calibration
        S1[jj] = np.sum(w)  # (23)
        S2[jj] = np.sum(w**2)  # (24)

        # Calibration of spectral estimates
    if scaling == "spectrum":
        C = 2.0 / (S1**2)  # (28)
        Pxx = Pxx * C
    elif scaling == "density":
        C = 2.0 / (fs * S2)  # (29)
        Pxx = Pxx * C

    return f, Pxx
