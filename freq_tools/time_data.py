"""Submodule containing classes for time-based data."""

from scipy.signal import welch
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

    def to_oscillator_noise(self):
        """
        Caluclates the two-sided amplitude spectral density (ASD) in Hz/sqrt(Hz).

        Returns
        -------
        noise : OscillatorNoise
            Creates a SpectralDenisty object and initializes it with the calculated ASD
            using the Welch method
        """
        f, Pxx = welch(
            self.freqs,
            self.sample_rate,
            ("kaiser", 100),
            return_onesided=True,
            nperseg=1024,
            scaling="density",
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
