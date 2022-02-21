"""
Submodule containing transfer fuction models and classes for transfer function data.
"""

import copy

import matplotlib.pyplot as plt
import numpy as np

from .freq_data import OscillatorNoise


class TransferFunction:
    """
    Class for transfer functions measured e.g. by a network analyzer

    Parameters
    ----------
    freq : list_like
        Fourier frequencies of the transfer function in Hz
    magnitude, phase : list_like
        Magnitude (in dB or as a linear factor, depending on `is_in_dB`) and phase (in
        degree) of the transfer function.
    label : str (optional)
    is_in_dB : bool (default True)
        indicates if the magnitude is in dB. If False, magnitude is provided as a linear
        factor. In this case the magnitude is automatically converted to dB.
    norm_to_dc : bool (default False)
        if True, the magnitude will be normalized to the value of the lowest frequency.
        Might be useful if comparing setups with different gains.
    phase_unwrap : bool (default True)
        unwraps the phase and removes jumps that might occur due to the measurment
    shift_phase : float (default 0)
        this value is added to the phase

    Attributes
    ----------
    freq, magnitude, phase : 1darray
        as described above. Note that magnitude is always in dB
    label : str
     Optional label
    """

    def __init__(
        self,
        freqs,
        magnitude,
        phase,
        label="",
        is_in_dB=True,
        norm_to_dc=False,
        phase_unwrap=True,
        shift_phase=0,
    ):
        self.freqs = np.array(freqs)
        self.magnitude = np.array(magnitude)
        if not is_in_dB:
            # convert magnitude if not already provided in dB
            self.magnitude = 20 * np.log10(self.magnitude)
        if norm_to_dc:
            # normalize magnitude to the magnitude at the lowest frequency, e.g. for
            # comparing different setups
            self.magnitude = self.magnitude - self.magnitude[0]

        self.phase = np.array(phase) + shift_phase
        if phase_unwrap:
            # remove discontinuities
            self.phase = 360 * np.unwrap(2 * np.pi * self.phase / 360) / (2 * np.pi)
        self.label = label

    def bode_plot(self, ax1=None, ax2=None, xlim=(0, 2e8), ylim=(-370, 10)):
        """
        Shows a Bode plot.

        Parameters
        ----------
        ax1, ax2 : Figure, Axis
            figure and axis objects that are used for the plotting. If not provided, a
            new figure will be created. `ax1` is used for the magnitude, `ax2` for the
            phase.
        xlim : tuple (default (0, 2e8))
            the plot limits of the frequency axis
        ylim : tuple (default (-370, 10))
            The plot limits for the phase
        """

        if (ax1 is None) or (ax2 is None):
            fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        else:
            fig = ax1.figure

        ax1.plot(self.freqs, self.magnitude, label=self.label)
        ax2.plot(self.freqs, self.phase, label=self.label)

        ax2.set_xscale("log")
        ax2.set_xlabel("Frequency / Hz")
        ax1.set_ylabel("Magnitude / dB")
        ax2.set_ylabel("Phase / °")

        ax1.grid(True, which="major", axis="both")
        ax2.grid(True, which="major", axis="both")

        ax2.set_xlim([max(xlim[0], self.freqs[0]), xlim[1]])
        ax2.set_ylim(ylim)

        return fig, (ax1, ax2)


class TransferFunctionModel:
    """
    Base class for transfer function models. It defines some basic behaviour. The heart
    of the transfer function model is the `tf` method that has to be implemented by
    subclasses. Provides some basic methods like `magnitude` and `phase`.

    Parameters
    ----------
    *args :
        Placeholder, not used. The respective subclasses have to implement behaviour of
        positional arguments
    **kwargs :
        All keyworded arguments are added as attribues.
    """

    def __init__(self, *args, **kwargs):
        # set all keyworded arguments as attributes
        del args
        for key, value in kwargs.items():
            setattr(self, key, value)

    def tf(self, freqs):
        """
        The (complex) transfer function H(f). Has to be implemented by the subclasses.
        """
        raise NotImplementedError("Has to be implemented in subclass")

    def magnitude(self, freqs):
        """
        Magnitude of the (complex) transfer function.

        Parameters
        ----------
        freqs : list_like
            frequencies where the transfer function model is evaluated.

        Returns
        -------
        1darray
        """
        return np.abs(self.tf(freqs))

    def phase(self, freqs):
        """
        Phase of the (complex) transfer function.

        Parameters
        ----------
        freqs : list_like
            frequencies where the transfer function model is evaluated.

        Returns
        -------
        1darray
        """
        return np.angle(self.tf(freqs))

    def plot_magnitude(
        self, freqs, ax=None, xscale="log", yscale="log", ylabel="magnitude"
    ):
        """
        Plots the magnitude of the transfer function.

        Parameters
        ----------
        freqs : list_like
            frequencies where the transfer function model is evaluated.
        ax : Axis (optional)
            If axis is provided, they will be used for the plot. if not provided, a new
            plot will automatically be created.

        Returns
        -------
        fig, ax : Figure and Axis
            The Figure and Axis handles of the plot that was used.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        ax.plot(freqs, self.magnitude(freqs))
        ax.set_xscale = xscale
        ax.set_yscale = yscale
        ax.set_ylabel = ylabel
        ax.set_xlabel("Frequency / Hz")
        return fig, ax

    def plot_phase(
        self, freqs, ax=None, xscale="log", yscale="linear", ylabel="phase / °"
    ):
        """
        Plots the phase of the transfer function.

        Parameters
        ----------
        freqs : list_like
            frequencies where the transfer function model is evaluated.
        ax : Axis (optional)
            If axis is provided, they will be used for the plot. if not provided, a new
            plot will automatically be created.

        Returns
        -------
        fig, ax : Figure and Axis
            The Figure and Axis handles of the plot that was used.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        ax.plot(freqs, self.phase(freqs))
        ax.set_xscale = xscale
        ax.set_yscale = yscale
        ax.set_ylabel = ylabel
        ax.set_xlabel("Frequency / Hz")
        return fig, ax

    def bode_plot(self, freqs, ax1=None, ax2=None):
        """
        Shows a Bode plot.

        Parameters
        ----------
        freqs : list_like
            frequencies where the transfer function model is evaluated.
        ax1, ax2 :  Axis (optional)
            If axes are provided, they will be used for the plot. if not provided, a new
             plot will be created. `ax1` is used for the magnitude, `ax2` for the phase.

        Returns
        -------
        fig, (ax1, ax2) : Figure and Axis
            The Figure and Axis handles of the plot that was used.
        """
        if (ax1 is None) or (ax2 is None):
            _, (ax1, ax2) = plt.subplots(2, sharex=True)

        fig, ax1 = self.plot_magnitude(freqs, ax=ax1)
        fig, ax2 = self.plot_phase(freqs, ax=ax2)

        return fig, (ax1, ax2)


class MachZehnderTransferFunction(TransferFunctionModel):
    """
    Class for the transfer function G(2*pi*f) of a Mach Zehnder Atom interferometer [1]

    Parameters
    ----------
    T, tau : float
        The interferometer time and pulse duration in s

    Attributes
    ----------
    T : float
    tau : float
    Omega_r : float
        Rabi frequency in 1/s, calculated from `tau`

    References
    ----------
    [1] P. Cheinet et al. - Measurement of the sensitivity function in a time-domain
        atomic interferometer
    """

    def __init__(self, T=260e-3, tau=36e-6, convert_to_g=False, k_eff=1.61e7):
        super().__init__(T=T, tau=tau, convert_to_g=convert_to_g, k_eff=k_eff)
        self.Omega_r = np.pi / 2 / self.tau

    def tf(self, freqs):
        """
        The complex transfer function.

        Parameters
        ----------
        f : list_like
            Fourier frequencies in Hz for which the transfer function is calculated

        Returns
        -------
        tf : 1darray
            transfer function in rad/rad.
        """
        omega = 2 * np.pi * freqs
        H_ai = (
            (4j * omega * self.Omega_r)
            / (omega**2 - self.Omega_r**2)
            * np.sin(omega * (self.T + 2 * self.tau) / 2)
            * (
                np.cos(omega * (self.T + 2 * self.tau) / 2)
                + self.Omega_r / omega * np.sin(omega * self.T / 2)
            )
        )
        if self.convert_to_g:
            H_ai = H_ai / (self.k_eff * self.T**2)
        return H_ai

    def scale_noise(self, noise, drop_nan=False):
        """
        Scales phase noise with the squared magnitude of transfer function as in Eq. (7)
         of [1].

        Parameters
        ----------
        noise : PhaseNoise
        drop_nan : bool (optional)
            If True, NaN values are removed from the scaled noise (default False).

        Returns
        -------
        scaled_noise : PhaseNoise

        References
        ----------
        [1] P. Cheinet et al. - Measurement of the sensitivity function in a time-domain
            atomic interferometer
        """
        assert noise.n_sided == 1, "the 1-sided spectral density should be used"
        psd_phase_scaled = self.magnitude(noise.freqs) ** 2 * noise.psd_phase
        # copy because arrays are mutable
        freqs = copy.copy(noise.freqs)

        if drop_nan:
            nan_idx = np.isnan(psd_phase_scaled)
            freqs = freqs[~nan_idx]
            psd_phase_scaled = psd_phase_scaled[~nan_idx]
            print(f"Dropped {sum(nan_idx)} NaN values.")

        scaled_noise = OscillatorNoise(
            freqs,
            psd_phase_scaled,
            label=noise.label,
            representation="psd_phase",
            n_sided=1,
        )

        if self.convert_to_g:
            # disable conversions as they no longer make sense for AI noise
            scaled_noise._allowed_representations = ["psd_phase"]
            scaled_noise._unit_dict = {"psd_phase": "m/s²/Hz"}
            scaled_noise._ylabel_dict = {"psd_phase": "AI noise"}

        return scaled_noise

    def plot_magnitude(
        self,
        freqs,
        ax=None,
        xscale="log",
        yscale="log",
        ylabel=r"$|H(2\pi f)|$ / rad/rad",
        freq0=None,
        window=1,
    ):
        """
        Plots the magnitude of the transfer function with optional averaging above a
        threshold frequency

        Parameters
        ----------
        freqs : 1darray
            Fourier frequencies in Hz for which the transfer function should be plotted
        freq0 : float (optional)
            threshold frequency in Hz above which the transfer function is averaged
        window : int
            averaging window, i.e. number of frequencies used for the moving average
            above the threshold frequency.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        if not freq0:
            freq0 = max(freqs)
        # calculate for both regimes separate and stich together
        f1 = freqs[freqs <= freq0]
        tf1 = self.magnitude(f1)
        f2 = _running_mean(freqs[freqs > freq0], window)
        tf2 = _running_mean(self.magnitude(freqs[freqs > freq0]), window)
        freqs = np.concatenate([f1, f2])
        tf = np.concatenate([tf1, tf2])
        ax.loglog(freqs, tf)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Frequency / Hz")
        return fig, ax


def _running_mean(x, N):
    # stolen from https://stackoverflow.com/a/27681394/2750945
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)
