"""
Submodule containing frequency-based data.

References
----------
[1] Riehle, F. (2006). Frequency Standards: Basics and Applications.
    Weinheim: Wiley-VCH. https://doi.org/10.1109/2944.902135
[2] IEEE Standard Definitions of Physical Quantities for Fundamental Frequency and Time
    Metrology — Random Instabilities (Vol. 2008).
    https://doi.org/10.1109/IEEESTD.2009.6581834
[3] Walls, F. L. (2001). Phase noise issues in femtosecond lasers. Laser Frequency
    Stabilization,
    Standards, Measurement, and Applications, 4269, 170.
    https://doi.org/10.1117/12.424466
[4] C. Freier (2017). Atom Interferometry at Geodetic Observatories,
    PhD Thesis, http://dx.doi.org/10.18452/17795
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class FreqData:
    """
    Base class for frequency data, i.e. values (y axis) as a function of frequency
    (x axis). Its functionality is purposfully kept simple and its main purpose is to
    implement basic behaviour like magic functions, e.g. support for slicing.

    Parameters
    ----------
    freqs : list_like
        Fourier frequency in Hz
    values : list_like
        the values corresponding to freq
    **kwargs :
        All keyworded arguments will be added as attributes.

    Attribues
    ---------
    freqs : 1darray
        Fourier frequencies in Hz
    values :
        Values at the Fourier frequencies.
    interpolation_options : dict
        options passed to `scipy.interpolation.interp1d`
    """

    def __init__(self, freqs, values, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.freqs = np.array(freqs)
        self.values = np.array(values)

        self.interpolation_options = {
            "kind": "linear",
            "fill_value": 0.0,
            "bounds_error": False,
        }

        assert len(self.values) == len(self.freqs)

    def __getitem__(self, key):
        new_instance = copy.deepcopy(self)
        new_instance.freqs = new_instance.freqs[key]
        new_instance.values = new_instance.values[key]
        return new_instance

    def __len__(self):
        return len(self.freqs)

    def values_interp(self, freqs):
        """
        Interpolated values. Behaviour of the interpolation can be controlled with the
        `interpolation_options` attribute of FreqData.

        Parameters
        ----------
        freq : float or list_like
            Frequncy in Hz
        """
        func = interp1d(self.freqs, self.values, **self.interpolation_options)
        return func(freqs)

    def join(self, others):
        """
        Joins another instance of FreqData. Only one value per fourier frequency is
        used, with preference for the ones appearing in `self`, followed by the first
        items in `other`.

        Parameters
        ----------
        others : (list of) FreqData
        """
        if not isinstance(others, list):
            # if only one FreqData is to be joined wrap it in a list for the for loop to
            #  work
            others = [others]
        freqs = np.concatenate([self.freqs, *[other.freqs for other in others]])
        values = np.concatenate([self.values, *[other.values for other in others]])
        # if there are the same `freqs` multiple times, prefere the `values` in `self`
        # or the first that appear in `other`
        self.freqs, idx = np.unique(freqs, return_index=True)
        self.values = values[idx]

    def plot(self, ax=None, xscale="log", yscale="log", ylabel=""):
        """
        Plots the `values` as a function of `freqs`.

        Parameters
        ----------
        ax : Axis (optional)
            If axis is provided, they will be used for the plot. if not provided, a new
            plot will automatically be created.
        xscale : {'log', 'linear'}
            Scaling of the x axis
        yscale : {'log', 'linear'}
            Scaling for the y axis
        ylabel : str
            the ylabel, if not explicitly set, an automatic label is generated

        Returns
        -------
        fig, ax : Figure and Axis
            The Figure and Axis handles of the plot that was used.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        ax.plot(self.freqs, self.values, label=self.label)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Frequency / Hz")

        return fig, ax


class OscillatorNoise(FreqData):
    """
    A class holding spectral density data of oscillator noise, i.e. frequency or phase
    noise. Its main purpose is to make it easy to convert between ASD(f), PSD(f) and
    L(f) in terms of both frequency and phase noise. The data is provided in one of
    these representations and makes all other representations available.

    Parameters
    ----------
    freqs : list_like
        The Fourier frequencies in Hz.
    values : list_like
        The oscillator noise. Unit depends on the `representation`:
            - ASD of phase noise: rad/√Hz
            - PSD of phase noise rad²/Hz
            - ASD of frequency noise: Hz/√Hz
            - PSD of frequency noise: Hz²/Hz
            - L(f) ("script ell of f"): dBc/Hz

        Note that these units might be different, if the OscillatorNoise instance is
        generated by the `calc_rms_noise` function (it integrates over the frequency
        axis, dropping the Hz in the denominator) or by scaling with a transfer
        function. E.g. the MachZehnderTransferFunction, might convert rad² to m/s².
        If in doubt, check the `unit` attribute.

    representation : {'asd_phase', 'psd_phase', 'asd_freq', 'psd_freq', 'script_L'}
        The representation the `values` are provided in.
    n_sided : {1, 2}
        Determiens whether the `values` represent one- or two-sided distributions for
        PSD and ASD. Note that this does not affect `script_L` since it is precicely
        defined [2]. It will, however, affect the calculation of `script_L`. Defaults to
        one-sided.
    divide_by : int
        If a prescaler was used provide the divide-by factor. The repsective oscillator
        noise will be automatcially scalled accordingly to provide the noise of the
        original signal.
    label : str
        Optional label used for plotting.

    Attributes
    ----------
    freqs
    values
    representation
    n_sided
    label
    asd_freq
    asd_phase
    psd_phase
    psd_freq
    script_L
    unit
    """

    def __init__(
        self, freqs, values, representation=None, n_sided=1, divide_by=1, label=""
    ):

        _allowed_representations = [
            "asd_freq",
            "asd_phase",
            "psd_freq",
            "psd_phase",
            "script_L",
        ]

        super().__init__(
            freqs,
            values,
            label=label,
            _allowed_representations=list(_allowed_representations),
            representation=representation,
        )

        self._unit_dict = {
            "asd_freq": "Hz/$\\sqrt{\\mathrm{Hz}}$",
            "asd_phase": "$\\mathrm{rad}/\\sqrt{\\mathrm{Hz}}$",
            "psd_freq": "Hz${}^2$/Hz",
            "psd_phase": "rad${}^2$/Hz",
            "script_L": "dBc/Hz",
        }

        self._ylabel_dict = {
            "asd_freq": "{}-sided ASD",
            "asd_phase": "{}-sided ASD",
            "psd_freq": "{}-sided PSD",
            "psd_phase": "{}-sided PSD",
            "script_L": "L(f)",
        }

        # setting _n_sided directly since setting n_sided will fail at this stage
        assert n_sided in [1, 2], "Sidedness has to be either 1 or 2"
        self._n_sided = n_sided

        if (
            self.representation == "script_L"
        ):  # L(f) is represented on a logarithmic scale
            values = values + 20 * np.log10(divide_by)
        # all other representations are linear
        elif self.representation[0:3] == "asd":
            values = values * divide_by
        elif self.representation[0:3] == "psd":
            values = values * divide_by**2

        # only one representation of the spectral density is set, rest is calculated
        # when needed
        setattr(self, "_" + self.representation, values)

    @property
    def ylabel(self):
        """y axis label used for plotting; doesn't contain the unit."""  # noqa D403
        return self._ylabel_dict[self.representation]

    @property
    def unit(self):
        """String containing the unit of `values`"""
        return self._unit_dict[self.representation]

    @property
    def representation(self):
        """The representation of `values`."""
        return self._representation

    @representation.setter
    def representation(self, representation):
        assert (
            representation in self._allowed_representations
        ), "representation must be one of {}".format(self._allowed_representations)
        self._representation = representation

    @property
    def n_sided(self):
        """
        Either 1 or 2, depending on whether the spectral density is one or two-sided.
        """  # noqa: D200
        return self._n_sided

    @n_sided.setter
    def n_sided(self, new_n):
        assert new_n in [1, 2]
        # resetting the psd_phase here is an arbitrary choice. Could have changed any
        # other representation. All other values will be recalculated from it.
        if self._n_sided < new_n:  # sidedness switches from 1 to 2
            new_psd_phase = self.psd_phase / 2
        elif self._n_sided > new_n:  # sidedness switches from 2 to 1
            new_psd_phase = self.psd_phase * 2
        elif self.n_sided == new_n:
            new_psd_phase = self.psd_phase
        self._force_recalculation()  # delete all representations
        self._psd_phase = new_psd_phase  # set the rescaled psd, rest will be calculated
        self._n_sided = new_n

    @property
    def values(self):
        """
        Array containing the values of the spectral density. Maps to one representation,
        depending on `representation` attribute.
        """
        return getattr(self, self.representation)

    @values.setter
    def values(self, vals):
        self._force_recalculation()
        setattr(self, "_" + self.representation, vals)

    def _force_recalculation(self):
        # This function is called when `n_sided` is changed or the `values` are changed.
        # check if `_allowed_representations` is already set. This is not the case
        #  during class instanciation, then nothing has to be recalculated.
        if hasattr(self, "_allowed_representations"):
            # remove data from representations because they have to be recalculated
            for attr in self._allowed_representations:
                try:
                    delattr(self, attr)
                except AttributeError:
                    # some attributes might not have been set, then continue
                    pass

    @property
    def asd_freq(self):
        """Amplitude spectral density of the frequency noise."""
        if not hasattr(self, "_asd_freq"):
            assert (
                "asd_freq" in self._allowed_representations
            ), "conversion to asd_freq not allowed"
            self._asd_freq = self.freqs * self.asd_phase
        return self._asd_freq

    @property
    def asd_phase(self):
        """Amplitude spectral density of the phase noise."""
        if not hasattr(self, "_asd_phase"):
            assert (
                "asd_phase" in self._allowed_representations
            ), "conversion to asd_phase not allowed"
            self._asd_phase = np.sqrt(self.psd_phase)
        return self._asd_phase

    @property
    def psd_freq(self):
        """Power spectral density of the frequency noise."""
        if not hasattr(self, "_psd_freq"):
            assert (
                "psd_freq" in self._allowed_representations
            ), "conversion to psd_freq not allowed"
            self._psd_freq = self.asd_freq**2
        return self._psd_freq

    @property
    def psd_phase(self):
        """Power spectral density of the phase noise."""
        if not hasattr(self, "_psd_phase"):
            assert (
                "psd_phase" in self._allowed_representations
            ), "conversion to psd_phase not allowed"
            # psd_phase can either be derived from psd_freq or script_L
            try:
                # convert to linear scale, factor 1/10 in exponent because dBc are used
                self._psd_phase = 10 ** (self.script_L / 10)
                if self.n_sided == 1:
                    # one-sided distributions have a factor 2, see Table A1 in [2]
                    self._psd_phase *= 2
            except RecursionError:
                self._psd_phase = self.psd_freq / self.freqs**2
        return self._psd_phase

    @property
    def script_L(self):
        """The phase noise L(f) (pronounced "script ell of f")"""
        if not hasattr(self, "_script_L"):
            assert (
                "script_L" in self._allowed_representations
            ), "conversion to script_L not allowed"
            # see Table A.1 in [2] for the conversion from S_phi(f) and L(f)
            L = self.psd_phase
            if self.n_sided == 1:
                L /= 2
            L = 10 * np.log10(L)  # convert to dBc/Hz
            self._script_L = L
        return self._script_L

    def plot(self, ax=None, xscale="log", yscale="log", ylabel=""):
        """
        Plot the spectral density of the noise.

        Parameters
        ----------
        ax : matplotlib.axes.Axes (optional)
            Axis to plot on. If not given, a new figure is created.
        xscale : str {"log", "linear"} (optional)
            Scale of the x-axis.
        yscale : str {"log", "linear"} (optional)
            Scale of the y-axis.
        ylabel : str (optional)
            Label of the y-axis.
        """

        if not ylabel:
            # automatically create ylabel
            ylabel = self.ylabel.format(self.n_sided) + " / " + self.unit

        if self.representation == "script_L":
            yscale = "linear"
        fig, ax = super().plot(ax=ax, xscale=xscale, yscale=yscale, ylabel=ylabel)

        ax.grid(True, which="both", ls="-")
        return fig, ax


class SpectrumAnalyzerData(FreqData):
    """
    Spectral data from a spectrum analyzer, e.g. a beatnote.

    Parameters
    ----------
    freqs : list_like
        frequencies (x axis) of the signal alayzer in Hz
    values : list_like
        the level at `freqs` in dBm
    rbw : float
        resolution bandwidth in Hz
    divide_by : int (optional)
        If a prescaler was used provide the divide-by factor. The repsective oscillator
        noise will be automatcially scalled accordingly to provide the noise of the
        original signal.
    label : str
        Optional label used for plotting.

    Attributes
    ----------
    freqs : 1darray
        frequencies in Hz
    values : 1darray
        signal level in dBm
    rbw : 1darray
        resolution bandwidth in Hz, used for conversion to OscillatorNoise, cp. method
        `to_oscillator_noise`
    label : str
        optional label used for plotting
    """

    def __init__(self, freqs, values, rbw, divide_by=1, label=""):

        super().__init__(freqs, values, rbw=rbw, label=label)
        self.values += 20 * np.log10(divide_by)

    def to_oscillator_noise(self, sideband="right"):
        """
        Convert to spectrum to oscillator noise by determining the carrier signal
        amplitude and normalizing one of the sidebands to the carrier level, i.e. it is
        the "old" definition of L(f) [2].

        Parameters
        ----------
        sideband : {'left', 'right'}
            Determiens whether the left or right sideband of the spectrum are used for
            calculating the oscillator noise.

        Returns
        -------
        OscillatorNoise
        """
        # see Table A.1 in [2] for the conversion from S_phi(f) and L(f)
        peak_level = max(self.values)

        center_freq = self.freqs[self.values == peak_level]

        if sideband == "left":
            selected_freqs = self.freqs < center_freq
        elif sideband == "right":
            selected_freqs = self.freqs > center_freq

        freqs = self.freqs[selected_freqs] - center_freq
        level = self.values[selected_freqs]

        # convert from dBm to dBc / Hz, no factor 1/2 because we analyze the
        # single-sideband or two-sided spectral density
        values = (level - peak_level) - 10 * np.log10(self.rbw)

        # divide_by already processed in __init__
        spec_density = OscillatorNoise(
            freqs, values, representation="script_L", n_sided=1, label=self.label
        )
        return spec_density

    def plot(self, ax=None, xscale="log", yscale="linear", ylabel="level / dBm"):
        """
        Plot the spectrum analyzer data.

        Parameters
        ----------
        ax : matplotlib.axes.Axes (optional)
            Axis to plot on. If not given, a new figure is created.
        xscale : str {"log", "linear"} (optional)
            Scale of the x-axis.
        yscale : str {"log", "linear"} (optional)
            Scale of the y-axis.
        ylabel : str (optional)
            Label of the y-axis (default: "level / dBm")
        """
        return super().plot(ax=ax, xscale=xscale, yscale=yscale, ylabel=ylabel)


def calc_rms_noise(noise):
    """
    Integrate the phase noise from highest to lowest Fourier frequency.

    This is for example shown in Fig. 3.17 in [4].

    Returns
    -------
    rms_noise : OscillatorNoise
        Root-mean-square (RMS) noise. Units are changed (no Hz in denominator due to
            integration).
    """
    rms_noise = []
    for k in np.arange(len(noise.values)):
        rms_noise.append(np.trapz(noise.values[k:], x=noise.freqs[k:]))
    # Eq. 2.43 in [4] calculates the square of the RMS noise.
    rms_noise = np.sqrt(np.array(rms_noise))
    rms_noise = OscillatorNoise(
        noise.freqs,
        rms_noise,
        label=noise.label,
        n_sided=noise.n_sided,
        representation=noise.representation,
    )

    rms_noise._allowed_representations = list(noise._allowed_representations)
    rms_noise._unit_dict = dict(noise._unit_dict)
    rms_noise._ylabel_dict = dict(noise._ylabel_dict)
    # change the labels
    for key in rms_noise._allowed_representations:
        # remove /Hz from the unit, due to integration over frequncy axis
        rms_noise._unit_dict[key] = noise._unit_dict[key][:-3]
        # add 'integrated' to the label
        rms_noise._ylabel_dict[key] = "integrated " + noise._ylabel_dict[key]

    return rms_noise
