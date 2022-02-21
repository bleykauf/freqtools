"""Submodule containing frequency-based models."""

from freqtools.freq_data import OscillatorNoise
import numpy as np
import matplotlib.pyplot as plt


class FreqModel:
    """
    Base class for frequency based models, i.e. values (y axis) as a function of
    frequency (x axis). Its functionality is purposfully kept simple and its main
    purpose is to implement basic behaviour.

    Parameters
    ----------
    *args :
        Placeholder, not used. The respective subclasses have to implement behaviour of
        positional
        arguments
    **kwargs :
        All keyworded arguments are added as attribues.
    """

    def __init__(self, *args, **kwargs):
        del args
        for key, value in kwargs.items():
            setattr(self, key, value)

    def values(self, freqs):
        raise NotImplementedError("Subclasses have to implement this method.")

    def plot(self, freqs, ax=None, xscale="log", yscale="log", ylabel=""):
        """
        Plot the model.

        Parameters
        ----------
        ax : Axis (optional)
            If axis is provided, they will be used for the plot. if not provided, a new
            plot will automatically be created.
        xscale : {"log" or "linear"}
            Scaling of the x axis.
        yscale : {"log" or "linear"}
            Scaling for the y axis.
        ylabel : str
            Label for the y axis.

        Returns
        -------
        fig, ax : Figure, Axis
            The Figure and Axis handles of the plot that was used.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        ax.plot(freqs, self.values(freqs), label=self.label)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Frequency / Hz")
        plt.grid(True, which="both", ls="-")
        return fig, ax


class OscillatorNoiseModel(FreqModel):
    """
    A base class holding  models of spectral densities of oscillator noise, i.e.
    frequency or phase noise. Its main purpose is to make it easy to convert between
    ASD(f), PSD(f) and L(f) in terms of both frequency and phase noise. The data is
    provided in one of these representations and makes all other representations
    available.

    Parameters
    ----------
    *args :
        Placeholder, not used. The respective subclasses have to implement behaviour of
        positional arguments
    n_sided : 1 (optional)
        placeholder, for now only one-sided distributions are supported.
    label : str
        Optional label used for plotting.
    **kwargs :
        All keyworded arguments are added as attribues.

    Attributes
    ----------
    n_sided
    label : str
        Optional label used for plotting
    representation
    unit
    ylabel
    """

    def __init__(self, n_sided=1, label="", representation=None, **kwargs):

        _allowed_representations = [
            "asd_freq",
            "asd_phase",
            "psd_freq",
            "psd_phase",
            "script_L",
        ]

        super().__init__(
            label=label,
            n_sided=n_sided,
            _allowed_representations=list(_allowed_representations),
            representation=representation,
            **kwargs
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

    @property
    def ylabel(self):
        """y axis label used for plotting; doesn't contain the unit."""  # noqa: D403
        return self._ylabel_dict[self.representation].format(self.n_sided)

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
        """Currently only one-sided distribtuions are supported."""
        return self._n_sided

    @n_sided.setter
    def n_sided(self, new_n):
        # FIXME: support for two-sided distributions.
        assert new_n == 1, "Only 1-sided distributions are supported as of yet."
        self._n_sided = new_n

    def values(self, freqs):
        """
        Array containing the values of the spectral density model. Maps to one
        representation, depending on `representation` attribute.
        """
        method = getattr(self, self.representation)
        return method(freqs)

    def asd_freq(self, freqs):
        """
        Amplitude spectral density of the frequency noise.

        Parameters
        ----------
        freqs : list_like
            Frequencies where the model is evaluated

        Returns
        -------
        1darray
        """
        return np.array(freqs) * self.asd_phase(freqs)

    def asd_phase(self, freqs):
        """
        Amplitude spectral density of the phase noise.

        Parameters
        ----------
        freqs : list_like
            Frequencies where the model is evaluated

        Returns
        -------
        1darray
        """
        return np.sqrt(self.psd_phase(freqs))

    def psd_freq(self, freqs):
        """
        Power spectral density of the frequency noise.

        Parameters
        ----------
        freqs : list_like
            Frequencies where the model is evaluated

        Returns
        -------
        1darray
        """
        return self.asd_freq(freqs) ** 2

    def psd_phase(self, freqs):
        """
        Power spectral density of the phase noise.

        Parameters
        ----------
        freqs : list_like
            Frequencies where the model is evaluated

        Returns
        -------
        1darray
        """
        # psd_phase can either be derived from psd_freq or script_L
        try:
            # convert to linear scale, factor 1/10 in exponent because dBc are used
            psd_phase = 10 ** (self.script_L(freqs) / 10)
            if self.n_sided == 1:
                # one-sided distributions have a factor 2, see Table A1 in [1]
                psd_phase *= 2
        except AttributeError:
            psd_phase = self.psd_freq(freqs) / np.array(freqs) ** 2
        return psd_phase

    def script_L(self, freqs):
        """
        The phase noise L(f) (pronounced "script ell of f").

        Parameters
        ----------
        freqs : list_like
            Frequencies where the model is evaluated

        Returns
        -------
        1darray
        """
        # see Table A.1 in [1] for the conversion from S_phi(f) and L(f)
        L = self.psd_phase(freqs)
        if self.n_sided == 1:
            L /= 2
        L = 10 * np.log10(L)  # convert to dBc/Hz
        return L

    def plot(self, freqs, ax=None, xscale="log", yscale="log", ylabel=""):
        """
        Plot the spectral density model.

        Parameters
        ----------
        freqs : list_like
            Frequencies where the model is evaluated.
        ax : matplotlib.axes.Axes (optional)
            The axes to plot on. If not given, a new figure is created.
        xscale : str {"log", "linear"} (optional)
            The scale of the x-axis.
        yscale : str {"log", "linear"} (optional)
            The scale of the y-axis.
        ylabel : str (optional)
            The label of the y-axis.
        """

        if not ylabel:
            # automatically create ylabel
            ylabel = self.ylabel + " / " + self.unit
        fig, ax = super().plot(
            freqs, ax=ax, xscale=xscale, yscale=yscale, ylabel=ylabel
        )

        if not self.representation == "script_L":
            ax.set_yscale("log")

        return fig, ax

    def to_oscillator_noise(self, freqs):
        """
        Convert the noise model to a `OscillatorNoise` object.

        Parameters
        ----------
        freqs : 1d-array
            The Fourier frequencies in Hz.

        Returns
        -------
        oscillator_noise : OscillatorNoise
            The model represented as an `OscillatorNoise` object.
        """
        oscillator_noise = OscillatorNoise(
            freqs,
            self.values(freqs),
            representation=self.representation,
            n_sided=self.n_sided,
            divide_by=1,
        )
        return oscillator_noise


class PowerLawNoise(OscillatorNoiseModel):
    r"""
    Power law phase and frequency noise models [1] for common noise types:

    .. math:: S_\phi = b_{i} \cdot f^{i}

    or

    .. math:: S_\phi = d_{i} \cdot f^{i}


    Parameters
    ----------
    coeff : float or list of floats
        Coefficient b_i (for phase noise) or d_i (for frequency noise), cp. [1]. Has to
        b a list if `edge_freqs` is set.
    exponent : int or list of ints
        The coefficient of the power  law noise. The noise type depends on the `base`
        for a given exponent, cp. [1]. Has to be a list if `edge_freqs` is set.
    edge_freqs : list of floats (optional)
        Allows to construct composite models that have different noise types for
        different frequency ranges. In this case, `coeff` and `exponent` have to be
        lists of length `len(edge_freqs) + 1`. The edge frequencies are the frequencies
        where the noise type changes.

        Allowed coefficients for phase noise:
            - -4 : random walk frequency
            - -3 : flicker frequency
            - -2 : white frequency
            - -1 : flicker phase
            -  0 : white phase

        Allowed coefficients for frequency noise:
            - -2 : random walk frequency
            - -1 : flicker frequency
            -  0 : white frequency
            -  1 : flicker phase
            -  2 : white phase

    base : {'phase', 'freq'}:
        determines whether the exponent and coefficient is given in terms of phase or
        frequency.

    References
    ----------
    [1] Enrico Rubiola - Enrico's Chart of Phase Noise and Two-Sample Variances
        (http://rubiola.org/pdf-static/Enrico%27s-chart-EFTS.pdf)
    """

    def __init__(
        self,
        coeff=1,
        exponent=0,
        base="phase",
        representation="psd_phase",
        edge_freqs=None,
    ):

        assert base in ["phase", "freq"]
        if base == "freq":
            # express everything in terms of psd_phase
            if type(exponent) == list:
                exponent = np.array(exponent)
            exponent = exponent - 2
        _label_dict = {
            -4: "random walk frequency",
            -3: "flicker frequency",
            -2: "white frequency",
            -1: "flicker phase",
            0: "white phase",
        }
        try:
            label = _label_dict[exponent] + " noise"
        except (KeyError, TypeError):
            label = "noise model"
        super().__init__(
            coeff=coeff, exponent=exponent, label=label, representation=representation
        )
        if edge_freqs:
            self.edge_freqs = list(edge_freqs)
            self.edge_freqs.append(np.inf)

    def psd_phase(self, freqs):
        """
        Power spectral density of the phase noise.

        Parameters
        ----------
        freqs : list_like
            Frequencies where the model is evaluated.

        Returns
        -------
        1darray :
            The power spectral density of the phase noise.
        """
        # Implement PSD of phase, all other representations can be calculated by virtue
        # of subclassing OscillatorNoiseModel.

        # FIXME: Improve the cases
        if type(self.coeff) == list:
            previous_f_edge = 0
            freqs = np.array(freqs)
            values = []
            for f_edge, coeff, exp in zip(self.edge_freqs, self.coeff, self.exponent):
                idx = np.where(np.logical_and(freqs > previous_f_edge, freqs <= f_edge))
                new_vals = coeff * freqs[idx] ** exp
                values.append(new_vals)
                previous_f_edge = f_edge

            # flatten the list of lists
            values = [item for sublist in values for item in sublist]

            if len(values) < len(freqs):
                # add the last value
                values.append(coeff * freqs[-1] ** exp)
            values = np.array(values)
        else:
            values = self.coeff * freqs**self.exponent
        return values


class JohnsonNoise(OscillatorNoiseModel):
    """
    Johnson Noise model.

    Parameters
    ----------
    signal_power : float
        Carrier signal power in dBm / Hz
    temperature : float (default 300.)
        Temperature in kelvin

    Attributes
    ----------
    signal_power : float
    temperature : float

    References
    ----------
    [1] Wikipedia: Johnson–Nyquist noise
        (https://en.wikipedia.org/wiki/Johnson%E2%80%93Nyquist_noise)
    """

    def __init__(
        self,
        signal_power,
        temperature=300.0,
        label="Johnson Noise",
        representation=None,
    ):
        super().__init__(temperature=temperature, label=label, n_sided=1)
        self.signal_power = signal_power

    def script_L(self, freqs):
        """
        Calculate the script_L representation of the Johnson noise.

        Parameters
        ----------
        freqs : list_like
            Frequencies where the model is evaluated.

        Returns
        -------
        1darray :
            The script_L representation of the Johnson noise.
        """

        # Implement L(f), all other representations can be calculated by virtue of
        # subclassing OscillatorNoiseModel.
        kb = 1.380649e-23  # Boltzmann constant in J/K
        freqs = np.ones(len(freqs))
        # 1e-3 because normalized to mW, normalized to signal power, length of freqds
        noise = (
            10 * np.log10(4 * kb * self.temperature / 1e-3) * freqs - self.signal_power
        )
        # subtract 3 dB since above quantity is defined as one-sided according to [1]
        noise -= 3
        return noise


class PhotonShotNoise(OscillatorNoiseModel):
    """
    Shot noise of an optical beatnote

    Parameters
    ----------
    signal_power : float
        Signal power in dBm / Hz
    radiant_sensitivity : float (default 0.3)
        Radiant sensitivity of the photodiode in A/W. Default taken for Hamamatsu G4176.
    optical_power : float (default 1e-3)
        optical power in W
    resisitivity : float (default 50)
        resistivity in Ohm.
    """

    def __init__(
        self,
        signal_power,
        optical_power=1e-3,
        radiant_sensitivity=0.3,
        representation=None,
        resistivity=50,
        label="Photon shot noise",
    ):

        super().__init__(
            radiant_sensitivity=radiant_sensitivity,
            resistivity=resistivity,
            label=label,
            optical_power=optical_power,
            n_sided=1,
        )
        self.signal_power = signal_power

    def script_L(self, freqs):
        """
        Calculate the script_L representation of the Johnson noise.

        Parameters
        ----------
        freqs : list_like
            Frequencies where the model is evaluated.

        Returns
        -------
        1darray :
            The script_L representation of the photon shot noise.
        """

        e = 1.6e-19  # electron charge in C
        freqs = np.ones(len(freqs))
        noise = (
            10
            * np.log10(
                2
                * e
                * self.radiant_sensitivity
                * self.optical_power
                * self.resistivity
                / 1e-3
            )
            * freqs
            - self.signal_power
        )
        # FIXME: Assume above expression is a one-sided distribution, but didn't check.
        noise -= 3
        return noise


class NoiseFloor(OscillatorNoiseModel):
    """
    Used for converting a spectrum analyzer measurement to oscilaltor noise model of the
     noise floor by dividing the detection noise by the carrier signal ampliude.

    Parameters
    ----------
    signal_power : float
        Signal power in dBm / Hz
    noise_floor : float
        measured noise floor in dBm / Hz
    divide_by : int (optional)
        dividy-by factor if prescaler was used for the measurements

    Attributes
    ----------
    signal_power : float
        Signal power in dBm / Hz
    noise_floor : float
            measured noise floor in dBm / Hz
    divide_by : int
        dividy-by factor if prescaler was used for the measurements
    """

    def __init__(
        self,
        signal_power,
        noise_floor,
        representation=None,
        divide_by=1,
        label="Detection noise",
    ):
        super().__init__(label=label, divide_by=divide_by, n_sided=1)
        self.signal_power = signal_power
        self.noise_floor = noise_floor

    def script_L(self, freqs):
        """
        Calculate the script_L representation of the noise floor.

        Parameters
        ----------
        freqs : list_like
            Frequencies where the model is evaluated.

        Returns
        -------
        1darray :
            The script_L representation of the noise floor.
        """
        freqs = np.ones(len(freqs))
        noise = (
            freqs * self.noise_floor + 20 * np.log10(self.divide_by) - self.signal_power
        )
        noise -= 3  # is measured as one-sided distribution
        return noise


class BetaLine(OscillatorNoiseModel):
    """
    The beta separation line as a function of frequency. It is originally defined for
    the single-sided spectral density (in Hz²/Hz).

    References
    ----------
    [1] Di Domenico, G., Schilt, S., & Thomann, P. (2010). Simple approach to the
        relation between laser frequency noise and laser line shape.
        Applied Optics, 49(25), 4801.
        https://doi.org/10.1364/AO.49.004801
    """

    def __init__(self, representation="psd_freq", **kwargs):
        super().__init__(
            representation=representation, label=r"$\beta$ separation line", **kwargs
        )

    def psd_freq(self, freqs):
        """
        The values of the beta separation line in Hz²/Hz as a function of frequency

        Parameters
        ----------
        freqs : float or list_like
            Frequency in Hz

        Returns
        -------
        1d array :
            The values of the beta separation line.
        """
        return 8 * np.log(2) * np.array(freqs) / np.pi**2

    def intersection(self, density, which="first"):
        """
        Returns the freqeuncy where the PSD and the beta separation line intersect.

        Parameters
        ----------
        density : OscillatorNoise
            A OscillatorNoise object. Correct representation (PSD of frequency) will
            automatically be used.
        which : {'first', 'last'}
            if there are more intersections between beta separation line and PSD, this
            argument determines whether the lowest (first, default) or highest (last)
            intersection frequency should be returned.

        Returns
        -------
        float :
            the frequency where the two lines intersect in Hz
        """
        psd_vals = density.psd_freq
        beta_vals = self.values(density.freqs)
        # indices of the intersections, i.e. where the sign of the difference between
        # the PSD and the beta separation line switches.
        idx = np.argwhere(np.diff(np.sign(psd_vals - beta_vals))).flatten()
        first_or_last = {"first": 0, "last": -1}
        if idx.size == 0:  # array is empty
            return np.inf
        return density.freqs[idx][first_or_last[which]]

    def linewidth(self, density, f_min=1e3, which="first"):
        """
        The FWHM linewidth according to equation (10) in [1].

        Parameters
        ----------
        density : OscillatorNoise
            A PhaseFreqNoise object. Correct scaling and base (PSD of frequency) will
            automatically be used.
        f_min : float
            minimum values of the frequency that should be considered in Hz. The
            default value for f_min (1e-3) corresponds to 1 ms.
        which : {'first', 'last'}
            if there are more intersections between beta separation line and PSD, this
            argument determines whether the lowest (first, default) or highest (last)
            intersection frequency should be returned.
        """
        f_max = self.intersection(density, which=which)
        idx = np.where(np.logical_and(density.freqs <= f_max, density.freqs >= f_min))
        freqs = density.freqs[idx]
        psd_vals_over_line = density.values[idx]
        # equation (10) in [1]
        area = np.trapz(psd_vals_over_line, x=freqs)
        fwhm = np.sqrt(8 * np.log(2) * area)  # equation (9) in [1]
        return fwhm


class AtomShotNoise(FreqModel):
    """
    Atomic shot noise of an atom interferometer gravimeter.

    Parameters
    ----------
    n_atoms : float
        Number of atoms.
    contrast : float
        Peak-to-peak contrast of the fringe.
    T : float
        Interferometer time in seconds.
    keff : float
        Effective wavevector of the atom interferometer in 1/m.
    """

    def __init__(self, n_atoms, contrast, T, keff, **kwargs):
        super().__init__(n_atoms=n_atoms, contrast=contrast, T=-T, keff=keff, **kwargs)

    def values(self, freqs):
        """Shot noise limit in m/s²."""
        sigma_p = 1 / np.sqrt(self.n_atoms)  # atomic shot noise
        sigma_g = 2 * sigma_p / (self.contrast * self.keff * self.T**2)  # in m/s**2
        return sigma_g
