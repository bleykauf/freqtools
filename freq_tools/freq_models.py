"""
Submodule containing frequency-based models.
"""

import copy
import numpy as np
import matplotlib.pyplot as plt

class FreqModel():
    """
    Base class for frequency based models, i.e. values (y axis) as a function of frequency
    (x axis). Its functionality is purposfully kept simple and its main purpose is to implement 
    basic behaviour.

    Parameters
    ----------
    *args :
        Placeholder, not used. The respective subclasses have to implement behaviour of positional
        arguments
    **kwargs :
        All keyworded arguments are added as attribues.
    """

    def __init__(self, *args, **kwargs):
        del args
        for key, value in kwargs.items():
            setattr(self, key, value)

    def plot(self, freqs, ax=None , xscale='log', yscale='log', ylabel=''):
        """
        Parameters
        ----------
        ax : Axis (optional)
            If axis is provided, they will be used for the plot. if not provided, a new plot will
            automatically be created.
        xscale : {'log', 'linear'}
            Scaling of the x axis
        yscale : {'log', 'linear'}
            Scaling for the y axis
        ylabel : str
            the ylabel

        Returns
        -------
        fig, ax : Figure and Axis
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
        ax.set_xlabel('Frequency / Hz')
        plt.grid(True, which = 'both', ls = '-')
        return fig, ax


class OscillatorNoiseModel(FreqModel):
    """
    A base class holding  models of spectral densities of oscillator noise, i.e. frequency or phase 
    noise. Its main purpose is to make it easy to convert between ASD(f), PSD(f) and L(f) in terms 
    of both frequency and phase noise. The data is provided in one of these representations and 
    makes all other representations available.

    Parameters
    ----------
    *args :
        Placeholder, not used. The respective subclasses have to implement behaviour of positional
        arguments
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
    
    def __init__(self, n_sided=1, label='', representation=None, **kwargs):

        _allowed_representations = ['asd_freq', 'asd_phase', 'psd_freq', 'psd_phase', 'script_L']

        super().__init__(label=label, n_sided=n_sided,
                        _allowed_representations=list(_allowed_representations),
                        representation=representation, 
                        **kwargs)

        self._unit_dict = {'asd_freq'  : 'Hz/$\\sqrt{\\mathrm{Hz}}$',
                           'asd_phase' : '$\\mathrm{rad}/\\sqrt{\\mathrm{Hz}}$',
                           'psd_freq'  : 'Hz${}^2$/Hz',
                           'psd_phase' : 'rad${}^2$/Hz',
                           'script_L'  : 'dBc/Hz'}
        
        self._ylabel_dict = {'asd_freq' : '{}-sided ASD',
                             'asd_phase' : '{}-sided ASD',
                             'psd_freq'  : '{}-sided PSD',
                             'psd_phase' : '{}-sided PSD',
                             'script_L'  : 'L(f)'}
        
    @property
    def ylabel(self):
        """y axis label used for plotting; doesn't contain the unit."""
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
        assert representation in self._allowed_representations, \
            'representation must be one of {}'.format(self._allowed_representations)
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
        Array containing the values of the spectral density model. Maps to one representation, 
        depending on `representation` attribute.
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
        return self.asd_freq(freqs)**2 

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
            psd_phase = 10**(self.script_L(freqs) / 10)
            if self.n_sided == 1:
                # one-sided distributions have a factor 2, see Table A1 in [1]
                psd_phase *= 2 
        except AttributeError:
            psd_phase = self.psd_freq(freqs) / np.array(freqs)**2
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
            L /=  2
        L = 10 * np.log10(L) # convert to dBc/Hz
        return L

    def plot(self, freqs, ax=None, xscale='log', yscale='log', ylabel=''):

        if not ylabel:
            # automatically create ylabel
            ylabel = self.ylabel + ' / ' + self.unit
        fig, ax = super().plot(freqs, ax=ax, xscale=xscale, yscale=yscale, ylabel=ylabel)
        
        if not self.representation == 'script_L':
            ax.set_yscale('log')

        return fig, ax

class PowerLawNoise(OscillatorNoiseModel):
    r"""
    Power law phase and frequency noise models [1] for common noise types:
    
    .. math:: S_\phi = b_{i} \cdot f^{i}

    or

    .. math:: S_\phi = d_{i} \cdot f^{i}
    

    Parameters
    ----------
    coeff : float
        Coefficient b_i (for phase noise) or d_i (for frequency noise), cp. [1].
    exponent : int
        The coefficient of the power  law noise. The noise type depends on the `base` for a 
        given exponent, cp. [1]. 

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
        determines whether the exponent and coefficient is given in terms of 

    References
    ----------
    [1] Enrico Rubiola - Enrico's Chart of Phase Noise and Two-Sample Variances 
        (http://rubiola.org/pdf-static/Enrico%27s-chart-EFTS.pdf)
    """
    def __init__(self, coeff=1, exponent=0, base='phase', representation='psd_phase'):

        assert base in ['phase', 'freq']
        if base == 'freq':
            # express everything in terms of psd_phase
            exponent = exponent - 2 
        _label_dict = {-4 : 'random walk frequency',
                       -3 : 'flicker frequency',
                       -2 : 'white frequency',
                       -1 : 'flicker phase',
                        0 : 'white phase'}

        label = _label_dict[exponent]
        super().__init__(coeff=coeff, exponent=exponent, label=label, 
                         representation=representation)
    
    def psd_phase(self, freqs):
        # Implement PSD of phase, all other representations can be calculated by virtue of 
        # subclassing OscillatorNoiseModel.
        return self.coeff * freqs**self.exponent

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
    [1] Johnson–Nyquist noise (https://en.wikipedia.org/wiki/Johnson%E2%80%93Nyquist_noise)
    """
    def __init__(self, signal_power, temperature=300., label='Johnson Noise',
                 representation=None):
        super().__init__(temperature=temperature, label=label, n_sided=1)
        self.signal_power = signal_power
        
    def script_L(self, freqs):
        # Implement L(f), all other representations can be calculated by virtue of subclassing
        # OscillatorNoiseModel.
        kb=1.380649e-23 # Boltzmann constant in J/K
        freqs = np.ones(len(freqs))
        # 1e-3 because normalized to mW, normalized to signal power, length of freqds
        noise = 10 * np.log10(4 * kb * self.temperature / 1e-3) * freqs - self.signal_power
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
    def __init__(self, signal_power, optical_power=1e-3, radiant_sensitivity=0.3, 
                 representation=None, resistivity=50, label='Photon shot noise'):
    
        super().__init__(radiant_sensitivity=radiant_sensitivity, resistivity=resistivity,
                label=label, optical_power=optical_power, n_sided=1)
        self.signal_power = signal_power

    def script_L(self, freqs):
        e  = 1.6e-19 # electron charge in C
        freqs = np.ones(len(freqs))
        noise = 10 * np.log10(2 * e * self.radiant_sensitivity * self.optical_power * 
            self.resistivity / 1e-3) * freqs - self.signal_power
        # FIXME: Assume the above expression is a one-sided distribution, but didn't check
        noise -= 3
        return noise

class NoiseFloor(OscillatorNoiseModel):
    """
    Used for converting a spectrum analyzer measurement to oscilaltor noise model of the noise 
    floor by dividing the detection noise by the carrier signal ampliude.

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
    def __init__(self, signal_power, noise_floor,  representation=None, 
                 divide_by=1, label='Detection noise'):
        super().__init__(label=label, divide_by=divide_by, n_sided=1)
        self.signal_power = signal_power
        self.noise_floor = noise_floor

    def script_L(self, freqs):
        freqs = np.ones(len(freqs))
        noise = freqs * self.noise_floor + 20 * np.log10(self.divide_by) - self.signal_power
        noise -= 3 # is measured as one-sided distribution
        return noise 


class BetaLine(OscillatorNoiseModel):
    """
    The beta separation line as a function of frequency. It is originally defined for the single-
    sided spectral density (in Hz²/Hz).

    References
    ----------
    [1] Di Domenico, G., Schilt, S., & Thomann, P. (2010). Simple approach to the relation between 
        laser frequency noise and laser line shape. Applied Optics, 49(25), 4801.
        https://doi.org/10.1364/AO.49.004801
    """

    def __init__(self, representation='psd_freq', **kwargs):
        super().__init__(representation=representation, label=r'$\beta$ separation line', **kwargs)

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

    
    def intersection(self, density, search_range=(1e0, 1e8), **kwargs):
        """
        Returns the intersection between a PSD and the beta separation line

        Parameters
        ----------
        density : PhaseFreqNoise
            A PhaseFreqNoise object. Correct scaling and base (PSD of frequency) will 
            automatically be used.
        search_range : tupel (default (1e0, 1e8))
            intersection is searched within these frequency limits (in Hz)
        **kwargs:
            keyworded arguments that control the `interpolation_options` of the density, e.g. 
            'fill_value'.

        Returns
        -------
        float : 
            the frequency where the two lines intersect in Hz
        """
        psd = _get_psd(density, **kwargs)
        freqs = np.logspace(np.log10(search_range[0]), np.log10(search_range[1]), 1000)
        psd_vals = psd.values_interp(freqs)
        beta_vals = self.values(freqs)
        # indices of the intersections
        idx = np.argwhere(np.diff(np.sign(psd_vals - beta_vals))).flatten()
        return freqs[idx][0]
    

    def linewidth(self, noise, f_min=1e3, f_max=None, n=1000, **kwargs):
        """
        The FWHM linewidth according to equation (10) in [1].

        Parameters
        ----------
        density : PhaseFreqNoise
            A PhaseFreqNoise object. Correct scaling and base (PSD of frequency) will 
            automatically be used.
        f_min, fmax : float
            minimum and maximum values of the frequency that should be considered in Hz. The 
            default value for f_min (1e-3) corresponds to 1 ms. If no maximum frequency is given
            it is determined from the intersection with the beta separation line.
        n : int
            the number of points used for the integration
        **kwargs:
            keyworded arguments that control the `interpolation_options` of the density, e.g. 
            'fill_value'.
        """
        psd = _get_psd(noise, **kwargs)
        if not f_max:
            f_max = self.intersection(noise)
        freqs = np.logspace(np.log10(f_min), np.log10(f_max), n)
        beta_vals = self.psd_freq(freqs)
        psd_vals = psd.values_interp(freqs)
        # equation (10) in [1]
        area =  np.trapz(np.heaviside(psd_vals, beta_vals) * psd_vals, x=freqs)
        fwhm = np.sqrt(8 * np.log(2) * area) # equation (9) in [1]
        return fwhm


def _get_psd(noise, **kwargs):
    # make a copy of a OscillatorNoise object with correct base and scaling for usage for the
    # beta separation line.
    psd = copy.deepcopy(noise) # so it toesn't mess with the original object
    psd.representation = 'psd_freq'
    # additional user-set interpolation options, might overwrite fill_value
    for key, value in kwargs.items():
        psd.interpolation_options[key] = value
    return psd