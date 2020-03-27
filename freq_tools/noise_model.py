import numpy as np
import matplotlib.pyplot as plt
import copy

class BetaLine():
    """
    The beta separation line as a function of frequency. It is defined for the single-sided 
    spectral density (in Hz²/Hz).

    References
    ----------
    [1] Di Domenico, G., Schilt, S., & Thomann, P. (2010). Simple approach to the relation between 
        laser frequency noise and laser line shape. Applied Optics, 49(25), 4801.
        https://doi.org/10.1364/AO.49.004801
    """
    def values(self, freqs):
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

    def plot(self, freqs, fig=None, ax=None):
        """
        Plots the beta separation line.
        """
        if not fig:
            fig, ax = plt.subplots()
        ax.plot(freqs, self.values(freqs), label=r'$\beta$ separation line')
        ax.set_xscale('log')
        ax.set_xlabel('Frequency / Hz')
        plt.grid(True, which = 'both', ls = '-')
        return fig, ax
    
    def intersection(self, density, search_range=(1e0, 1e8), **kwargs):
        """
        Returns the intersection between a PSD and the beta separation line

        Parameters
        ----------
        density : SpectralDensity
            A SpectralDensity object. Correct scaling and base (PSD of frequency) will 
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
        psd = self._get_psd(density, **kwargs)
        freqs = np.logspace(np.log10(search_range[0]), np.log10(search_range[1]), 1000)
        psd_vals = psd.values_interp(freqs)
        beta_vals = self.values(freqs)
        # indices of the intersections
        idx = np.argwhere(np.diff(np.sign(psd_vals - beta_vals))).flatten()
        return freqs[idx][0]
    

    def linewidth(self, density, f_min=1e3, f_max=None, n=1000, **kwargs):
        """
        The FWHM linewidth according to equation (10) in [1].

        Parameters
        ----------
        density : SpectralDensity
            A SpectralDensity object. Correct scaling and base (PSD of frequency) will 
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
        psd = self._get_psd(density, **kwargs)
        if not f_max:
            f_max = self.intersection(density)
        freqs = np.logspace(np.log10(f_min), np.log10(f_max), n)
        beta_vals = self.values(freqs)
        psd_vals = psd.values_interp(freqs)
        # equation (10) in [1]
        area =  np.trapz(np.heaviside(psd_vals, beta_vals) * psd_vals, x=freqs)
        fwhm = np.sqrt(8 * np.log(2) * area) # equation (9) in [1]
        return fwhm

    def _get_psd(self, density, **kwargs):
        # make a copy of a SpectralDensity object with correct base and scaling for usage for the
        # beta separation line.
        psd = copy.deepcopy(density) # so it toesn't mess with the original object
        psd.base = 'freq'
        psd.scaling = 'psd'
        psd.interpolation_options['fill_value'] = (psd.values[0], psd.values[-1])
        # additional user-set interpolation options, might overwrite fill_value
        for key, value in kwargs.items():
            psd.interpolation_options[key] = value
        return psd

class PhaseNoiseModel():
    """
    Plots the time record of the data.

    Parameters
    ----------
    fig, ax : Figure, Axis (optional)
        If a figure AND axis are provided, they will be used for the plot. if not provided, a
        new plot will automatically be created.

    Returns
    -------
    fig, ax : Figure and Axis
        The Figure and Axis handles of the plot that was used.
    """
    def __init__(self, *args, **kwargs):
        del args
        for key, value in kwargs.items():
            setattr(self, key, value)

    def plot(self, freqs, fig=None, ax=None):
        if not fig:
            fig, ax = plt.subplots()
        ax.plot(freqs, self.values(freqs), label=self.label)
        ax.set_xscale('log')
        ax.set_xlabel('Frequency / Hz')
        ax.set_ylabel('phase noise / dBc/Hz')
        plt.grid(True, which = 'both', ls = '-')
        return fig, ax

class JohnsonNoise(PhaseNoiseModel):
    """
    Thermal noise in dBc / Hz

    Parameters
    ----------
    signal_power : float
        Signal power in dBm / Hz
    temperature : float (default 300.)
        Temperature in kelvin
    
    Attributes
    signal_power : float
    temperature : float
    """
    def __init__(self, signal_power, temperature=300., label='Thermal noise'):
        super().__init__(temperature=temperature, label=label)
        self.signal_power = signal_power
        
    def values(self, freqs):
        """
        The thermal noise in dBc/Hz over a given frequency axis

        Parameters
        ----------
        freqs : list_like
            The frequency in Hz. Is only used to determine the length of the requested array

        Returns
        -------
        noise : 1d array
            thermal noise in dBc/Hz
        """
        kb=1.380649e-23 # Boltzmann constant in J/K
        freqs = np.ones(len(freqs))
        # 1e-3 because normalized to mW, normalized to signal power, length of freqds
        noise = 10 * np.log10(4 * kb * self.temperature / 1e-3) * freqs - self.signal_power
        # subtract 3 dB since above quantity is defined as one-sided according to 
        # https://en.wikipedia.org/wiki/Johnson%E2%80%93Nyquist_noise
        noise -= 3
        return noise
    

class PhotonShotNoise(PhaseNoiseModel):
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
                resistivity=50, label='Photon shot noise'):
    
        super().__init__(radiant_sensitivity=radiant_sensitivity, resistivity=resistivity,
                label=label, optical_power=optical_power)
        self.signal_power = signal_power

    def values(self, freqs):
        """
        The shot noise in dBc/Hz over a given frequency axis

        Parameters
        ----------
        freqs : list_like
            The frequency in Hz. Is only used to determine the length of the requested array

        Returns
        -------
        noise : 1d array
            thermal noise in dBc/Hz
        """
        e  = 1.6e-19 # electron charge in C
        freqs = np.ones(len(freqs))
        noise = 10 * np.log10(2 * e * self.radiant_sensitivity * self.optical_power * 
            self.resistivity / 1e-3) * freqs - self.signal_power
        # FIXME: Assume this to be a one-sided distribution, but didn't check
        noise -= 3
        return noise

class NoiseFloor(PhaseNoiseModel):
    """
    Experimentally determined detection noise.

    Parameters
    ----------
    signal_power : float
        Signal power in dBm / Hz
    noise_floor : float
        measured noise floor in dBm / Hz
    divide_by : int (optional)
        fividy-by factor if prescaler was used
    """
    def __init__(self, signal_power, noise_floor, divide_by=1, label='Detection noise'):
        super().__init__(label=label, divide_by=divide_by)
        self.signal_power = signal_power
        self.noise_floor = noise_floor

    def values(self, freqs):
        """
        The noise floor in dBc/Hz over a given frequency axis

        Parameters
        ----------
        freqs : list_like
            The frequency in Hz. Is only used to determine the length of the requested array

        Returns
        -------
        noise : 1d array
            thermal noise in dBc/Hz
        """
        freqs = np.ones(len(freqs))
        noise = freqs * self.noise_floor + 20 * np.log10(self.divide_by) - self.signal_power
        # single sided 
        noise -= 3
        return noise 
