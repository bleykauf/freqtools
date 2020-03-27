import numpy as np
import matplotlib.pyplot as plt


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
