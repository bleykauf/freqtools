from scipy.signal import welch
import numpy as np
import matplotlib.pyplot as plt
import allantools

class CounterData():
    """
    Counter data, i.e. a time series of frequency data.

    Parameters
    ----------
    freqs : list_like
        measured frequencies in Hz
    duration : float
        duration of counter measurement the measurement in s
    divide_by : int (optional, default 1)
        if a prescaler was used, CounterData will automatically scale the resulting spectral 
        densities.

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
    sample_rate : in Hz
        sampling rate in Hz
    asd
    adev
    plot_time_record
    plot_adev
    data_to_dict
    """
    def __init__(self, freqs, duration, divide_by=1):
        self.divide_by = divide_by
        self.freqs = freqs
        self.mean_frequency = np.mean(self.freqs)
        self.duration = duration
        self.n_samples = len(self.freqs)
        self.sample_rate = int(self.n_samples/self.duration)

    def asd(self, method='welch'):
        """
        Caluclates the two-sided amplitude spectral density (ASD) in Hz/sqrt(Hz).

        Parameters
        ----------
        method : {'welch'}
            not used for now
            
        Returns
        -------
        asd : SpectralDensity
            Creates a SpectralDenisty object and initializes it with the calculated ASD
        """
        # FIXME: provide other methods to calculate the ASD
        if method == 'welch':
            f, Pxx = welch(self.freqs, self.sample_rate, ('kaiser', 100), 
                nperseg=1024, scaling='density')
            asd = self.divide_by * np.sqrt(Pxx)
        return SpectralDensity(f, asd, scaling='asd', base='freq')

    def adev(self, scaling=780e-9/2.99e8):
        """
        Calculates the Allan deviation of the data.

        Returns
        -------
        taus, adev, adeverror : list
            The taus for which the Allan deviation has been calculated, the adev at these taus and
            their statistical error.
        """
        freqs = np.array(self.freqs)*scaling
        tau_max = np.log10(len(self.freqs))
        taus = np.logspace(0,tau_max)/self.sample_rate
        (taus, adev, adeverror, _) = allantools.adev(freqs, data_type='freq',
             rate=self.sample_rate, taus=taus)
        return taus, adev, adeverror

    def plot_time_record(self, fig=None, ax=Non):
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
        t = np.linspace(0,self.duration,num=self.n_samples)
        if not fig:
            fig, ax = plt.subplots()
        ax.plot(t, self.freqs, 
            label = 'Mean frequency: ({:3f}+/-{:3f}) MHz'.format(
                self.mean_frequency*1e-6,
                np.std(self.freqs)*1e-6
                )
            )
        ax.set_xlabel('time t (s)')
        ax.set_ylabel('frequency deviation (Hz)')
        ax.legend()
        plt.grid(b='on', which = 'minor', axis = 'both')
        plt.box(on='on')
        return fig, ax

    def plot_adev(self, fig=None, ax=None):
        """
        Plots the Allan deviation of the data.

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
        taus, adev, adeverror = self.adev()
        if not fig:
            fig, ax = plt.subplots()
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.errorbar(taus, adev, yerr=adeverror)
        ax.set_xlabel('Averaging time t (s)')
        ax.set_ylabel(r'Allan deviation $\sigma_y(t)$')
        plt.grid(b='on', which = 'minor', axis = 'both')
        plt.box(on='on')
        return fig, ax

    def data_to_dict(self):
        """Saves all properties to a dict for saving to a file etc."""
        data_dict = {
            'mean_frequency' : self.mean_frequency,
            'duration' : self.duration,
            'n_samples' : self.n_samples,
            'sample_rate' : self.sample_rate,
            'frequencies' : self.freqs,
            'divide_by' : self.divide_by
        }
        return data_dict

class SpectralDensity():
    """
    A class to make it easy to convert between ASD(f) and PSD(f) of both phase and frequency. If
     either of these densities is provided, all other representations can be calculated.

    Parameters
    ----------
    freqs : list_like
        the Fourier frequencies in Hz
    density : list_like
        PSD(f), ASD(f) with respect to frequency or phase, depending on `scaling` and `base`. The
        units are assumed to be without prefixes, i.e. Hz**2/Hz for the PSD(f) of the frequency
    scaling : {'asd', 'psd'}, default 'asd'
    base : {'freq', 'phase'}, default 'freq'

    Attributes
    ----------
    scaling : {'asd', 'psd'}
    base : {'freq', 'phase'}
    density : 1darray
        The density in the representation determiend by `base` and `scaling`
    asd_freq, asd_phase, psd_freq, psd_phase : 1darray
        the spectral density in differenct representations. `density` maps to on of these 
        properties
    plot
    """
    def __init__(self, freqs, density, scaling='asd', base='freq'):
        self._scaling = scaling
        self._base = base
        self.freqs = np.array(freqs)

        # only one representation of the spectral density is set, the rest is calculated when 
        # needed
        attr = '{}_{}'.format(self.scaling, self.base)
        setattr(self, '_'+attr, density)
        self._alias_density()

    def _alias_density(self):
        # changes what self.density returns. This fuction is called, whenever `base` or `scaling`
        # are changed.
        attr = '{}_{}'.format(self.scaling, self.base)
        self.density = getattr(self, attr)

    @property
    def base(self):
        return self._base
    @base.setter
    def base(self, base):
        assert base in ['freq', 'phase']
        self._base = base
        # change what self.density returns
        self._alias_density()

    @property
    def scaling(self):
        return self._scaling
    @scaling.setter
    def scaling(self, scaling):
        assert scaling in ['asd', 'psd']
        self._scaling = scaling
        # change what self.density returns
        self._alias_density()

    @property
    def asd_freq(self):
        if not hasattr(self, '_asd_freq'):
            self._asd_freq = self.freqs * self._asd_phase
        return self._asd_freq

    @property
    def asd_phase(self):
        if not hasattr(self, '_asd_phase'):
            self._asd_phase = np.sqrt(self.psd_phase)
        return self._asd_phase

    @property
    def psd_freq(self):
        if not hasattr(self, '_psd_freq'):
            self._psd_freq = self.asd_freq**2 
        return self._psd_freq            

    @property
    def psd_phase(self):
        if not hasattr(self, '_psd_phase'):
            self._psd_phase = self.psd_freq / self.freqs**2
        return self._psd_phase

    def plot(self, fig=None, ax=None):
        """
        Plots the spectral density in the representation determiend by `base` and `scaling`
        
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
        label_dict = {'asd_freq'  : 'ASD (Hz / $\\sqrt{\\mathrm{Hz}}$)',
                      'asd_phase' : 'ASD ($\\mathrm{rad} / \\sqrt{\\mathrm{Hz}}$)',
                      'psd_freq'  : 'PSD (Hz${}^2$ / Hz)',
                      'psd_phase' : 'PSD (rad${}^2$ / Hz)'}
        attr = '{}_{}'.format(self.scaling, self.base)
        label = label_dict[attr]
        if not fig:
            fig, ax = plt.subplots()
        fig, ax = plt.subplots()
        ax.loglog(self.freqs, self.density)
        ax.set_xlabel('Frequency / Hz')
        ax.set_ylabel(label)
        plt.grid(True, which = 'both', ls = '-')
        return fig, ax

def merge(sds):
    # merging instances of SpectralDensity   
    freqs = np.concatenate([sd.freqs for sd in sds])
    # FIXME: test if scaling and base is equal for all SpectralDensities
    density = np.concatenate([sd.density for sd in sds])
    freqs, idx = np.unique(freqs, return_index=True)
    density = density[idx]
    return SpectralDensity(freqs, density, 
        scaling=sds[0].scaling, base=sds[0].base)