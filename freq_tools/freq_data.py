from scipy.signal import welch
import numpy as np
import matplotlib.pyplot as plt
import allantools
from .phase_noise import PhaseNoise

def import_spectrum_analyzer_data(filename, rbw, divide_by=1, delimiter=',', label=''):
    data = np.genfromtxt(filename, dtype=float , delimiter='\t', comments='%', names=['freqs','level'])
    spectrum_analyzer_data = SpectrumAnalyzerData(data['level'], data['freqs'], rbw=rbw, label=label)
    return spectrum_analyzer_data

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
    sample_rate : float
        sampling rate in Hz
    divide_by : int
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
        return SpectralDensity(f, asd, scaling='asd', base='freq', two_sided=True)

    def adev(self, scaling=780e-9/2.99e8):
        """
        Calculates the Allan deviation of the data.

        Parameters
        ----------
        scaling : float (optional)
            default scaling for the adev is the frequency of the rubidium D2 line

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

    def plot_time_record(self, fig=None, ax=None):
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


class SpectrumAnalyzerData():
    """
    Class holding data from a spectrum analyzer.

    Parameters
    ----------
    level : list_like
        the level at `freqs` in dBm
    freqs : list_like
        frequencies (x axis) of the signal alayzer in Hz
    rbw : float
        resolution bandwidth in Hz
        label : str (optional)
            Optional label used for some plots and passed
    Attributes
    ----------
    freqs : 1darray
        frequencies in Hz
    level : 1darray
        signal level in dBm. it is the level of the measured signal, i.e. the divided signal if a 
        prescaler was used, c.p. `divide_by`
    rbw : 1darray
        resolution bandwidth in Hz
    divide_by : int
        divide-by value if a prescaler was used
    label : str
        optional label used for some plots
    """

    def __init__(self, level, freqs, rbw, divide_by=1, label=''):
        self.freqs = np.array(freqs)
        self.divide_by = divide_by
        self.level = np.array(level) + 20*np.log10(self.divide_by)
        self.rbw = rbw
        self.label = label

    def to_phase_noise(self, sideband='right', label=''):
        """
        Finds the peak and calculate the 
        Converts the PSD of the phase, S_phi(f), to phase_noise L(f). Depending on whether the
        densities are one- or two-sided, the scaling will be adjusted [1].

        Parameters
        ----------
        label : str (optional)
            Optionally pass a label to the PhaseNoise constructor

        Returns
        -------
        phase_phase : PhaseNoise
        """
        # see Table A.1 in [1] for the conversion from S_phi(f) and L(f)
        if label == '':
            label = self.label

        peak_level = max(self.level)

        center_freq =self.freqs[self.level == peak_level]

        if sideband == 'left':
            selected_freqs = self.freqs < center_freq
        elif sideband == 'right':
            selected_freqs = self.freqs > center_freq

        freqs = self.freqs[selected_freqs] - center_freq
        level = self.level[selected_freqs]

        # convert from dBm to dBc / Hz, no factor 1/2 because we analyse the single-sideband or 
        # two-sided spectral density
        noise = (level - peak_level) - 10 * np.log10(self.rbw)

        # divide_by already processed in __init__
        phase_noise = PhaseNoise(freqs, noise, label=self.label, divide_by=1)
        return phase_noise

    def plot(self, fig=None, ax=None):
        """
        Plots the spectrum.
        
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
        if not fig:
            fig, ax = plt.subplots()
        ax.plot(self.freqs, self.level)
        ax.set_xlabel('frequency / Hz')
        ax.set_ylabel('level / dBm')
        return fig, ax


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
        units are assumed to be without prefixes, i.e. Hz**2/Hz for the PSD(f) of the frequency. 
    scaling : {'asd', 'psd'}, default 'asd'
    base : {'freq', 'phase'}, default 'freq'
    two_sided : bool (default True)
        Specifies whether the two_sided spectral densities are used. If set to False, one-sided 
        spectral densities are assumed.

    Attributes
    ----------
    scaling : {'asd', 'psd'}
    base : {'freq', 'phase'}
    density : 1darray
        The density in the representation determiend by `base` and `scaling`
    asd_freq, asd_phase, psd_freq, psd_phase : 1darray
        the spectral density in differenct representations. `density` maps to on of these 
        properties
    """
    def __init__(self, freqs, density, scaling='asd', base='freq', two_sided=True):
        self._scaling = scaling
        self._base = base
        self.freqs = np.array(freqs)
        self.two_sided = two_sided

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

    def to_phase_noise(self, label=''):
        """
        Converts the PSD of the phase, S_phi(f), to phase_noise L(f). Depending on whether the
        densities are one- or two-sided, the scaling will be adjusted [1].

        Parameters
        ----------
        label : str (optional)
            Optionally pass a label to the PhaseNoise constructor

        Returns
        -------
        phase_phase : PhaseNoise

        References
        ----------
        [1] IEEE Standard Definitions of Physical Quantities for Fundamental Frequency and Time 
            Metrology — Random Instabilities (IEEE Std 1139™-2008)
        """
        # see Table A.1 in [1] for the conversion from S_phi(f) and L(f)
        if self.two_sided:
            L = self.psd_phase 
        else:
            L = self.psd_phase / 2
        # convert to dBc/Hz
            L = 10 * np.log10(L)
        return PhaseNoise(self.freqs, L, label=label)

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

def merge(to_merge, label=''):
    """
    Merges two instances of frequency or phase data. Supported classes are:
    
    - SpectralDensity (e.g. from two different counter measurements with  difference sample rates)
    - PhaseNoise (for example for different frequency ranges)
    
    For each Fourier frequency, only the density value of the object that first appears in
    `sto_merge` is used.

    Parameters
    ----------
    to_merge : list of objects to merge
    
    Returns
    -------
    merged : type of `to_merge`
    """
    # FIXME: better typechecking
    if isinstance(to_merge[0], SpectralDensity):
        freqs = np.concatenate([sd.freqs for sd in to_merge])
        # FIXME: test if scaling and base is equal for all SpectralDensities
        density = np.concatenate([sd.density for sd in to_merge])
        # use the values that first appears in `to_merge`
        freqs, idx = np.unique(freqs, return_index=True)
        density = density[idx]
        return SpectralDensity(freqs, density, scaling=to_merge[0].scaling, base=to_merge[0].base)
    elif isinstance(to_merge[0], PhaseNoise):
        freqs = np.concatenate([L.freqs for L in to_merge])
        noise = np.concatenate([L.noise for L in to_merge])
        # use the values that first appears in `to_merge`
        freqs, idx = np.unique(freqs, return_index=True)
        noise = noise[idx]
        if label == '':
            label = to_merge[0].label
        return PhaseNoise(freqs, noise, divide_by=1, label=label)
    else:
        raise TypeError('Unsupported datatype.')



