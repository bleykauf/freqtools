import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def import_csv(filename, as_class, delimiter=',', **kwargs):
    """
    Import data from a .csv and create a FreqData object from it.
    All of the keyworded arguments needed to construct the class inheriting from Freqdata(e.g. 
    `rbw` for `SpectrumAnalyzerData`) have to be passed to the function as keyworded arguments. 

    Parameters
    ----------
    filename : str
    as_class : FreqData
        FreqData or one of its subclasses

    Returns
    -------
    instance : as defined in as_class
    """
    data = np.genfromtxt(filename, dtype=float , delimiter=delimiter, comments='%', 
        names=['freqs','values'])
    instance = as_class(data['freqs'], data['values'], **kwargs)
    return instance


class FreqData():
    def __init__(self, freqs, values, **kwargs):
        self.freqs = np.array(freqs)
        self.values = np.array(values)
        assert(len(self.values) == len(self.values))
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        new_instance = copy.deepcopy(self)
        new_instance.freqs = new_instance.freqs[key]
        new_instance.values = new_instance.values[key]
        return new_instance

    def __len__(self):
        return len(self.freqs)

    interpolation_options = {'kind' : 'linear',
                             'fill_value' : 0.0,
                             'bounds_error' : False}

    def interpolated_values(self, freqs):
        func = interp1d(self.freqs, self.values, **self.interpolation_options)
        return func(freqs)
        
    def join(self, others):
        """
        Joins another instance of FreqData.       
        Only one value per fourier frequency is used, with preference for the ones appearing in `self`,
        followed by the first items in `other`.

        Parameters
        ----------
        others : (list of) FreqData
        """
        if not isinstance(others, list):
            # if only one FreqData is to be joined
            others = [others]
        freqs = np.concatenate([self.freqs, *[other.freqs for other in others]])
        values = np.concatenate([self.values, *[other.values for other in others]])
        # if there are the same `freqs` multiple times, prefere the `values` in `self` or the first
        #  that appear in `other`
        self.freqs, idx = np.unique(freqs, return_index=True)
        self.values = values[idx]


class SpectralDensity(FreqData):
    """
    A class to make it easy to convert between ASD(f) and PSD(f) of both phase and frequency. If
     either of these densities is provided, all other representations can be calculated. 

    Parameters
    ----------
    freqs : list_like
        the Fourier frequencies in Hz
     : list_like
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
    def __init__(self, freqs, values, scaling='asd', base='freq', two_sided=True):
        super().__init__(freqs, values, two_sided=two_sided)
        self._base = base
        self._scaling = scaling
        # only one representation of the spectral density is set, rest is calculated when needed
        attr = '{}_{}'.format(self.scaling, self.base)
        setattr(self, '_'+attr, values)
        self._alias_values()

    def _alias_values(self):
        # changes what self.values returns. This fuction is called, whenever `base` or `scaling`
        # are changed.
        attr = '{}_{}'.format(self.scaling, self.base)
        self.values = getattr(self, attr)

    @property
    def base(self):
        return self._base
    @base.setter
    def base(self, base):
        assert base in ['freq', 'phase']
        self._base = base
        # change what self.values returns
        self._alias_values()

    @property
    def scaling(self):
        return self._scaling
    @scaling.setter
    def scaling(self, scaling):
        assert scaling in ['asd', 'psd']
        self._scaling = scaling
        # change what self.values returns
        self._alias_values()

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

    def to_phase_noise(self, **kwargs):
        """
        Converts the PSD of the phase, S_phi(f), to phase_noise L(f). Depending on whether the
        densities are one- or two-sided, the scaling will be adjusted [1].

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
        return PhaseNoise(self.freqs, L, **kwargs)

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
        ax.loglog(self.freqs, self.values)
        ax.set_xlabel('Frequency / Hz')
        ax.set_ylabel(label)
        plt.grid(True, which = 'both', ls = '-')
        return fig, ax


class SpectrumAnalyzerData(FreqData):
    """
    Class holding data from a spectrum analyzer.

    Parameters
    ----------
    values : list_like
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
    values : 1darray
        signal level in dBm. it is the level of the measured signal, i.e. the divided signal if a 
        prescaler was used, c.p. `divide_by`
    rbw : 1darray
        resolution bandwidth in Hz
    divide_by : int
        divide-by value if a prescaler was used
    label : str
        optional label used for some plots
    """

    def __init__(self, freqs, values, rbw=1, divide_by=1, label=''):
        super().__init__(freqs, values, rbw=rbw, divide_by=divide_by, label=label)
        self.values += 20*np.log10(self.divide_by)

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

        peak_level = max(self.values)

        center_freq =self.freqs[self.values == peak_level]

        if sideband == 'left':
            selected_freqs = self.freqs < center_freq
        elif sideband == 'right':
            selected_freqs = self.freqs > center_freq

        freqs = self.freqs[selected_freqs] - center_freq
        level = self.values[selected_freqs]

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
        ax.plot(self.freqs, self.values)
        ax.set_xlabel('frequency / Hz')
        ax.set_ylabel('level / dBm')
        return fig, ax


class PhaseNoise(FreqData):
    """
    Class for phase noise L(f).

    Parameters
    ----------
    freqs : list_like
        Fourier frequencies in Hz
    values : list_like
        The phase noise in dBc/Hz
    divide_by : int (optional)
        If a prescaler was used, the phase noise will automatically be corrected to reflect the 
        phase noise of the original oscillator
    label : str (optional)
        The label used for plotting
    
    Attributes
    ----------
    plot
    to_rad2_per_Hz
    accumulate
    plot
    plot_accumulate
    ylabel : str
        the ylabel used for plotting
    yscale : {'log', 'linear'}
        determines how the y axis is scaled 
    """
    def __init__(self, freqs, values, label='', divide_by=1):
        super().__init__(freqs, values, label=label, divide_by=divide_by)
        self.values += 20*np.log10(self.divide_by)
         # for plotting
        self.ylabel = 'phase noise / dBc/Hz'
        self.yscale = 'linear'

    def to_rad2_per_Hz(self, one_sided=True):
        """
        Converts from L(f) in dBc/Hz to S_phi(f) in rad**2/Hz.

        Parameters
        ----------
        one_sided : bool (default True)
            determines whether the returned  spectral density is one- or two-sided (default one-
            sided)

        Returns
        -------
        S_phi : 1darray
            The spectral density of the phase noise in rad**2/Hz

        References
        ----------
        [1] IEEE Standard Definitions of Physical Quantities for Fundamental Frequency and Time 
        Metrology — Random Instabilities (IEEE Std 1139™-2008)
        """
        # factor 1/10 in exponent because decibel are used
        S_phi = 10**(self.values/10)
        if one_sided:
            S_phi *= 2 # one-sided distributions have a factor 2, see Table A1 in [1]
        return S_phi

    def accumulate(self, convert_to_g=False, k_eff=1.61e7, T=260e-3):
        """
        The accumulated phase noise calculated by integrating from the highest to lowest Fourier
        frequency as in Fig. 3.17 of Christian Freier's PhD thesis [1]. 

        Parameters
        ---------- 
        convert_to_g : bool (default False)
            If True, the phase noise will be converted to atom interferometer phase noise, using
            `k_eff` and `T`
        k_eff, T : float 
            effective wavevector and interferometer time in 1/m and s, respectively. Default values
            are standard values for GAIN.

        Returns
        -------
        accumulated_noise : AccumulatedPhaseNoise
            units depending in rad or nm/s**2, depending on `convert_to_g`

        Reference
        ---------
        [1] C. Freier - Atom Interferometry at Geodetic Observatories (2017), PhD Thesis
        """
        S_phi = self.to_rad2_per_Hz()
        accumulated_noise = []
        for k in np.arange(len(self.values)):
            accumulated_noise.append(np.trapz(S_phi[k:], x=self.freqs[k:]))
        accumulated_noise = np.sqrt(np.array(accumulated_noise))
        ylabel = 'accumulated phase noise / rad'
        if convert_to_g:
            accumulated_noise = accumulated_noise / (k_eff * T**2) * 1e9
            ylabel = 'accumulated AI noise / nm/s²'
        accumulated_noise = AccumulatedPhaseNoise(self.freqs, accumulated_noise, label=self.label)
        accumulated_noise.ylabel = ylabel
        accumulated_noise.yscale = 'log'
        return accumulated_noise

    def plot(self, fig=None, ax=None):
        """
        Plots the phase noise as a function of the Fourier frequency.

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
        ax.set_xscale('log')
        ax.set_yscale(self.yscale)
        ax.plot(self.freqs, self.values, label=self.label)
        ax.set_ylabel(self.ylabel)
        ax.set_xlabel('frequency $f$ / Hz')
        ax.grid(True, which='both', axis='both')
        return fig, ax


class AccumulatedPhaseNoise(FreqData):

    def __init__(self, freqs, values, label=''):
        super().__init__(freqs, values, label=label)

    # "inherit" only this method because most methods of PhaseNoise don't make sense for 
    # accumulated phase noise.
    plot = PhaseNoise.__dict__['plot']
