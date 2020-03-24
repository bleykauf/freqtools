from scipy.signal import welch
import numpy as np
import matplotlib.pyplot as plt
import allantools

from .freq_data import SpectralDensity

try:
    import scisave
except ImportError:
    has_scisave = False
else:
    has_scisave = True


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
    def __init__(self, freqs, duration, divide_by=1, **kwargs):
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
        # TODO: provide other methods to calculate the ASD
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
        t = np.linspace(0, self.duration, num=self.n_samples)
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

def import_json(filename, silent=False):
    if not has_scisave:
        ImportError('scisave (https://git.physik.hu-berlin.de/pylab/scisave) is required.')
        return None
    else:
        data = scisave.load_measurement(filename, silent=silent)
        freqs = data['results']['frequencies']
        device_settings = data['device_settings']
        counter_data = CounterData(freqs, **device_settings)
        return counter_data


def lpsd(x, windowfcn=np.hanning, fmin=None, fmax=None, Jdes=1000, Kdes, Kmin, fs, xi=0.5):
    """
    LPSD Power spectrum estimation with a logarithmic frequency axis.
    
    Estimates the power spectrum or power spectral density of the time series x at JDES frequencies
    equally spaced (on a logarithmic scale) from `fmin` to `fmax`.
    
    Parameters
    ----------
    x : array_like
    time series to be transformed. 
    "We assume to have a long stream x(n), n=0, ..., N-1 of equally spaced input data sampled with 
    frequency fs. Typical values for N range from 10^4 to >10^6" - Section 8 of [1]

    windowfcn : function_handle
        Function handle to windowing function (default numpy.hanning) 
        "Choose a window function w(j, l) to reduce spectral leakage within the estimate. ... The 
        computations of the window function will be performed when the segment lengths L(j) have 
        been determined." - Section 8 of [1]
    
    fmin, fmax : float
        Lowest and highest frequency to estimate. 
        "... we propose not to use the first few frequency bins. The first frequency bin that 
        yields unbiased spectral estimates depends on the window function used. The bin is given by
         the effective half-width of the window transfer function." - Section 7 of [1].
        
    Jdes : int
        Desired number of Fourier frequencies (default 1000)
        "A typical value for J is 1000" - Section 8 of [1]
    
    Kdes : int
        Desired number of averages
    
    Kmin : int
        Minimum number of averages
    
    fs : float
        Sampling rate
    
    xi : float
        Fractional overlap between segments (0 <= xi < 1), default 0.5.
        See Figures 5 and 6 [1]. 
        "The amount of overlap is a trade-off between computational effort and flatness of the data
        weighting." [1]

    Returns
    -------
    Pxx : 1darray
        Vector of (uncalibrated) power spectrum estimates
    f : 1darray
        Vector of frequencies corresponding to Pxx
    C : dict
        dict containing calibration factors to calibrate Pxx into either power spectral density or
        power spectrum.  

        - Pxx: 
        - f: 
        - C: 

    Notes
    -----
    The implementation follows references [1] and [2] quite closely; in
    particular, the variable names used in the program generally correspond
    to the variables in the paper; and the corresponding equation numbers
    are indicated in the comments.
    
    References
    ----------
    [1] Michael Tröbs and Gerhard Heinzel, "Improved spectrum estimation
      from digitized time series on a logarithmic frequency axis," in
      Measurement, vol 39 (2006), pp 120-129.
        * http://dx.doi.org/10.1016/j.measurement.2005.10.010
        * http://pubman.mpdl.mpg.de/pubman/item/escidoc:150688:1
    
    [2] Michael Tröbs and Gerhard Heinzel, Corrigendum to "Improved
      spectrum estimation from digitized time series on a logarithmic
      frequency axis."       
    """
    # Originally implemented in Matlab by Tobin Fricke: https://github.com/tobin/lpsd
    # Translated from Matlab to Python by Rudolf W Byker (https://github.com/rudolfbyker/lpsd)
    # Adapted for freq_tools

    if not callable(windowfcn):
        raise TypeError("windowfcn must be callable")
    if not (fmax > fmin):
        raise ValueError("fmax must be greater than fmin")
    if not (Jdes > 0):
        raise ValueError("Jdes must be greater than 0")
    if not (Kdes > 0):
        raise ValueError("Kdes must be greater than 0")
    if not (Kmin > 0):
        raise ValueError("Kmin must be greater than 0")
    if not (Kdes >= Kmin):
        raise ValueError("Kdes must be greater than or equal to Kmin")
    if not (fs > 0):
        raise ValueError("fs must be greater than 0")
    if not (0 <= xi < 1):
        raise ValueError("xi must be: 0 <= xi 1")

    N = len(x)  # Table 1
    jj = np.arange(Jdes, dtype=int)  # Table 1

    if not (fmin >= float(fs) / N):  # Lowest frequency possible
        raise ValueError("The lowest possible frequency is {}, but fmin={}".format(float(fs) / N), fmin)
    if not (fmax <= float(fs) / 2):  # Nyquist rate
        raise ValueError("The Nyquist rate is {}, byt fmax={}".format(float(fs) / 2, fmax))

    g = np.log(fmax) - np.log(fmin)  # (12)
    f = fmin * np.exp(jj * g / float(Jdes - 1))  # (13)
    rp = fmin * np.exp(jj * g / float(Jdes - 1)) * (np.exp(g / float(Jdes - 1)) - 1)  # (15)

    # r' now contains the 'desired resolutions' for each frequency bin, given the rule that we want the resolution to be
    # equal to the difference in frequency between adjacent bins. Below we adjust this to account for the minimum and
    # desired number of averages.

    ravg = (float(fs) / N) * (1 + (1 - xi) * (Kdes - 1))  # (16)
    rmin = (float(fs) / N) * (1 + (1 - xi) * (Kmin - 1))  # (17)

    case1 = rp >= ravg  # (18)
    case2 = np.logical_and(
        rp < ravg,
        np.sqrt(ravg * rp) > rmin
    )  # (18)
    case3 = np.logical_not(np.logical_or(case1, case2))  # (18)

    rpp = np.zeros(Jdes)

    rpp[case1] = rp[case1]  # (18)
    rpp[case2] = np.sqrt(ravg * rp[case2])  # (18)
    rpp[case3] = rmin  # (18)

    # r'' contains adjusted frequency resolutions, accounting for the finite length of the data, the constraint of the
    # minimum number of averages, and the desired number of averages.  We now round r'' to the nearest bin of the DFT
    # to get our final resolutions r.
    L = np.around(float(fs) / rpp).astype(int)  # segment lengths (19)
    r = float(fs) / L  # actual resolution (20)
    m = f / r  # Fourier Tranform bin number (7)

    # Allocate space for some results
    Pxx = np.empty(Jdes)
    S1 = np.empty(Jdes)
    S2 = np.empty(Jdes)

    # Loop over frequencies.  For each frequency, we basically conduct Welch's method with the fourier transform length
    # chosen differently for each frequency.
    # TODO: Try to eliminate the for loop completely, since it is unpythonic and slow. Maybe write doctests first...
    for jj in range(len(f)):

        # Calculate the number of segments
        D = int(np.around((1 - xi) * L[jj]))  # (2)
        K = int(np.floor((N - L[jj]) / float(D) + 1))  # (3)

        # reshape the time series so each column is one segment  <-- FIXME: This is not clear.
        a = np.arange(L[jj])
        b = D * np.arange(K)
        ii = a[:, np.newaxis] + b  # Selection matrix
        data = x[ii]  # x(l+kD(j)) in (5)

        # Remove the mean of each segment.
        data -= np.mean(data, axis=0)  # (4) & (5)

        # Compute the discrete Fourier transform
        window = windowfcn(L[jj])  # (5)
        sinusoid = np.exp(-2j * np.pi * np.arange(L[jj])[:, np.newaxis] * m[jj] / L[jj])  # (6)
        data = data * (sinusoid * window[:, np.newaxis])  # (5,6)

        # Average the squared magnitudes
        Pxx[jj] = np.mean(np.abs(np.sum(data)) ** 2)  # (8)

        # Calculate some properties of the window function which will be used during calibration
        S1[jj] = sum(window)  # (23)
        S2[jj] = sum(window ** 2)  # (24)

    # Calculate the calibration factors
    C = {
        'PS': 2. / (S1 ** 2),  # (28)
        'PSD': 2. / (fs * S2)  # (29)
    }

    return Pxx, f, C