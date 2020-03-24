from scipy.signal import welch
import numpy as np
import matplotlib.pyplot as plt
import allantools

from .freq_data import SpectralDensity

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