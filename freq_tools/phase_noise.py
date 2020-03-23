import matplotlib.pyplot as plt
import numpy as np

def import_phase_noise(filename, divide_by=1, delimiter=',', label=''):
    """
    Import phase noise data from a csv or similar file and create a PhaseNoise object from it. The
    csv export of Timelab works, no quarantee for other file formats.

    Parameters
    ----------
    filename : str
    divide_by: int
        prescaling factor if applicable
    delimieter : str (default ',')
        delimiter of the csv file
    label : str (optional)
        label passed to PhaseNoise
    """
    data = np.genfromtxt(filename, delimiter=delimiter, names=['freqs', 'noise'])
    return PhaseNoise(data['freqs'], data['noise'], divide_by=divide_by, label=label)


class PhaseNoise():
    """
    Class for phase noise L(f).

    Parameters
    ----------
    freqs : list_like
        Fourier frequencies in Hz
    noise : list_like
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
    """
    def __init__(self, freqs, noise, divide_by=1, label=''):
        self.freqs = np.array(freqs)
        self.divide_by = divide_by
        self.noise = np.array(noise) + 20*np.log10(self.divide_by)
        self.label = label

    def __getitem__(self, key):
        freqs = self.freqs[key]
        noise = self.noise[key]
        phase_noise = PhaseNoise(freqs, noise, divide_by=1, label=self.label)
        # change divide from 1 to the actual value to avoid scaling twice, since this is done in 
        # the PhaseNoise constructor
        phase_noise.divide_by = self.divide_by
        return phase_noise

    def __len__(self):
        return len(self.freqs)

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
        S_phi = 10**(self.noise/10)
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
        accumulated_noise : 1darray
            units depending in rad or nm/s**2, depending on `convert_to_g`

        Reference
        ---------
        [1] C. Freier - Atom Interferometry at Geodetic Observatories (2017), PhD Thesis
        """
        S_phi = self.to_rad2_per_Hz()
        accumulated_noise = []
        for k in np.arange(len(self.noise)):
            accumulated_noise.append(np.trapz(S_phi[k:], x=self.freqs[k:]))
        accumulated_noise = np.sqrt(np.array(accumulated_noise))
        if convert_to_g:
            accumulated_noise = accumulated_noise / (k_eff * T**2) * 1e9
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
        ax.plot(self.freqs, self.noise, label=self.label)
        ax.set_ylabel('phase noise / dBc/Hz')
        ax.set_xlabel('frequency $f$ / Hz')
        ax.grid(True, which='both', axis='both')
        return fig, ax

    def plot_accumulated(self, fig=None, ax=None, convert_to_g=False, k_eff=1.61e7, T=260e-3):
        """
        Plots the accumulated phase noise as a function of Fourier frequency.

         Parameters
        ----------
        fig, ax : Figure, Axis (optional)
            If a figure AND axis are provided, they will be used for the plot. if not provided, a
            new plot will automatically be created.
        convert_to_g, k_eff, T : 
            see documentation for `accumulate` for description.

        Returns
        -------
        fig, ax : Figure and Axis
            The Figure and Axis handles of the plot that was used.       
        """
        accumulated_noise = self.accumulate(convert_to_g, k_eff, T)
        if not fig:
            fig, ax = plt.subplots()
        ax.loglog(self.freqs, accumulated_noise, label=self.label)
        if convert_to_g:
            ylabel = 'accumulated AI noise / nm/s²'
        else:
            ylabel = 'accumulated phase noise / rad'
        ax.set_ylabel(ylabel)
        ax.set_xlabel('frequency $f$ / Hz')
        ax.grid(True, which='both', axis='both')
        return fig, ax


class MachZehnderTransferFunction():
    """
    Class for the transfer function G(2*pi*f) of a Mach Zehnder Atom interferometer [1]

    Parameters
    ----------
    T, tau : float
        The interferometer time and pulse duration in s

    Attributes
    ----------
    T : float
    tau : float
    Omega_r : float
        Rabi frequency in 1/s, calculated from `tau`

    References
    ----------
    [1] P. Cheinet et al. - Measurement of the sensitivity function in a time-domain atomic 
        interferometer 
    """
    def __init__(self, T=260e-3, tau=36e-6):
        self.T = T
        self.tau = tau
        self.Omega_r = np.pi/2 / self.tau

    def tf(self, f):
        """
        The complex transfer function.

        Parameters
        ----------
        f : list_like
            Fourier frequencies in Hz for which the transfer function is calculated

        Returns
        -------
        tf : 1darray
            transfer function in rad/rad.
        """
        omega = 2 * np.pi * f
        H_ai = (4j * omega * self.Omega_r) / (omega**2 - self.Omega_r**2) * \
            np.sin(omega * (self.T+2*self.tau) / 2) * \
            (np.cos(omega * (self.T+2*self.tau) / 2) + self.Omega_r/omega * \
            np.sin(omega*self.T/2))
        return H_ai

    def scale_noise(self, noise):
        """
        Scales phase noise with the suared magnitude of the transfer function as in Eq. (7) of [1].

        Parameters
        ----------
        noise : PhaseNoise

        Returns
        -------
        scaled_noise : PhaseNoise
        
        References
        ----------
        [1] P. Cheinet et al. - Measurement of the sensitivity function in a time-domain atomic 
            interferometer 
        """
        S_phi_scaled =  abs(self.tf(noise.freqs))**2 * noise.to_rad2_per_Hz(one_sided=True)
        # convert back to dBc/Hz, factor 2 because using one-sided S_phi
        L_scaled = 10 * np.log10(S_phi_scaled/2)
        return PhaseNoise(noise.freqs, L_scaled, label=noise.label)

    def plot(self, f, f0=None, window=1):
        """
        Plots the magnitude of the transfer function with optional averaging above a threshold
        frequency

        Parameters
        ----------
        f : 1darray
            Fourier frequencies in Hz for which the transfer function should be plotted
        f0 : float (optional)
            threshold frequency in Hz above which the transfer function is averaged
        window : int
            averaging window, i.e. number of frequencies used for the moving average above the 
            threshold frequency.
        """
        if not f0:
            f0 = max(f)
        # calculate for both regimes seperatre and stich together
        f1 = f[f<=f0]
        tf1 = self.tf(f1)
        f2 = _running_mean(f[f>f0], window)
        tf2 = _running_mean(self.tf(f[f>f0]), window)
        f = np.concatenate([f1, f2])
        tf = np.concatenate([tf1, tf2])
        fig, ax = plt.subplots()
        ax.loglog(f, abs(tf))
        ax.set_xlabel('Frequency $f$ / Hz')
        ax.set_ylabel(r'$|H(2\pi f)|$ / rad/rad')
        return fig, ax


def _running_mean(x, N):
    # stolen from https://stackoverflow.com/a/27681394/2750945
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)