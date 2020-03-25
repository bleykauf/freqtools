import matplotlib.pyplot as plt
import numpy as np
from .freq_data import PhaseNoise

try:
    import scisave
except ImportError:
    has_scisave = False
else:
    has_scisave = True

class TransferFunction():
    """
    Class for transfer functions measured e.g. by a network analyzer

    Parameters
    ----------
    freq : list_like
        Fourier frequencies of the transfer function in Hz
    magnitude, phase : list_like
        Magnitude (in dB or as a linear factor, depending on `is_in_dB`) and phase (in degree) of
        the transfer function.
    label : str (optional)
    is_in_dB : bool (default True)
        indicates if the magnitude is in dB. If False, magnitude is provided as a linear factor. In
        this case the magnitude is automatically converted to dB.
    norm_to_dc : bool (default False)
        if True, the magnitude will be normalized to the value of the lowest frequency. Might be 
        useful if comparing setups with different gains.
    phase_unwrap : bool (default True)
        unwraps the phase and removes jumps that might occur due to the measurment
    shift_phase : float (default 0)
        this value is added to the phase

    Attributes
    ----------
    freq, magnitude, phase : 1darray
        as described above. Note that magnitude is always in dB
    label : str
     Optional label
    """
    def __init__(self, freq, magnitude, phase, label='', is_in_dB=True, norm_to_dc=False,
                phase_unwrap=True, shift_phase=0):
        self.freq = np.array(freq)
        self.magnitude = np.array(magnitude)
        if not is_in_dB:
            # convert magnitude if not already provided in dB
            self.magnitude = 20*np.log10(self.magnitude)
        if norm_to_dc:
            # normalize magnitude to the magnitude at the lowest frequency, e.g. for comparing 
            # different setups
            self.magnitude = self.magnitude - self.magnitude[0]

        self.phase = np.array(phase)  + shift_phase
        if phase_unwrap:
            # remove discontinuities
            self.phase = 360 * np.unwrap(2*np.pi*self.phase/360) / (2*np.pi)
        self.label = label

    def bode_plot(self, fig=None, ax1=None, ax2=None, xlim=(0, 2e8), ylim=(-370, 10)):
        """
        Shows a Bode plot.

        Parameters
        ----------
        fig, ax1, ax2 : Figure, Axis
            figure and axis objects that are used for the plotting. If not provided, a new figure
            will be created. `ax1` is used for the magnitude, `ax2` for the phase.
        xlim : tuple (default (0, 2e8))
            the plot limits of the frequency axis
        ylim : tuple (default (-370, 10))
            The plot limits for the phase
        """
        
        if not fig:
            fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        
        ax1.plot(self.freq, self.magnitude, label=self.label)
        ax2.plot(self.freq, self.phase, label=self.label)
        
        ax2.set_xscale('log')
        ax2.set_xlabel('Frequency / Hz')
        ax1.set_ylabel('Magnitude / dB')
        ax2.set_ylabel('Phase / °')

        ax1.grid(True, which='major', axis='both')
        ax2.grid(True, which='major', axis='both')
        
        ax2.set_xlim([max(xlim[0], self.freq[0]),xlim[1]])
        ax2.set_ylim(ylim)
        
        return fig, (ax1, ax2)

def import_json(filename, silent=False, **kwargs):
    if not has_scisave:
        ImportError('scisave (https://git.physik.hu-berlin.de/pylab/scisave) is required.')
        tf = None
    else:
        data = scisave.load_measurement(filename, silent=silent)
        freqs = data['results']['frequency']
        magnitude = data['results']['magnitude']
        phase = data['results']['phase']
        tf = TransferFunction(freqs, magnitude, phase, **kwargs)
    return tf


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
        Scales phase noise with the squared magnitude of transfer function as in Eq. (7) of [1].

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