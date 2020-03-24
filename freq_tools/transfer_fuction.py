import matplotlib.pyplot as plt
import numpy as np
from .freq_data import PhaseNoise


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