import matplotlib.pyplot as plt
import numpy as np

def import_phase_noise(filename, divide_by=1, delimiter=','):
    data = np.genfromtxt(filename, delimiter=delimiter, names=['freqs', 'noise'])
    return PhaseNoise(data['freqs'], data['noise'], divide_by=divide_by)

class PhaseNoise():
    
    def __init__(self, freqs, noise, divide_by=1):
        self.freqs = freqs
        self.divide_by = divide_by
        self.noise = noise + 20*np.log10(self.divide_by)

    def plot(self, fig=None, ax=None):
        if not fig:
            fig, ax = plt.subplots()
        ax.grid()
        ax.set_xscale('log')
        ax.plot(self.freqs, self.noise)
        ax.set_ylabel('phase noise / dBc/Hz')
        ax.set_xlabel('frequency $f$ / Hz')
        return fig, ax

class MachZehnderTransferFunction():
    def __init__(self, T=260e-3, tau=30e-6, normalize_to_g=False, k_eff=1.61e7):
        self.T = T
        self.tau = tau
        self.Omega_r = np.pi/2 / self.tau
        self.normalize_to_g = normalize_to_g
        self.k_eff = k_eff

    def tf(self, f, k_eff=1.61e7):
        # transfer function at frequency f
        omega = 2 * np.pi * f
        H_ai = (4j * omega * self.Omega_r) / (omega**2 - self.Omega_r**2) * \
            np.sin(omega * (self.T+2*self.tau) / 2) * \
            (np.cos(omega * (self.T+2*self.tau) / 2) + self.Omega_r/omega * \
            np.sin(omega*self.T/2))
        if self.normalize_to_g:
            H_ai = H_ai / (self.k_eff*self.T**2)
        return abs(H_ai)

    def plot(self, f, window=1, f0=None):
        if not f0:
            # set the low pass corner frequency as the frequency above which 
            # moving average is used for transfer function plotting
            f0 = self.Omega_r / (2*np.pi*np.sqrt(3))
        # calculate for both regimes seperatre and stich together
        f1 = f[f<=f0]
        tf1 = self.tf(f1)
        f2 = _running_mean(f[f>f0], window)
        tf2 = _running_mean(self.tf(f[f>f0]), window)
        f = np.concatenate([f1, f2])
        tf = np.concatenate([tf1, tf2])
        fig, ax = plt.subplots()
        ax.loglog(f, tf)
        ax.set_xlabel('Frequency $f$ / Hz')
        if self.normalize_to_g:
            unit = ' / m/s²/rad'
        else:
            unit = ' / rad/rad'
        ax.set_ylabel('$|H(2\pi f)|$' + unit)
        return fig, ax

def _running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)