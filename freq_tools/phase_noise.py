import matplotlib.pyplot as plt
import numpy as np

def import_phase_noise(filename, divide_by=1, delimiter=',', label=''):
    data = np.genfromtxt(filename, delimiter=delimiter, names=['freqs', 'noise'])
    return PhaseNoise(data['freqs'], data['noise'], divide_by=divide_by, label=label)

class PhaseNoise():
    
    def __init__(self, freqs, noise, divide_by=1, label=''):
        self.freqs = freqs
        self.divide_by = divide_by
        # noise is L(f) in dBc/Hz
        self.noise = noise + 20*np.log10(self.divide_by)
        self.label = label

    def to_rad2_per_Hz(self, single_sided=True):
        # factor 1/10 in exponent because decibel are used
        S_phi = 10**(self.noise/10)
        if single_sided:
            S_phi *= 2
        return S_phi

    def plot(self, fig=None, ax=None):
        if not fig:
            fig, ax = plt.subplots()
        ax.set_xscale('log')
        ax.plot(self.freqs, self.noise, label=self.label)
        ax.set_ylabel('phase noise / dBc/Hz')
        ax.set_xlabel('frequency $f$ / Hz')
        ax.grid(True, which='both', axis='both')
        return fig, ax

    def accumulate(self, norm_to_g=False, k_eff=1.61e7, T=260e-3):
        S_phi = self.to_rad2_per_Hz()
        accumulated_noise = []
        for k in np.arange(len(self.noise)):
            accumulated_noise.append(np.trapz(S_phi[k:], x=self.freqs[k:]))
        accumulated_noise = np.sqrt(np.array(accumulated_noise))
        if norm_to_g:
            accumulated_noise = accumulated_noise / (k_eff * T**2) * 1e9
        return accumulated_noise

    def plot_accumulated(self, fig=None, ax=None, norm_to_g=False, k_eff=1.61e7, T=260e-3):
        accumulated_noise = self.accumulate(norm_to_g, k_eff, T)
        if not fig:
            fig, ax = plt.subplots()
        ax.loglog(self.freqs, accumulated_noise, label=self.label)
        if norm_to_g:
            ylabel = 'accumulated AI noise / nm/s²'
        else:
            ylabel = 'accumulated phase noise / rad'
        ax.set_ylabel(ylabel)
        ax.set_xlabel('frequency $f$ / Hz')
        ax.grid(True, which='both', axis='both')
        return fig, ax

class MachZehnderTransferFunction():
    def __init__(self, T=260e-3, tau=30e-6):
        self.T = T
        self.tau = tau
        self.Omega_r = np.pi/2 / self.tau

    def tf(self, f):
        # magnitude of transfer function at frequency f
        omega = 2 * np.pi * f
        H_ai = (4j * omega * self.Omega_r) / (omega**2 - self.Omega_r**2) * \
            np.sin(omega * (self.T+2*self.tau) / 2) * \
            (np.cos(omega * (self.T+2*self.tau) / 2) + self.Omega_r/omega * \
            np.sin(omega*self.T/2))
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
        ax.set_ylabel(r'$|H(2\pi f)|$ / rad/rad')
        return fig, ax

    def scale_noise(self, noise):
        S_phi_scaled =  self.tf(noise.freqs)**2 * noise.to_rad2_per_Hz()
        # convert back to dBc/Hz, factor 2 because single-sidedness of S_phi
        L_scaled = 10 * np.log10(S_phi_scaled/2)
        return PhaseNoise(noise.freqs, L_scaled, label=noise.label)

def _running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)