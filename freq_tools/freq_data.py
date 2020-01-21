from scipy.signal import welch
import numpy as np
import matplotlib.pyplot as plt
import allantools

class FreqData():
    def __init__(self, frequencies, duration, label='testing'):
        self.mean_frequency = np.mean(frequencies)
        self.frequencies = frequencies - self.mean_frequency
        self.duration = duration
        self.n_samples = len(self.frequencies)
        self.sample_rate = int(self.n_samples/duration)

    def data_to_dict(self):
        data_dict = {
            'mean_frequency' : self.mean_frequency,
            'duration' : self.duration,
            'n_samples' : self.n_samples,
            'sample_rate' : self.sample_rate,
            'frequencies' : self.frequencies
        }
        return data_dict

    def asd(self):
        # only calculate asd once
        f, Pxx = welch(self.frequencies, self.sample_rate, ('kaiser', 100),
            nperseg=1024, scaling='density')
        return (f, np.sqrt(Pxx))

    def adev(self, scaling=780e-9/2.99e8):
        freqs = np.array(self.frequencies)*scaling
        tau_max = np.log10(len(self.frequencies))
        taus = np.logspace(0,tau_max)/self.sample_rate
        (taus, adev, adeverror, _) = allantools.adev(freqs, data_type='freq',
             rate=self.sample_rate, taus=taus)
        return taus, adev, adeverror

    def plot_time_record(self):
        t = np.linspace(0,self.duration,num=self.n_samples)
        fig, ax = plt.subplots()
        ax.plot(t, self.frequencies, 
            label = 'Mean frequency: ({}+/-{}) MHz'.format(
                self.mean_frequency*1e-6,
                np.std(self.frequencies)*1e-6
                )
            )
        ax.set_xlabel('Averaging time t (s)')
        ax.set_ylabel(r'Allan deviation $\sigma_y(t)$')
        plt.grid(b='on', which = 'minor', axis = 'both')
        plt.box(on='on')
        return fig, ax
    
    def plot_asd(self):
        fig, ax = plt.subplots()
        asd = self.asd()
        ax.loglog(asd[0], asd[1])
        ax.set_xlabel('Frequency / Hz')
        ax.set_ylabel(r'Frequency noise ASD  / $Hz/\sqrt{Hz}$')
        plt.grid(True, which = 'both', ls = '-')
        return fig, ax

    def plot_adev(self):
        taus, adev, adeverror = self.adev()
        fig, ax = plt.subplots()
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.errorbar(taus, adev, yerr=adeverror)
        ax.set_xlabel('Averaging time t (s)')
        ax.set_ylabel(r'Allan deviation $\sigma_y(t)$')
        plt.grid(b='on', which = 'minor', axis = 'both')
        plt.box(on='on')
        return fig, ax


    

