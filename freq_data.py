from scipy.signal import welch
import numpy
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
            'duration' : self.duration = self.duration,
            'n_samples' : self.n_samples,
            'sample_rate' : self.sample_rate
            'frequencies' : self.frequencies
        }

    @property
    def asd(self):
        # only calculate asd once
        if not hasattr(self, '_asd'):
            f, Pxx = welch(self.frequencies, self.sample_rate, 
                            ('kaiser', 100), nperseg=1024, scaling='density')
            self._asd = (f, np.sqrt(Pxx))
        return self._asd

    def plot_asd(self):
        fig, ax = plt.subplots()
        ax.loglog(self.asd[0], self.asd[1])
        ax.set_xlabel('Frequency / Hz')
        plt.set_ylabel(r'Frequency noise ASD  / $Hz/\sqrt{Hz}$')
        plt.grid(True, which = 'both', ls = '-')
        return fig, ax




    

