from scipy.signal import welch
import numpy as np
import matplotlib.pyplot as plt
import allantools

class FreqData():
    def __init__(self, frequencies, duration, divide_by=1):
        self.divide_by = divide_by
        self.freqs = frequencies
        self.mean_frequency = np.mean(self.freqs)
        self.duration = duration
        self.n_samples = len(self.freqs)
        self.sample_rate = int(self.n_samples/self.duration)

    def data_to_dict(self):
        data_dict = {
            'mean_frequency' : self.mean_frequency,
            'duration' : self.duration,
            'n_samples' : self.n_samples,
            'sample_rate' : self.sample_rate,
            'frequencies' : self.freqs,
            'divide_by' : self.divide_by
        }
        return data_dict

    def asd(self):
        f, Pxx = welch(self.freqs, self.sample_rate, ('kaiser', 100), 
            nperseg=1024, scaling='density')
        asd = self.divide_by * np.sqrt(Pxx)
        return SpectralDensity(f, asd, scaling='asd', base='freq')

    def adev(self, scaling=780e-9/2.99e8):
        freqs = np.array(self.freqs)*scaling
        tau_max = np.log10(len(self.freqs))
        taus = np.logspace(0,tau_max)/self.sample_rate
        (taus, adev, adeverror, _) = allantools.adev(freqs, data_type='freq',
             rate=self.sample_rate, taus=taus)
        return taus, adev, adeverror

    def plot_time_record(self):
        t = np.linspace(0,self.duration,num=self.n_samples)
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


class SpectralDensity():

    def __init__(self, freqs, density, scaling='asd', base='freq'):
        
        self._scaling = scaling
        self._base = base
        self.freqs = freqs

        # only one representation of the spectral density is set, the rest is
        # calculated when needed
        attr = '{}_{}'.format(self.scaling, self.base)
        setattr(self, '_'+attr, density)
        self._alias_density()

    def _alias_density(self):
        # aliasing based on scaling and base
        attr = '{}_{}'.format(self.scaling, self.base)
        self.density = getattr(self, attr)




    @property
    def base(self):
        return self._base
    @base.setter
    def base(self, base):
        assert base in ['freq', 'phase']
        self._base = base
        self._alias_density()

    @property
    def scaling(self):
        return self._scaling
    @scaling.setter
    def scaling(self, scaling):
        assert scaling in ['asd', 'psd']
        self._scaling = scaling
        self._alias_density()

    @property
    def asd_freq(self):
        if not hasattr(self, '_asd_freq'):
            self._asd_freq = 2 * np.pi * f * self._asd_phase
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
            self._psd_phase = self.psd_freq / (4 *  np.pi**2 * self.freqs**2)
        return self._psd_phase

    def plot(self):
        label_dict = {'asd_freq'  : 'ASD (Hz / $\\sqrt{\\mathrm{Hz}}$)',
                      'asd_phase' : 'ASD ($\\mathrm{rad} / \\sqrt{\\mathrm{Hz}}$)',
                      'psd_freq'  : 'PSD (Hz${}^2$ / Hz)',
                      'psd_phase' : 'PSD (rad${}^2$ / Hz)'}
        attr = '{}_{}'.format(self.scaling, self.base)
        label = label_dict[attr]

        fig, ax = plt.subplots()
        ax.loglog(self.freqs, self.density)
        ax.set_xlabel('Frequency / Hz')
        ax.set_ylabel(label)
        plt.grid(True, which = 'both', ls = '-')
        return fig, ax

def merge(sds):
       
    freqs = np.concatenate([sd.freqs for sd in sds])
    # FIXME: test if scaling and base is equal for all SpectralDensities
    density = np.concatenate([sd.density for sd in sds])
    freqs, idx = np.unique(freqs, return_index=True)
    density = density[idx]
    return SpectralDensity(freqs, density, 
        scaling=sds[0].scaling, base=sds[0].base)
