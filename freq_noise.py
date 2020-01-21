# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 15:35:42 2018

@author: Klaus
Countermessungen mit Pendulum
Abgeschreiben von lock_noise.py
"""

from time import sleep
from cnt90 import CNT90
import json
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import allantools
import datetime

plt.rcParams['figure.facecolor'] = 'lightgrey'

def append_number_to_filename(filename):
    """
    Adds a number suffix to a filename.
    """

    counter = 0
    while True:
        test_filename = filename + '_{}'.format(counter)
        try:
            open(test_filename + '.json', 'r')
        except FileNotFoundError:
            return test_filename
        counter += 1
        if counter > 9000:
            raise Exception('file name counter is over 9000!')

def frequency_measurement(meas_time=1, n_samples=10**5, channel='A', data_name='testing'):

    sample_rate = int(n_samples/meas_time)
    counter = CNT90('USB0::0x14EB::0x0091::956628::INSTR')

    print('starting measurement, t=', meas_time, 'sample rate', sample_rate)
    frequencies = counter.frequency_measurement(channel, meas_time, sample_rate)
    mean_frequency = np.mean(frequencies)    
    frequencies = frequencies - mean_frequency
    
    file_name = '_%s_%ds' % ('locked' if is_locked else 'free', meas_time)
    
    data_file_name = append_number_to_filename(data_name + file_name+'_freqnoise')
    
    with open(data_file_name + '.json', 'x') as f:
        json.dump({
            'measurement_time': meas_time,
            'measurement_rate': sample_rate,
            'frequencies': list(frequencies),
            'is_locked': is_locked
        }, f)
        
    time = np.linspace(0,meas_time,num=n_samples)
    
    plt.figure(figsize = (8,5))
    plt.plot(time, frequencies, label = 'Mean frequency: %.1f +/- %.1f' % (mean_frequency*1e-6, np.std(frequencies)*1e-6 ))
    plt.ylabel('Frequency / Hz')
    plt.xlabel('Time / s')
    plt.legend()
    plt.grid(True, which='both', ls='-')
    plt.savefig(data_file_name + '_timerec.pdf', dpi=300, bbox_inches='tight')
    plt.show()
        
    f, Pxx_spec = signal.welch(frequencies, sample_rate, ('kaiser', 100), nperseg=1024, scaling='density')
    plt.figure(figsize = (8,5))
    plt.loglog(f, np.sqrt(Pxx_spec))
    plt.xlabel('Frequency / Hz')
    plt.ylabel(r'Frequency noise ASD  / $Hz/\sqrt{Hz}$')
    plt.grid(True, which = 'both', ls = '-')
    plt.savefig(data_file_name + '.pdf', dpi=300,  bbox_inches='tight')
    plt.show()
    
    data_rel = np.array(frequencies)*767e-9/2.99e8
    # calculate the Allan deviation
    tau_max = np.log10(len(frequencies))
    taus = np.logspace(0,tau_max)/sample_rate
    (taus_used, adev, adeverror, _) = allantools.adev(data_rel, data_type='freq', rate=sample_rate, taus=taus)
        
    plt.figure(figsize = (8,5))
    # plot the Allan devation
    plt.subplot(111, xscale = 'log', yscale = 'log')
    plt.errorbar(taus_used, adev, yerr=adeverror)
    plt.xlabel('Averaging time t (s)')
    plt.ylabel(r'Allan deviation $\sigma_y(t)$')
    plt.grid(b='on', which = 'minor', axis = 'both')
    plt.box(on='on')
    plt.savefig(data_file_name + '_allan.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
# Measurement loop    
is_locked = False
n_samples = 10**2 # max. 10**5
channel = 'A'

for meas_time in [1, 10]:
    frequency_measurement(meas_time, n_samples=n_samples, channel=channel)
