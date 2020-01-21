# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 15:35:42 2018

@author: Klaus
Auswertung

"""
from __future__ import unicode_literals
import json
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from scipy import signal
import allantools
import os

# %%
def allan(data, fs):
    data_rel = data*767e-9/2.99e8
    # calculate the Allan deviation
    tau_max = np.log10(len(data)) # 
    taus = np.logspace(0,tau_max)/fs
    (taus_used, adev, adeverror, adev_n) = allantools.adev(data_rel, data_type='freq', rate=fs, taus=taus)
    time = np.linspace(1,len(data),len(data))/fs
    
    plt.figure(figsize = (8,5))
    # plot the Allan devation
    plt.subplot(111, xscale = 'linear', yscale = 'linear')
    plt.plot(time/3600, (data-np.mean(data))*1e-3, ls = '-',c = 'C0') # , marker = 'none'
#    plt.ylim(-1e2,1e2)
    plt.xlabel('Time / h')
    plt.ylabel('Frequency / kHz')
    plt.grid(b='on', which = 'major', axis = 'both')
    plt.box(on='on')
    plt.title('mean beat frequency is %.3f MHz' % np.mean(data*1e-6))
    plt.show()    
    
    plt.figure(figsize = (8,5))
    # plot the Allan devation
    plt.subplot(111, xscale = 'log', yscale = 'log')
    plt.errorbar(taus_used, adev, yerr=adeverror)
    plt.xlabel('Averaging time t / s')
    plt.ylabel('Allan deviation $\sigma_y(t)$')
    # plt.legend(fontsize = 'xx-small') # too many legend entries...
    plt.grid(b='on', which = 'minor', axis = 'both')
    plt.box(on='on')
    # plt.savefig(DATA_FOLDER + DATIME +  'testing-Allan.png',
    #            #This is simple recomendation for publication plots
    #            dpi=300,
    #            # Plot will be occupy a maximum of available space
    #            bbox_inches='tight',
    #            )
    plt.show()

def noisespec(data,fs):
    f, Pxx_spec = signal.welch(data, fs, ('kaiser', 100), nperseg=1024, scaling='density')
    plt.figure(figsize = (8,5))
    plt.loglog(f, np.sqrt(Pxx_spec))
    plt.xlabel('Frequency / Hz')
    plt.ylabel('Frequency noise ASD  / $Hz/\sqrt{Hz}$')
    plt.grid(True, which = 'both', ls = '-')
    plt.savefig(DATA_FOLDER + DATIME +  'testing-freqnoise.png',
#            ##    This is simple recomendation for publication plots
                dpi=300,
                # Plot will be occupy a maximum of available space
                bbox_inches='tight',
                )
    plt.show()
    
    
def noisespecappend(data,fs):
    f, Pxx_spec = signal.welch(data, fs, ('kaiser', 100), nperseg=1024, scaling='density')
#    plt.figure(figsize = (8,5))
    plt.loglog(f, np.sqrt(Pxx_spec))
    plt.xlabel('Frequency / Hz')
    plt.ylabel('Frequency noise ASD  / $Hz/\sqrt{Hz}$')
    plt.grid(True, which = 'both', ls = '-')
#    plt.savefig(  'testing-freqnoise_2.png',
#                ####This is simple recomendation for publication plots
#                dpi=300,
#                #### Plot will be occupy a maximum of available space
#                bbox_inches='tight',
#                )
    #plt.show()
    
def noisespecssumup(filename):
    with open(filename) as data_file:
        data = json.load(data_file)
        freq = (data["frequencies"])
        fs = (data["measurement_rate"])
        noisespecappend(freq,fs)

# %%

# os.chdir(r'H:\git_MAIUS-B\Referenzlaser')

# for f in ['2018-07-09_16-59-testting_Klotz-GAINref_free_1s.json', '2018-07-09_17-00-testting_Klotz-GAINref_free_5s.json', '2018-07-09_17-01-testting_Klotz-GAINref_free_50s.json',
#           '2018-07-09_16-52-testting_Klotz-GAINref_locked_1s.json', '2018-07-09_16-54-testting_Klotz-GAINref_locked_5s.json', '2018-07-09_16-55-testting_Klotz-GAINref_locked_50s.json']:
#     noisespecssumup(f)
    
    
# %%


# os.chdir(r'C:/Data/Klaus/MAIUS/reflaser/K/MAIUS_vs_Q2/')
# for f in ['2018-09-26_16-05-41_MAIUS_vs_Q2_free_1s.json',
#           '2018-09-26_16-06-57_MAIUS_vs_Q2_free_10s.json',
#           '2018-09-26_16-08-21_MAIUS_vs_Q2_free_100s.json',
# #          '2018-09-26_15-57-26_MAIUS_vs_Q2_locked_1s.json',
# #          '2018-09-26_15-58-57_MAIUS_vs_Q2_locked_10s.json',
# #          '2018-09-26_16-00-40_MAIUS_vs_Q2_locked_100s.json',
# #          '2018-09-26_16-17-53_MAIUS_vs_Q2_locked_1s.json',
# #          '2018-09-26_16-19-27_MAIUS_vs_Q2_locked_10s.json',
# #          '2018-09-26_16-26-51_MAIUS_vs_Q2_locked_1s.json',
# #          '2018-09-26_16-28-27_MAIUS_vs_Q2_locked_10s.json',
# #          '2018-09-26_16-30-01_MAIUS_vs_Q2_locked_100s.json',
# #          '2018-09-27_10-15-06_MAIUS_vs_Q2_locked_10s.json',
# #          '2018-09-27_10-11-26_MAIUS_vs_Q2_locked_100s.json',
# #          '2018-09-27_10-07-50_MAIUS_vs_Q2_locked_1s.json',
#           '2018-09-27_14-43-32_MAIUS_vs_Q2_locked_10s.json',
#           '2018-09-27_14-41-33_MAIUS_vs_Q2_locked_1s.json',
#           '2018-09-27_14-45-16_MAIUS_vs_Q2_locked_100s.json'
#           ]:
#     noisespecssumup(f)    
# #%%
# os.chdir(r'C:\Users\quantus\Desktop\frequencymeasurement\gupta')

# for f in [
#           'Testing_rerun_1_free_1s.0.json',
#           'Testing_rerun_1_free_10s.0.json',
#           'Testing_rerun_1_free_100s.0.json',
#           'Testing_rerun_1_free_1s.1.json',
#           'Testing_rerun_1_free_10s.1.json',
#           'Testing_rerun_1_free_100s.1.json'
# ##          'ML120_locked_1s.6.json',
# #          'ML120_locked_1s.7.json',
# #          'ML120_locked_1s.8.json',
# #          'ML120_locked_1s.9.json',
# #          'ML120_locked_1s.10.json',
# #          'ML120_locked_1s.11.json',
# #          'ML120_locked_1s.12.json'
# #          '25jan_2_free_0s.0.json',
# #          '25jan_2_free_1s.0.json',
# #          '25jan_2_free_10s.0.json',
# #          '25jan_2_free_100s.0',
#           ]:
#     noisespecssumup(f)  
#    allandata()




# %% read the data

# data = np.genfromtxt("2018-07-06_Teststack_Klotz-GAIN_04.dat", dtype=float, delimiter='\t', comments='%', names=['Frequency'])
# data = data['Frequency']
# data = data[1:290000]
# allan(data,10)
# noisespec(data,10)

# data = np.genfromtxt("2018-07-09_Teststack_Klotz-GAIN_01.dat", dtype=float, delimiter='\t', comments='%', names=['Frequency'])
# data = data['Frequency']
# data = data[1:50000]
# allan(data,10)
# noisespec(data,10)

# # %%
# data = np.genfromtxt("2018-07-09_Teststack_Klotz-GAIN_02.dat", dtype=float, delimiter='\t', comments='%', names=['Frequency'])
# data = data['Frequency']
# data = data[900000:1800000]
# allan(data,10)
# #noisespec(data,10)

# # %%
# data = np.genfromtxt("2018-08-08_MAIUS_RefRb-Klotzlaser_100ms_LV_01.dat", dtype=float, delimiter='\t', comments='%', names=['Frequency'])
# data = data['Frequency']
# data = data[100:8*3600*10]
# allan(data,10)
# noisespec(data,10)

# # %%
# data = np.genfromtxt("2018-08-09_MAIUS_RefRb-GAINref_100ms_LV_01.dat", dtype=float, delimiter='\t', comments='%', names=['Frequency'])
# data = data['Frequency']
# #data = data[100:8*3600*10]
# allan(data,10)
# noisespec(data,10)

# # %%
# data = np.genfromtxt("2018-08-09_MAIUS_RefRb-GAINref_wo-iso_100ms_LV_01.dat", dtype=float, delimiter='\t', comments='%', names=['Frequency'])
# data = data['Frequency']
# data = data[6500:8000]
# allan(data,10)
# noisespec(data,10)

# # %% 
# data = np.genfromtxt("2018-08-09_MAIUS_RefRb-GAINref_wo-iso_9010_100ms_LV_01.dat", dtype=float, delimiter='\t', comments='%', names=['Frequency'])
# data = data['Frequency']
# #data = data[6500:8000]
# allan(data,10)
# noisespec(data,10)

# # %% 
# data = np.genfromtxt("2018-08-09_MAIUS_RefRb-GAINref_wo-iso_9010_12MHz_100ms_LV_01.dat", dtype=float, delimiter='\t', comments='%', names=['Frequency'])
# data = data['Frequency']
# #data = data[6500:8000]
# allan(data,10)
# noisespec(data,10)

# # %% Potassium

# data = np.genfromtxt("2018-09-19_K_reflaser_beat_01.txt", dtype=float, delimiter='\t', comments='%', names=['Frequency'])
# data = data['Frequency']
# #data = data[6500:8000]
# allan(data,10)
# noisespec(data,10)

# # %%


# #os.chdir(r'H:\git_MAIUS-B\Referenzlaser\K\MAIUS_vs_Q2')
# os.chdir(r'C:/Data/Klaus/MAIUS/reflaser/K/MAIUS_vs_Q2')
# data = np.genfromtxt("2018-09-26_MAIUS_vs_Q2_locked_10Hz_02.dat", dtype=float, delimiter='\t', comments='%', names=['Frequency'])
# data = data['Frequency']
# data = data[int(2.8*36000):int(4.3*36000)]
# #data = data[104400:151200]
# allan(data,10)
# noisespec(data,10)

# data = np.genfromtxt(r'2018-09-27_MAIUS_vs_Q2Kapsel_locked_10Hz_01.dat', dtype=float, delimiter='\t', comments='%', names=['Frequency'])
# data = data['Frequency']
# data = data[int(1):int(0.8*36000)]
# allan(data,10)
# noisespec(data,10)


# data = np.genfromtxt(r'2018-09-27_MAIUS_vs_Q2Kapsel_locked_1Hz_02 - Kopie.dat', dtype=float, delimiter='\t', comments='%', names=['','','','Frequency','','','','','','','',''])
# data = data['Frequency']
# data = data[int(0.1*3600):int(1.1*3600)]
# allan(data,1)
# noisespec(data,1)

# data = np.genfromtxt(r'2018-09-27_MAIUS_vs_Q2Kapsel_locked_10Hz_03.dat', dtype=float, delimiter='\t', comments='%', names=['Frequency'])
# data = data['Frequency']
# data = data[int(0.1*36000):int(0.841*36000)]
# allan(data,10)
# noisespec(data,10)

# data = np.genfromtxt(r'2018-09-27_MAIUS_vs_Q2Kapsel_locked_1Hz_04.dat', dtype=float, delimiter='\t', comments='%', names=['','','','Frequency','','','','','','','',''])
# data = data['Frequency']
# data = data[int(0.05*3600):int(0.87*3600)]
# allan(data,1)
# noisespec(data,1)

# data = np.genfromtxt(r'2018-09-27_MAIUS_vs_Q2Kapsel_locked_1Hz_06.dat', dtype=float, delimiter='\t', comments='%', names=['','','','Frequency','','','','','','','',''])
# data = data['Frequency']
# data = data[int(0.05*3600):int(11*3600)]
# allan(data,1)
# noisespec(data,1)


# # %% collecto all timeseries with pandas ...?


# def beatpdtr(filename, starttime, timestep, offset = 0):
#     if '_10Hz_' in filename:
#         uc = 0
#     elif '_1Hz_' in filename:
#         uc = 3       
#     beatdat = pd.read_csv(filename, sep = '\t', delimiter = '\t', header = 2, usecols = [uc], names = ['beat'])
#     #beatdat.head(10)
#     lng = len(beatdat)
#     date_rng = pd.date_range(start = starttime, periods = lng, freq = timestep)
#     beatdat.index = date_rng
#     #beatdat.head(10)
#     off = pd.Series(offset*np.ones(len(date_rng)), index=date_rng, name = 'off')
#     beatdat = beatdat.join(off)
#     beatdat['beat'] = beatdat.sum(axis = 1)
#     beatdat = beatdat.drop(columns = 'off')
# #    print(beatdatshift.head(10))
# #    plt.figure()
#     plt.plot(beatdat, '.', alpha = 0.5, label = filename)
#     plt.xlabel('Date', fontsize = 14)
#     plt.ylabel('Frequency / Hz', fontsize=14)
#     plt.ylim(2.07e8,2.10e8)
#     plt.grid(b='on', which = 'major', axis = 'both')
#     plt.box(on='on')
#     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 10})
# #    plt.show()   
    
    
# plt.figure(figsize = (12,6))    
# beatpdtr('2018-09-26_MAIUS_vs_Q2_locked_10Hz_01.dat', starttime = '2018-09-26 17:09:14', timestep = '190ms', offset = 90e6)
# beatpdtr('2018-09-26_MAIUS_vs_Q2_locked_10Hz_02.dat', starttime = '2018-09-26 17:58:24', timestep = '190ms', offset = 90e6)
# beatpdtr('2018-09-27_MAIUS_vs_Q2Kapsel_locked_10Hz_01.dat', starttime = '2018-09-27 10:32:34' ,timestep = '190ms')
# beatpdtr('2018-09-27_MAIUS_vs_Q2Kapsel_locked_10Hz_02.dat', starttime = '2018-09-27 14:52:01' ,timestep = '150ms')
# beatpdtr('2018-09-27_MAIUS_vs_Q2Kapsel_locked_10Hz_03.dat', starttime = '2018-09-27 14:53:19' ,timestep = '200ms')
# beatpdtr('2018-09-27_MAIUS_vs_Q2Kapsel_locked_10Hz_04_temperatureKcell.dat', starttime = '2018-09-27 15:48:38' ,timestep = '100ms')
# beatpdtr('2018-09-27_MAIUS_vs_Q2Kapsel_locked_10Hz_05.dat', starttime = '2018-09-27 16:15:52' ,timestep = '250ms')
# beatpdtr('2018-09-27_MAIUS_vs_Q2Kapsel_locked_10Hz_06.dat', starttime = '2018-09-27 17:26:21' ,timestep = '195ms')
# beatpdtr('2018-09-28_MAIUS_vs_Q2Kapsel_locked_10Hz_01.dat', starttime = '2018-09-28 08:41:15' ,timestep = '250ms')
# ##
# beatpdtr('2018-09-27_MAIUS_vs_Q2Kapsel_locked_1Hz_01.dat', starttime = '2018-09-27 12:20:09', timestep = '1100ms')
# beatpdtr('2018-09-27_MAIUS_vs_Q2Kapsel_locked_1Hz_02.dat', starttime = '2018-09-27 12:22:59', timestep = '1100ms')
# beatpdtr('2018-09-27_MAIUS_vs_Q2Kapsel_locked_1Hz_03.dat', starttime = '2018-09-27 13:48:37', timestep = '1100ms')
# beatpdtr('2018-09-27_MAIUS_vs_Q2Kapsel_locked_1Hz_04.dat', starttime = '2018-09-27 14:52:23', timestep = '1100ms')
# beatpdtr('2018-09-27_MAIUS_vs_Q2Kapsel_locked_1Hz_05.dat', starttime = '2018-09-27 16:16:13', timestep = '1100ms')
# beatpdtr('2018-09-27_MAIUS_vs_Q2Kapsel_locked_1Hz_06.dat', starttime = '2018-09-27 17:26:02', timestep = '1100ms')

# beatpdtr('2018-09-28_MAIUS_vs_Q2Kapsel_locked_1Hz_01.dat', starttime = '2018-09-28 08:41:30', timestep = '1100ms')

# plt.show()  
   

# %% figure out sampling rate

# tstart = datetime(2018, 9, 27, 17, 26, 2)
# duration = (tstart-datetime(2018, 9, 28, 8, 20, 13)).total_seconds()

# print(-duration/48775)


