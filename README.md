# Tools for phase and frequency data

## Installation

```
git clone https://git.physik.hu-berlin.de/pylab/freq_tools.git
cd freq_tools
python setup.py install
```

Alternatively, if you plan to make changes to the code, use:

```
python setup.py develop
```

## Features

The main feature are simple conversion between power and amplitude spectral densities as shown in this diagram.

![png](docs/representaions.png)

The other features are shown in the Example section below.

## Importing the module and getting help


```python
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

import freq_tools as ft
```

To get more information about the provided classes and functions use `?`, e.g.


```python
ft.OscillatorNoise?
```

or


```python
ft.io.import_csv?
```

## Examples
### Transfer Functions

### Counter data and frequency noise

In this example we are importing counter measurements of a beatnote of two free running lasers and extracting the corresponding frequency noise. Due to the limited internal buffer we took three separate measurements of durations between 1 and 100 seconds and thus varying sample rates.

First, we import the data into `CounterData` objects.


```python
files = ['counter_example_1s.json', 'counter_example_10s.json', 'counter_example_100s.json']
counter_data_list = [ft.import_json(file, as_class=ft.CounterData, silent=True) for file in files]
```

Each of the CounterData objects in `counter_data_list` is then converted to a `OscillatorNoise` object by using Welch's method and join them all together into one `OscillatorNoise` object:


```python
noise_list = [counter_data.to_oscillator_noise() for counter_data in counter_data_list]
noise = noise_list[0]
noise.join(noise_list[1:2]) #join the 2nd and 3rd counter data to the 1st one
noise.label = 'laser beatnote'

```

We are interestd in the Frequency noise of one laser, so we have to divide the PSD by 2 (assuming uncorrelated noise). Changing the `values` property will automatically cause all representations to be recalculated.


```python
noise.values = noise.values / 2
```

For linewidth calculation, we create a `BetaLine` object. By manually tweaking the coefficient, we find a model of flicker frequency noise using the `PowerLawNoise` class: 


```python
beta_line = ft.BetaLine()
flicker_noise = ft.PowerLawNoise(coeff=1.5e10, exponent=-1, base='freq', representation='psd_freq')
```


```python
fig, ax = noise.plot()
beta_line.plot(noise.freqs, ax=ax)
flicker_noise.plot(noise.freqs, ax=ax)
ax.legend()
```

![png](docs/output_15_2.png)


As can be seen from the plot, the data does not intersect with the $\beta$ separation line. One approach would be to use the flicker frequency noise model to create an `OscillatorNoise` object with `OsscillatorNoise(some_freqs, flicker_noise.values(some_freqs))` with appropriate frequencies `some_freqs` and calculate the linewidth with that (or use the `join` method of `OscillatorNoise` to extend the data with the model).

By default, the data is extrapolated as the last data point, i.e. assuming we reach a white frequency noise floor beyond data. With this, we determine an upper bound for the linewidth, using the $\beta$ separation line:


```python
beta_line.linewidth(noise) / 1e6
```


    0.7365025765587632



This means, an linewidth of roughly 700 kHz at 1 ms (the default value for `linewidth`).

### Spectrum analyzer data

Here, we analyze the data of a phaselock between two MILAs ECDLs.

First, we have a look a beatnote measured with a spectrum analyzer


```python
spectrum = ft.import_csv('laser_spectrum_analyzer_data.txt', as_class=ft.SpectrumAnalyzerData, delimiter='\t',
                                    rbw=3e5, label='Beatnote')
spectrum.plot()
```


![png](docs/output_21_1.png)


We can also convert this data to phase noise (in a `OscillatorNoise` object) and plot the resulting PSD of the phase:


```python
laser_noise_from_spec = spectrum.to_oscillator_noise()
laser_noise_from_spec.plot()
```




    (<Figure size 432x288 with 1 Axes>,
     <matplotlib.axes._subplots.AxesSubplot at 0x29b7c9bca08>)


![png](docs/output_23_1.png)


### Phase noise 

Alternatively, we record the data with a Microsemi 512 A phase noise test setup which gives much more accuarate data for frequencies closer to the carrier, however has a small frequency span of only 1 MHz


```python
laser_noise = ft.import_csv('laser_phase_noise.csv', as_class=ft.OscillatorNoise, 
                            representation='script_L', label='Laser')
laser_noise.plot()
```

![png](docs/output_26_1.png)


We now use the data from the spectrunm analyzer measurement from before and manually adjust the level (only God can judge us!). We only use the data beyond one Mhz.


```python
# only use these data outside of range of Microsemi tool
laser_noise_from_spec = laser_noise_from_spec[laser_noise_from_spec.freqs > 1e6] 
laser_noise_from_spec.values += 14 # manually addition to fit levels of microsemi and spectrum analyzer
laser_noise.join(laser_noise_from_spec)
```


```python
laser_noise.plot()
```
![png](docs/output_29_1.png)


### Calculating impact of laser phase noise on atom interferometer phase noise

In this example we use the laser phase noise data from above to calculate the impact of the laser's phase noise for a Mach Zehnder atom interferometer with an interferometer time 2 T = 400 ms and a pulse time 𝜏 = 20 μs. For this we first have to consider another noise source coming from the reference quartz the experiment is disciplined to. We load the data (accounting for the division factor, noise was measured at 10 MHz, the actual frequency is 6.8 GHz) and plot it together with the phase noise form the laser.


```python
quartz_noise = L_quartz = ft.import_csv('quartz_phase_noise.csv',as_class=ft.OscillatorNoise,
                      representation='script_L', divide_by=680, label='Quartz')
```

```python
fig, ax = laser_noise.plot()
quartz_noise.plot(ax=ax)
ax.legend()
```

![png](docs/output_33_1.png)


Next, we have to consider the transfer function of the atom interferometer. It is a highly oscillatory function with a low-pass behaviour which an be verified by calling its `plot_magnitude` method. Here we scale the transfer function with $(k_\text{eff} T^2)$ with $k_\text{eff} = \frac{4\pi}{780\,\mathrm{nm}}$ to get it in units of m/s²/rad.


```python
mz_tf = ft.MachZehnderTransferFunction(T=200e-3, tau=40e-6, convert_to_g=True)
```

To calculate the atom interferometer noise, we first have to scale the phase noise with the squared magnitude of the transfer function and integrate over all frequencies of interest:

\begin{equation}
\sigma_{\text{AI}}^2 = \int_{f_0}^\infty |H_\text{AI}(2\pi f)|^2 \cdot S_\phi(f) \mathrm{d}f
\end{equation}

We leave the lower integration bound $f_0$ as a free parameter in this equation and inegrate all phasenoies above this frequency, or conversly, up to a certain measurement time. $S_\phi$ here is the 1-sided PSD of phase. To conversion from $L(f)$ to $S_\phi$ is done automatically, as can be seen by looking at the `representation` argument of the scaled phase noise or the plot label. 


```python
laser_noise_scaled = mz_tf.scale_noise(laser_noise)
quartz_noise_scaled = mz_tf.scale_noise(quartz_noise)

fig, ax = laser_noise_scaled.plot()
quartz_noise_scaled.plot(ax=ax)
ax.legend()
```

![png](docs/output_37_1.png)


The last step now is the integration over all relevant frequencies. In this plot integration takes place from the right sight and for each frequency the accumulated phase noise of all higher frequencies is displayed.


```python
laser_noise_accum = laser_noise_scaled.accumulate()
quartz_noise_accum = quartz_noise_scaled.accumulate()

fig, ax = laser_noise_accum.plot()
quartz_noise_accum.plot(ax=ax)
ax.legend()
```

![png](docs/output_39_1.png)

