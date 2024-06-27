# Adaptive Filters

A package for adaptive filters using LMS and Parks-McClellan algorithms. This package includes two modules: `nondigitizedadaptive` and `digitizedcomplex`.

## Installation

You can install the package using pip:

pip install AdaptiveFilters

## Usage

### `nondigitizedadaptive` Module

The `nondigitizedadaptive` module implements an adaptive filter using the LMS algorithm with real coefficients.

#### Example Usage

```python
from adaptive_filters import nondigitizedadaptive

nondigitizedadaptive.run_filter(
    numtaps=101,
    bands=[0, 0.1, 0.2, 0.5],
    desired=[1, 0],
    weights=[1, 10],
    mu=0.001,
    num_iterations=1000,
    input_freq=5,
    noise_level=0.1,
    sampling_freq=1000
)
```
### `DigitizedAdaptive` Module
The `DigitizedAdaptive` module implements an adaptive filter using the LMS algorithm with digitized coefficients.

#### Example Usage
```python
from adaptive_filters import DigitizedAdaptive

DigitizedAdaptive.run_filter(
    numtaps=101,
    bands=[0, 0.1, 0.2, 0.5],
    desired=[1, 0],
    weights=[1, 10],
    mu=0.001,
    num_iterations=1000,
    input_freq=5,
    noise_level=0.1,
    sampling_freq=1000
)
```
### `DumbFilter.py` Module
The `DumbFilter` module implements an adaptive filter using a simplified approach with the LMS algorithm and Parks-McClellan algorithm for initial filter design.

#### Example Usage 
```python
from adaptive_filters import DumbFilter

DumbFilter.run_filter(
    sampling_freq=1024,
    numtaps=101,
    bands=[0, 0.1, 0.2, 0.5],
    desired=[1, 0],
    weights=[1, 10],
    mu=0.005,
    num_iterations=2000,
    input_freq=5,
    noise_level=0.1
)
```

### Functions
#### nondigitizedadaptive

`choose_implementation()`: Function to ask for implementation choice via a GUI dialog.

`parse_arguments()`: Function to parse command line arguments.

`design_initial_filter(numtaps, bands, desired, weights)`: Function for initial filter design using Parks-McClellan algorithm.

`adaptive_filter_lms(initial_filter, input_signal, desired_signal, mu, num_iterations)`: Function for adaptive filtering using LMS algorithm.

`generate_signals(num_iterations, input_freq, noise_level, sampling_freq)`: Function to generate input and desired signals.

`gui_implementation()`: GUI implementation to input parameters.

`visualize_parks_mcclellan(initial_filter)`: Function to visualize the Parks-McClellan filter frequency response.

`run_filter(numtaps, bands, desired, weights, mu, num_iterations, input_freq, noise_level, sampling_freq)`: Function to run the filter process.

###
#### DigitizedAdaptive

`choose_implementation()`: Function to ask for implementation choice via a GUI dialog.

`parse_arguments()`: Function to parse command line arguments.

`design_initial_filter(numtaps, bands, desired, weights)`: Function for initial filter design using Parks-McClellan algorithm.

`adaptive_filter_lms(initial_filter, input_signal, desired_signal, mu, num_iterations)`: Function for adaptive filtering using LMS algorithm with digitized coefficients.

`generate_signals(num_iterations, input_freq, noise_level, sampling_freq)`: Function to generate input and desired signals.

`gui_implementation()`: GUI implementation to input parameters.

`visualize_parks_mcclellan(initial_filter)`: Function to visualize the Parks-McClellan filter frequency response.

`visualize_parks_mcclellan(initial_filter)`: Function to visualize the Parks-McClellan filter frequency response.

`run_filter(numtaps, bands, desired, weights, mu, num_iterations, input_freq, noise_level, sampling_freq)`: Function to run the filter process.

#### DumbFilter

`choose_implementation()`: Function to ask for implementation choice via a GUI dialog.

`parse_arguments()`: Function to parse command line arguments.

`lagrange_interpolation(x, y, xi)`: Utility function to perform Lagrange interpolation.

`design_initial_filter(numtaps, bands, desired)`: Parks-McClellan Algorithm for initial filter design.

`adaptive_filter_lms(filter_coeffs, input_signal, desired_signal, mu, num_iterations)`: Function for adaptive filtering using LMS algorithm.

`generate_signals(num_iterations, input_freq, noise_level, sampling_freq)`: Function to generate input and desired signals.

`gui_implementation()`: GUI implementation to input parameters.

`visualize_filter(filter_coeffs, sampling_freq)`: Function to visualize filter frequency response.

`run_filter(sampling_freq, numtaps, bands, desired, weights, mu, num_iterations, input_freq, noise_level)`: Function to run the filter process.


### Testing
To test the adaptive filters using all three types of signals (input signal, desired signal, and output signal), you can run the following code snippets. These examples demonstrate how to generate and visualize the signals using all three modules.

### Testing with nondigitizedadaptive Module
```python
import matplotlib.pyplot as plt
from adaptive_filters import nondigitizedadaptive, DigitizedAdaptive, DumbFilter

# Parameters
numtaps = 101
bands = [0, 0.1, 0.2, 0.5]
desired = [1, 0]
weights = [1, 10]
mu = 0.001
num_iterations = 1000
input_freq = 5
noise_level = 0.1
sampling_freq = 1000

# Generate Signals
input_signal_nondigitized, desired_signal_nondigitized = nondigitizedadaptive.generate_signals(num_iterations, input_freq, noise_level, sampling_freq)
input_signal_digitized, desired_signal_digitized = DigitizedAdaptive.generate_signals(num_iterations, input_freq, noise_level, sampling_freq)
input_signal_dumb, desired_signal_dumb = DumbFilter.generate_signals(num_iterations, input_freq, noise_level, sampling_freq)

# Design Initial Filters
initial_filter_nondigitized = nondigitizedadaptive.design_initial_filter(numtaps, bands, desired, weights)
initial_filter_digitized = DigitizedAdaptive.design_initial_filter(numtaps, bands, desired, weights)
initial_filter_dumb = DumbFilter.design_initial_filter(numtaps, bands, desired)

# Perform Adaptive Filtering
output_signal_nondigitized, error_signal_nondigitized, final_coeffs_nondigitized = nondigitizedadaptive.adaptive_filter_lms(initial_filter_nondigitized, input_signal_nondigitized, desired_signal_nondigitized, mu, num_iterations)
output_signal_digitized, error_signal_digitized, final_coeffs_digitized = DigitizedAdaptive.adaptive_filter_lms(initial_filter_digitized, input_signal_digitized, desired_signal_digitized, mu, num_iterations)
output_signal_dumb, error_signal_dumb, final_coeffs_dumb = DumbFilter.adaptive_filter_lms(initial_filter_dumb, input_signal_dumb, desired_signal_dumb, mu, num_iterations)

# Plot Signals
plt.figure(figsize=(12, 8))

# Input Signals
plt.subplot(3, 1, 1)
plt.plot(input_signal_nondigitized, label='Input Signal (Nondigitized)')
plt.plot(input_signal_digitized, label='Input Signal (Digitized)', linestyle='--')
plt.plot(input_signal_dumb, label='Input Signal (Dumb)', linestyle=':')
plt.title('Input Signals')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()

# Desired Signals
plt.subplot(3, 1, 2)
plt.plot(desired_signal_nondigitized, label='Desired Signal (Nondigitized)')
plt.plot(desired_signal_digitized, label='Desired Signal (Digitized)', linestyle='--')
plt.plot(desired_signal_dumb, label='Desired Signal (Dumb)', linestyle=':')
plt.title('Desired Signals')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()

# Output Signals
plt.subplot(3, 1, 3)
plt.plot(np.abs(np.fft.fft(output_signal_nondigitized)), label='Output Signal (Nondigitized)')
plt.plot(np.abs(np.fft.fft(output_signal_digitized)), label='Output Signal (Digitized)', linestyle='--')
plt.plot(np.abs(np.fft.fft(output_signal_dumb)), label='Output Signal (Dumb)', linestyle=':')
plt.plot(np.abs(np.fft.fft(desired_signal_nondigitized)), label='Desired Signal')
plt.title('Output Signals')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()
```
### License
This project is licensed under the MIT License.

