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
### `digitizedcomplex` Module
The digitizedcomplex module implements an adaptive filter using the LMS algorithm with complex coefficients and includes additional functions for binary LMS and coefficient decimation.

#### Example Usage
```python
from adaptive_filters import digitizedcomplex

digitizedcomplex.run_filter(
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
#### digitizedcomplex



`choose_implementation()`: Function to ask for implementation choice via a GUI dialog.

`parse_arguments()`: Function to parse command line arguments.

`design_initial_filter(numtaps, bands, desired, weights)`: Function for initial filter design using Parks-McClellan algorithm.

`adaptive_filter_lms_complex(initial_filter, input_signal, desired_signal, mu, num_iterations)`: Function for adaptive filtering using LMS algorithm with complex coefficients.

`generate_signals_complex(num_iterations, input_freq, noise_level, sampling_freq)`: Function to generate input and desired signals with complex noise.

`gui_implementation()`: GUI implementation to input parameters.

`visualize_parks_mcclellan(initial_filter)`: Function to visualize the Parks-McClellan filter frequency response.

`decimate_coefficients(coefficients)`: Function to decimate filter coefficients to binary values.
binary_lms_complex(initial_filter, input_signal, desired_signal, mu, num_iterations): Binary LMS algorithm with complex coefficients.

`run_filter(numtaps, bands, desired, weights, mu, num_iterations, input_freq, noise_level, sampling_freq)`: Function to run the filter process.

### License
This project is licensed under the MIT License.

