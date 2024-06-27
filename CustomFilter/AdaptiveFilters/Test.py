import matplotlib.pyplot as plt
import numpy as np
from AdaptiveFilters import nondigitizedadaptive, digitizedcomplex, DumbFilter

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
input_signal_digitized, desired_signal_digitized = digitizedcomplex.generate_signals_complex(num_iterations, input_freq, noise_level, sampling_freq)
input_signal_dumb, desired_signal_dumb = DumbFilter.generate_signals(num_iterations, input_freq, noise_level, sampling_freq)

# Design Initial Filters
initial_filter_nondigitized = nondigitizedadaptive.design_initial_filter(numtaps, bands, desired, weights)
initial_filter_digitized = digitizedcomplex.design_initial_filter(numtaps, bands, desired, weights)
initial_filter_dumb = DumbFilter.design_initial_filter(numtaps, bands, desired)

# Perform Adaptive Filtering
output_signal_nondigitized, error_signal_nondigitized, final_coeffs_nondigitized = nondigitizedadaptive.adaptive_filter_lms(initial_filter_nondigitized, input_signal_nondigitized, desired_signal_nondigitized, mu, num_iterations)
output_signal_digitized, error_signal_digitized, final_coeffs_digitized = digitizedcomplex.adaptive_filter_lms_complex(initial_filter_digitized, input_signal_digitized, desired_signal_digitized, mu, num_iterations)
output_signal_dumb, error_signal_dumb, final_coeffs_dumb = DumbFilter.adaptive_filter_lms(initial_filter_dumb, input_signal_dumb, desired_signal_dumb, mu, num_iterations)

# Plot Signals
plt.figure(figsize=(12, 8))

# Input Signals
plt.subplot(3, 1, 1)
plt.plot(input_signal_nondigitized, label='Input Signal (Nondigitized)')
plt.plot(input_signal_digitized.real, label='Input Signal (Digitized Real)', linestyle='--')
plt.plot(input_signal_dumb, label='Input Signal (Dumb)', linestyle=':')
plt.title('Input Signals')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()

# Desired Signals
plt.subplot(3, 1, 2)
plt.plot(desired_signal_nondigitized, label='Desired Signal (Nondigitized)')
plt.plot(desired_signal_digitized.real, label='Desired Signal (Digitized Real)', linestyle='--')
plt.plot(desired_signal_dumb, label='Desired Signal (Dumb)', linestyle=':')
plt.title('Desired Signals')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()

# Output Signals
plt.subplot(3, 1, 3)
plt.plot(np.abs(np.fft.fft(output_signal_nondigitized)), label='Output Signal (Nondigitized)')
plt.plot(np.abs(np.fft.fft(output_signal_digitized.real)), label='Output Signal (Digitized Real)', linestyle='--')
plt.plot(np.abs(np.fft.fft(output_signal_dumb)), label='Output Signal (Dumb)', linestyle=':')
plt.title('Output Signals')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()