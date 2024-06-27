import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importing modules from the AdaptiveFilters package
from AdaptiveFilters import nondigitizedadaptive
from AdaptiveFilters import DigitizedAdaptive
from AdaptiveFilters import DumbFilter


# Parameters
numtaps = 51
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


# Function to display coefficients
def display_coefficients(initial_coeffs, final_coeffs, title):
    plt.figure(figsize=(12, 6))
    plt.plot(initial_coeffs, label='Initial Coefficients', linestyle='--')
    plt.plot(final_coeffs, label='Final Coefficients')
    plt.title(f'{title} Filter Coefficients')
    plt.xlabel('Coefficient Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    

# Display and Compare Coefficients
all_initial_coeffs = [initial_filter_nondigitized, initial_filter_digitized, initial_filter_dumb]
all_final_coeffs = [final_coeffs_nondigitized, final_coeffs_digitized, final_coeffs_dumb]
titles = ['Nondigitized Adaptive', 'Digitized Adaptive', 'Dumb Filter']

for initial_coeffs, final_coeffs, title in zip(all_initial_coeffs, all_final_coeffs, titles):
    display_coefficients(initial_coeffs, final_coeffs, title)
    
# Print coefficients for each signal
print("Initial Coefficients:")
print("Nondigitized Adaptive:", initial_filter_nondigitized)
print("Digitized Adaptive:", initial_filter_digitized)
print("Dumb Filter:", initial_filter_dumb)
print()

print("Final Coefficients:")
print("Nondigitized Adaptive:", final_coeffs_nondigitized)
print("Digitized Adaptive:", final_coeffs_digitized)
print("Dumb Filter:", final_coeffs_dumb)


# Plot Signals
plt.figure(figsize=(12, 8))

# Input Signals
plt.subplot(3, 1, 1)
plt.plot(input_signal_nondigitized, label='Input Signal (Nondigitized)')
plt.plot(input_signal_digitized, label='Input Signal (Digitized Real)', linestyle='--')
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
