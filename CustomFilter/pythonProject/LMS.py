import numpy as np
import matplotlib.pyplot as plt

# LMS Algorithm Parameters
mu = 0.01  # Step size (learning rate)
N = 32     # Number of filter taps
iterations = 1000  # Number of iterations for adaptation

# Generate synthetic input signal (e.g., a sine wave with noise)
np.random.seed(0)  # For reproducibility
n = np.arange(0, 500)
input_signal = np.sin(0.01 * np.pi * n) + np.random.normal(0, 0.5, len(n))

# Desired signal (original sine wave without noise)
desired_signal = np.sin(0.01 * np.pi * n)

# Initialize filter coefficients (weights) to zero
w = np.zeros(N)

# Initialize variables for storing results
output_signal = np.zeros(len(input_signal))
error_signal = np.zeros(len(input_signal))

# LMS Algorithm
for i in range(N, len(input_signal)):
    x = input_signal[i-N:i][::-1]  # Input vector (reversed order)
    y = np.dot(w, x)               # Filter output
    d = desired_signal[i]          # Desired signal
    e = d - y                      # Error signal
    w = w + 2 * mu * e * x         # Update filter coefficients
    output_signal[i] = y
    error_signal[i] = e

# Plotting the results
plt.figure(figsize=(14, 10))

# Plot the input signal
plt.subplot(3, 1, 1)
plt.plot(input_signal, label='Input Signal (Noisy)')
plt.title('Input Signal')
plt.legend()

# Plot the desired signal
plt.subplot(3, 1, 2)
plt.plot(desired_signal, label='Desired Signal (Original)', color='g')
plt.title('Desired Signal')
plt.legend()

# Plot the output signal and error signal
plt.subplot(3, 1, 3)
plt.plot(output_signal, label='Output Signal (Filtered)', color='r')
plt.plot(error_signal, label='Error Signal', color='orange')
plt.title('Output and Error Signals')
plt.legend()

plt.tight_layout()
plt.show()
