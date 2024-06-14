import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, firwin, butter, freqz

# Generate a test signal: square wave
t = np.linspace(0, 1, 500, endpoint=False)
square_wave = np.sign(np.sin(2 * np.pi * 5 * t))

# Design a linear phase FIR filter (low-pass)
numtaps = 51
cutoff = 0.1  # Normalized frequency (0.1 * Nyquist)
fir_coeff = firwin(numtaps, cutoff)

# Design a nonlinear phase IIR filter (Butterworth low-pass)
order = 3
b, a = butter(order, cutoff)

# Apply the filters
filtered_square_fir = lfilter(fir_coeff, 1, square_wave)
filtered_square_iir = lfilter(b, a, square_wave)

# Plot the original and filtered signals
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, square_wave)
plt.title('Original Square Wave')
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t, filtered_square_fir)
plt.title('Filtered Square Wave (Linear Phase FIR)')
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t, filtered_square_iir)
plt.title('Filtered Square Wave (Nonlinear Phase IIR)')
plt.grid()

plt.tight_layout()
plt.show()

# Plot the phase responses of the filters
w, h_fir = freqz(fir_coeff)
w, h_iir = freqz(b, a)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(w / np.pi, np.angle(h_fir))
plt.title('Phase Response (Linear Phase FIR)')
plt.xlabel('Normalized Frequency')
plt.ylabel('Phase (radians)')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(w / np.pi, np.angle(h_iir))
plt.title('Phase Response (Nonlinear Phase IIR)')
plt.xlabel('Normalized Frequency')
plt.ylabel('Phase (radians)')
plt.grid()

plt.tight_layout()
plt.show()
