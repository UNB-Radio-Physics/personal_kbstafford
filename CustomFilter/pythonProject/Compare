import numpy as np
import time

# Generate a sample signal
N = 1024  # Length of the signal
x = np.random.random(N)

# Compute DFT directly
start_time = time.time()
X_dft = np.zeros(N, dtype=complex)
for k in range(N):
    for n in range(N):
        X_dft[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
dft_time = time.time() - start_time

# Compute FFT using numpy
start_time = time.time()
X_fft = np.fft.fft(x)
fft_time = time.time() - start_time

print(f"DFT Time: {dft_time:.5f} seconds")
print(f"FFT Time: {fft_time:.5f} seconds")
