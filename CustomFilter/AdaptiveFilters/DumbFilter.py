import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
import argparse

# Function to ask for implementation choice
def choose_implementation():
    root = tk.Tk()
    root.withdraw()
    choice = messagebox.askquestion("Choose Implementation", "Do you want a GUI implementation?")
    return choice == 'yes'

# Function to Parse Command Line Arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Adaptive filter using Parks-McClellan and LMS algorithm")
    parser.add_argument('--sampling_freq', type=int, required=True, help='Sampling frequency')
    parser.add_argument('--numtaps', type=int, required=True, help='Number of filter taps (filter length)')
    parser.add_argument('--bands', type=float, nargs='+', required=True, help='Frequency bands')
    parser.add_argument('--desired', type=float, nargs='+', required=True, help='Desired gains for each band')
    parser.add_argument('--weights', type=float, nargs='+', required=True, help='Weights for each band')
    parser.add_argument('--mu', type=float, required=True, help='Step size for LMS algorithm')
    parser.add_argument('--num_iterations', type=int, required=True, help='Number of iterations for adaptation')
    parser.add_argument('--input_freq', type=float, required=True, help='Frequency of input signal')
    parser.add_argument('--noise_level', type=float, required=True, help='Noise level in the input signal')
    return parser.parse_args()

# Utility function to perform Lagrange interpolation
def lagrange_interpolation(x, y, xi):
    n = len(x)
    yi = 0.0
    for i in range(n):
        term = y[i]
        for j in range(n):
            if j != i:
                denom = x[i] - x[j]
                if denom == 0:
                    continue  # Avoid division by zero
                term *= (xi - x[j]) / denom
        yi += term
    return yi

# Parks-McClellan Algorithm for Initial Filter Design
def design_initial_filter(numtaps, bands, desired):
    bands = [(bands[i], bands[i+1]) for i in range(0, len(bands), 2)]
    desired = [desired[i] for i in range(len(desired))]
    
    def initialize_extremal_frequencies(num_extremals):
        return np.linspace(0, np.pi, num_extremals)
    
    def compute_error_function(frequencies, filter_coeffs, desired, bands):
        E = np.zeros_like(frequencies)
        for i, w in enumerate(frequencies):
            H = np.sum(filter_coeffs * np.cos(np.arange(numtaps) * w))
            for j, band in enumerate(bands):
                if band[0] <= w <= band[1]:
                    E[i] = desired[j] - H
                    break
        return E
    
    def find_local_maxima(E):
        maxima = (np.diff(np.sign(np.diff(E))) < 0).nonzero()[0] + 1
        return maxima
    
    num_extremals = numtaps + 1
    extremals = initialize_extremal_frequencies(num_extremals)
    iteration = 0
    filter_coeffs = np.random.uniform(-0.1, 0.1, numtaps)
    
    while iteration < 100:
        E = compute_error_function(extremals, filter_coeffs, desired, bands)
        maxima = find_local_maxima(E)
        
        if len(maxima) < num_extremals:
            maxima = np.concatenate((maxima, [len(E) - 1]))
        
        if len(maxima) == 0:
            print("No local maxima found, breaking the loop.")
            break
        
        extremals = np.interp(np.linspace(0, len(maxima) - 1, num_extremals), np.arange(len(maxima)), maxima)
        iteration += 1

    for n in range(numtaps):
        filter_coeffs[n] = lagrange_interpolation(extremals, E, 2 * np.pi * n / numtaps)
    
    # Digitize coefficients to 0 or 1
    threshold = np.median(filter_coeffs)
    digitized_coeffs = (filter_coeffs >= threshold).astype(float)
    digitized_coeffs = digitized_coeffs * np.sign(filter_coeffs).astype(float)
    
    return digitized_coeffs

# Define Function for Adaptive Filtering (LMS Algorithm)
def adaptive_filter_lms(filter_coeffs, input_signal, desired_signal, mu, num_iterations):
    numtaps = len(filter_coeffs)
    output_signal = np.zeros(num_iterations)
    error_signal = np.zeros(num_iterations)
    
    for n in range(numtaps, num_iterations):
        x = input_signal[n - numtaps:n]
        y = np.dot(filter_coeffs, x)
        e = desired_signal[n] - y
        filter_coeffs += mu * e * x
        output_signal[n] = y
        error_signal[n] = e

    # Digitize final filter coefficients to 0 or 1
    threshold = np.median(filter_coeffs)
    digitized_final_coeffs = (filter_coeffs >= threshold).astype(float)
    digitized_final_coeffs = digitized_final_coeffs * np.sign(filter_coeffs).astype(float)

    return output_signal, error_signal, digitized_final_coeffs

# Define Function to Generate Signals
def generate_signals(num_iterations, input_freq, noise_level, sampling_freq):
    t = np.arange(num_iterations) / sampling_freq
    desired_signal = np.sin(2 * np.pi * input_freq * t)
    noise = noise_level * np.random.randn(num_iterations)
    input_signal = desired_signal + noise
    return input_signal, desired_signal

# GUI Implementation
def gui_implementation():
    root = tk.Tk()
    root.title("Adaptive Filter Parameters")

    def get_params():
        samp_freq = int(samp_freq_entry.get())
        numtaps = int(numtaps_entry.get())
        bands = list(map(float, bands_entry.get().split()))
        desired = list(map(float, desired_entry.get().split()))
        weights = list(map(float, weights_entry.get().split()))
        mu = float(mu_entry.get())
        num_iterations = int(num_iterations_entry.get())
        input_freq = float(input_freq_entry.get())
        noise_level = float(noise_level_entry.get())

        root.destroy()
        run_filter(samp_freq, numtaps, bands, desired, weights, mu, num_iterations, input_freq, noise_level)

    tk.Label(root, text="Sampling Frequency:").grid(row=0)
    tk.Label(root, text="Number of Filter Taps:").grid(row=1)
    tk.Label(root, text="Frequency Bands:").grid(row=2)
    tk.Label(root, text="Desired Gains:").grid(row=3)
    tk.Label(root, text="Weights:").grid(row=4)
    tk.Label(root, text="Step Size (mu):").grid(row=5)
    tk.Label(root, text="Number of Iterations:").grid(row=6)
    tk.Label(root, text="Input Frequency:").grid(row=7)
    tk.Label(root, text="Noise Level:").grid(row=8)

    samp_freq_entry = tk.Entry(root)
    samp_freq_entry.insert(0, "1024")
    numtaps_entry = tk.Entry(root)
    numtaps_entry.insert(0, "101")
    bands_entry = tk.Entry(root)
    bands_entry.insert(0, "0 0.1 0.2 0.5")
    desired_entry = tk.Entry(root)
    desired_entry.insert(0, "1 0")
    weights_entry = tk.Entry(root)
    weights_entry.insert(0, "1 10")
    mu_entry = tk.Entry(root)
    mu_entry.insert(0, "0.005")
    num_iterations_entry = tk.Entry(root)
    num_iterations_entry.insert(0, "2000")
    input_freq_entry = tk.Entry(root)
    input_freq_entry.insert(0, "5")
    noise_level_entry = tk.Entry(root)
    noise_level_entry.insert(0, "0.1")

    samp_freq_entry.grid(row=0, column=1)
    numtaps_entry.grid(row=1, column=1)
    bands_entry.grid(row=2, column=1)
    desired_entry.grid(row=3, column=1)
    weights_entry.grid(row=4, column=1)
    mu_entry.grid(row=5, column=1)
    num_iterations_entry.grid(row=6, column=1)
    input_freq_entry.grid(row=7, column=1)
    noise_level_entry.grid(row=8, column=1)

    submit_button = tk.Button(root, text="Submit", command=get_params)
    submit_button.grid(row=9, columnspan=2)

    root.mainloop()

# Function to Visualize Filter Frequency Response
def visualize_filter(filter_coeffs, sampling_freq):
    num_points = 8000
    freq_response = np.zeros(num_points, dtype=np.complex128)
    w = np.linspace(0, np.pi, num_points)

    for i in range(num_points):
        freq_response[i] = np.sum(filter_coeffs * np.exp(-1j * w[i] * np.arange(len(filter_coeffs))))

    freqs = w * sampling_freq / (2 * np.pi)

    freq_response_magnitude = np.abs(freq_response)
    freq_response_magnitude[freq_response_magnitude == 0] = 1e-10  # Avoid log of zero

    plt.figure()
    plt.plot(freqs, 20 * np.log10(freq_response_magnitude))
    plt.title('Filter Frequency Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid()
    plt.show()

# Function to Run Filter
def run_filter(sampling_freq, numtaps, bands, desired, weights, mu, num_iterations, input_freq, noise_level):
    input_signal, desired_signal = generate_signals(num_iterations, input_freq, noise_level, sampling_freq)
    input_signal = input_signal / np.max(np.abs(input_signal))
    initial_filter = design_initial_filter(numtaps, bands, desired)
    print("Initial Filter Coefficients:", initial_filter)
    visualize_filter(initial_filter, sampling_freq)
    output_signal, error_signal, final_coeffs = adaptive_filter_lms(initial_filter, input_signal, desired_signal, mu, num_iterations)
    print("Final Adaptive Filter Coefficients:", final_coeffs)
    visualize_filter(final_coeffs, sampling_freq)

    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.plot(input_signal, label='Input Signal')
    plt.title('Input Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()

    plt.subplot(4, 1, 2)
    plt.plot(desired_signal, label='Desired Signal')
    plt.title('Desired Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()

    plt.subplot(4, 1, 3)
    plt.plot(output_signal, label='Output Signal')
    plt.title('Output Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()

    plt.subplot(4, 1, 4)
    plt.plot(np.abs(np.fft.fft(output_signal)), label='Output Signal')
    plt.plot(np.abs(np.fft.fft(desired_signal)), label='Desired Signal', linestyle='--')
    plt.plot(np.abs(np.fft.fft(input_signal)), label='Input Signal', linestyle=':')
    plt.title('Output, Desired, and Input Signals')
    plt.xlabel('Frequency Bin')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def main():
    if choose_implementation():
        gui_implementation()
    else:
        args = parse_arguments()
        run_filter(args.sampling_freq, args.numtaps, args.bands, args.desired, args.weights, args.mu, args.num_iterations, args.input_freq, args.noise_level)

if __name__ == "__main__":
    main()



