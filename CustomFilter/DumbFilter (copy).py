import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
import time
import argparse


# Step 1: Function to ask for implementation choice
def choose_implementation():
    root = tk.Tk()
    root.withdraw()
    choice = messagebox.askquestion("Choose Implementation", "Do you want a GUI implementation?")
    return choice == 'yes'


# Step 2: Define Function to Parse Command Line Arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Adaptive filter using Parks-McClellan algorithm")
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


# Step 3: Define Function for Initial Filter Design Using Iterative Optimization
def design_initial_filter(sampling_freq, numtaps, bands, desired, weights):
    start_time = time.time()

    # Convert normalized bands to actual frequencies
    bands_hz = [b * sampling_freq / 2 for b in bands]

    # Initialize the filter coefficients with small random values
    filter_coeffs = np.random.uniform(-0.1, 0.1, numtaps)

    # Iterative optimization (basic version)
    for _ in range(100):  # Increase the number of iterations for better convergence
        error = np.zeros(numtaps)
        h = np.zeros(numtaps)

        for band, gain, weight in zip(bands_hz, desired, weights):
            # Calculate frequency indices
            band_indices = np.linspace(band - 0.5 * sampling_freq, band + 0.5 * sampling_freq, numtaps, endpoint=False)
            band_indices = np.clip(band_indices, 0, sampling_freq - 1).astype(int)

            # Compute ideal frequency response
            ideal_response = gain * np.exp(1j * 2 * np.pi * band_indices / sampling_freq)

            # Compute current filter response
            response = np.zeros(numtaps, dtype=np.complex128)
            for i in range(numtaps):
                response[i] = np.dot(filter_coeffs, ideal_response)

            # Update error and h
            error += weight * (ideal_response - response).real
            h += weight * (np.exp(1j * 2 * np.pi * band_indices / sampling_freq)).real

        # Update filter coefficients
        filter_coeffs += 0.01 * error / (h + 1e-6)  # Adjust the step size as needed and avoid division by zero

    elapsed_time = time.time() - start_time
    print(f"Time taken to calculate filter coefficients: {elapsed_time:.4f} seconds")

    return filter_coeffs / np.sum(filter_coeffs)  # Normalize the coefficients


# Step 4: Define Function for Adaptive Filtering (LMS Algorithm)
def adaptive_filter_lms(initial_filter, input_signal, desired_signal, mu, num_iterations):
    filter_coeffs = initial_filter.astype(float).copy()  # Ensure coefficients are in float
    numtaps = len(initial_filter)
    output_signal = np.zeros(num_iterations)
    error_signal = np.zeros(num_iterations)

    for n in range(numtaps, num_iterations):
        # Extract the current segment of the input signal
        x = input_signal[n - numtaps:n]

        # Normalize the input segment
        x_norm = x / (np.dot(x, x) + 1e-6)  # Avoid division by zero

        # Calculate the output of the filter
        y = np.dot(filter_coeffs, x)

        # Calculate the error signal
        e = desired_signal[n] - y

        # Update the filter coefficients with dynamic step size
        filter_coeffs += 2 * mu * e * x_norm

        # Store the output and error signals
        output_signal[n] = y
        error_signal[n] = e

    return output_signal, error_signal, filter_coeffs


# Step 5: Define Function to Generate Signals
def generate_signals(num_iterations, input_freq, noise_level, sampling_freq):
    t = np.arange(num_iterations) / sampling_freq  # Use proper time vector
    desired_signal = np.sin(2 * np.pi * input_freq * t)
    noise = noise_level * np.random.randn(num_iterations)
    input_signal = desired_signal + noise
    return input_signal, desired_signal


# Step 6: GUI Implementation
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

    # Create and place widgets with default values
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
    numtaps_entry.insert(0, "51")
    bands_entry = tk.Entry(root)
    bands_entry.insert(0, "0 0.1 0.2 0.5")
    desired_entry = tk.Entry(root)
    desired_entry.insert(0, "1 0")
    weights_entry = tk.Entry(root)
    weights_entry.insert(0, "1 10")
    mu_entry = tk.Entry(root)
    mu_entry.insert(0, "0.01")
    num_iterations_entry = tk.Entry(root)
    num_iterations_entry.insert(0, "1000")
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


# Step 7: Function to Visualize Parks-McClellan Algorithm
def visualize_parks_mcclellan(initial_filter, sampling_freq):
    num_points = 8000  # Number of points for frequency response
    freq_response = np.zeros(num_points, dtype=np.complex128)
    w = np.linspace(0, np.pi, num_points)

    for i in range(num_points):
        freq_response[i] = np.sum(initial_filter * np.exp(-1j * w[i] * np.arange(len(initial_filter))))

    # Convert to Hz
    freqs = w * sampling_freq / (2 * np.pi)

    plt.figure()
    plt.plot(freqs, 20 * np.log10(np.abs(freq_response)))
    plt.title('Parks-McClellan Filter Frequency Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid()
    plt.show()


# Step 8: Digitize filter coefficients based on a dynamic threshold (Otsu's method)
def digitize_coefficients(coefficients):
    # Calculate histogram and probabilities
    hist, bin_edges = np.histogram(coefficients, bins=256, range=(np.min(coefficients), np.max(coefficients)))
    total = np.sum(hist)
    sumB = 0
    wB = 0
    maximum = 0.0
    sum1 = np.dot(np.arange(256), hist)

    for i in range(256):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += i * hist[i]
        mB = sumB / wB
        mF = (sum1 - sumB) / wF
        between = wB * wF * (mB - mF) ** 2
        if between >= maximum:
            threshold = i
            maximum = between

    threshold = bin_edges[threshold]

    # Digitize coefficients
    digitized_coefficients = (coefficients >= threshold).astype(float)  # Use float instead of int
    digitized_coefficients = digitized_coefficients * np.sign(coefficients).astype(float)  # Maintain the original sign
    return digitized_coefficients


# Step 9: Function to Run Filter
def run_filter(sampling_freq, numtaps, bands, desired, weights, mu, num_iterations, input_freq, noise_level):
    # Generate Input and Desired Signals
    input_signal, desired_signal = generate_signals(num_iterations, input_freq, noise_level, sampling_freq)

    # Normalize input signal
    input_signal = input_signal / np.max(np.abs(input_signal))

    # Design Initial Filter
    initial_filter = design_initial_filter(sampling_freq, numtaps, bands, desired, weights)
    print("Initial Filter Coefficients:", initial_filter)

    # Visualize the Parks-McClellan Algorithm
    visualize_parks_mcclellan(initial_filter, sampling_freq)

    # Digitize Initial Filter Coefficients
    digitized_initial_filter = digitize_coefficients(initial_filter)
    print("Digitized Initial Filter Coefficients:", digitized_initial_filter)

    # Perform Adaptive Filtering
    output_signal, error_signal, final_coeffs = adaptive_filter_lms(digitized_initial_filter, input_signal,
                                                                    desired_signal, mu, num_iterations)

    print("Final Adaptive Filter Coefficients:", final_coeffs)

    # Plot Results
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(input_signal, label='Input Signal')
    plt.title('Input Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(desired_signal, label='Desired Signal')
    plt.title('Desired Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(output_signal, label='Output Signal')
    plt.plot(desired_signal, label='Desired Signal', linestyle='--')
    plt.plot(input_signal, label='Input Signal', linestyle=':')
    plt.title('Output, Desired, and Input Signals')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


# Main Function
def main():
    if choose_implementation():
        gui_implementation()
    else:
        args = parse_arguments()
        run_filter(args.sampling_freq, args.numtaps, args.bands, args.desired, args.weights, args.mu,
                   args.num_iterations, args.input_freq,
                   args.noise_level)


if __name__ == "__main__":
    main()