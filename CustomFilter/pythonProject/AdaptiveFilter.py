import argparse
import numpy as np
from scipy.signal import remez
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog, messagebox

# Step 1: Function to ask for implementation choice
def choose_implementation():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    choice = messagebox.askquestion("Choose Implementation", "Do you want a GUI implementation?")
    return choice == 'yes'

# Step 2: Define Function to Parse Command Line Arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Adaptive filter using Parks-McClellan algorithm")
    parser.add_argument('--numtaps', type=int, required=True, help='Number of filter taps (filter length)')
    parser.add_argument('--bands', type=float, nargs='+', required=True, help='Frequency bands')
    parser.add_argument('--desired', type=float, nargs='+', required=True, help='Desired gains for each band')
    parser.add_argument('--weights', type=float, nargs='+', required=True, help='Weights for each band')
    parser.add_argument('--mu', type=float, required=True, help='Step size for LMS algorithm')
    parser.add_argument('--num_iterations', type=int, required=True, help='Number of iterations for adaptation')
    parser.add_argument('--input_freq', type=float, required=True, help='Frequency of input signal')
    parser.add_argument('--noise_level', type=float, required=True, help='Noise level in the input signal')
    return parser.parse_args()

# Step 3: Define Function for Initial Filter Design Using Parks-McClellan Algorithm
def design_initial_filter(numtaps, bands, desired, weights):
    return remez(numtaps, bands, desired, weight=weights)

# Step 4: Define Function for Adaptive Filtering (LMS Algorithm)
def adaptive_filter_lms(initial_filter, input_signal, desired_signal, mu, num_iterations):
    filter_coeffs = initial_filter.copy()
    numtaps = len(initial_filter)
    output_signal = np.zeros(num_iterations)
    error_signal = np.zeros(num_iterations)

    for n in range(numtaps, num_iterations):
        x = input_signal[n - numtaps:n]
        y = np.dot(filter_coeffs, x)
        e = desired_signal[n] - y
        filter_coeffs += 2 * mu * e * x
        output_signal[n] = y
        error_signal[n] = e

    return output_signal, error_signal, filter_coeffs

# Step 5: Define Function to Generate Signals
def generate_signals(num_iterations, input_freq, noise_level):
    t = np.arange(num_iterations)
    desired_signal = np.sin(2 * np.pi * input_freq * t / num_iterations)
    noise = noise_level * np.random.randn(num_iterations)
    input_signal = desired_signal + noise
    return input_signal, desired_signal

# Step 6: GUI Implementation
def gui_implementation():
    root = tk.Tk()
    root.title("Adaptive Filter Parameters")

    def get_params():
        numtaps = int(numtaps_entry.get())
        bands = list(map(float, bands_entry.get().split()))
        desired = list(map(float, desired_entry.get().split()))
        weights = list(map(float, weights_entry.get().split()))
        mu = float(mu_entry.get())
        num_iterations = int(num_iterations_entry.get())
        input_freq = float(input_freq_entry.get())
        noise_level = float(noise_level_entry.get())

        root.destroy()
        run_filter(numtaps, bands, desired, weights, mu, num_iterations, input_freq, noise_level)

    # Create and place widgets
    tk.Label(root, text="Number of Filter Taps:").grid(row=0)
    tk.Label(root, text="Frequency Bands:").grid(row=1)
    tk.Label(root, text="Desired Gains:").grid(row=2)
    tk.Label(root, text="Weights:").grid(row=3)
    tk.Label(root, text="Step Size (mu):").grid(row=4)
    tk.Label(root, text="Number of Iterations:").grid(row=5)
    tk.Label(root, text="Input Frequency:").grid(row=6)
    tk.Label(root, text="Noise Level:").grid(row=7)

    numtaps_entry = tk.Entry(root)
    bands_entry = tk.Entry(root)
    desired_entry = tk.Entry(root)
    weights_entry = tk.Entry(root)
    mu_entry = tk.Entry(root)
    num_iterations_entry = tk.Entry(root)
    input_freq_entry = tk.Entry(root)
    noise_level_entry = tk.Entry(root)

    numtaps_entry.grid(row=0, column=1)
    bands_entry.grid(row=1, column=1)
    desired_entry.grid(row=2, column=1)
    weights_entry.grid(row=3, column=1)
    mu_entry.grid(row=4, column=1)
    num_iterations_entry.grid(row=5, column=1)
    input_freq_entry.grid(row=6, column=1)
    noise_level_entry.grid(row=7, column=1)

    submit_button = tk.Button(root, text="Submit", command=get_params)
    submit_button.grid(row=8, columnspan=2)

    root.mainloop()

# Step 7: Function to Run Filter
def run_filter(numtaps, bands, desired, weights, mu, num_iterations, input_freq, noise_level):
    # Generate Input and Desired Signals
    input_signal, desired_signal = generate_signals(num_iterations, input_freq, noise_level)

    # Design Initial Filter
    initial_filter = design_initial_filter(numtaps, bands, desired, weights)
    print("Initial Filter Coefficients:", initial_filter)

    # Perform Adaptive Filtering
    output_signal, error_signal, final_coeffs = adaptive_filter_lms(initial_filter, input_signal, desired_signal, mu, num_iterations)

    print("Final Adaptive Filter Coefficients:", final_coeffs)

    # Plot Results
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(input_signal, label='Input Signal')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(desired_signal, label='Desired Signal')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(output_signal, label='Output Signal')
    plt.plot(error_signal, label='Error Signal')
    plt.legend()
    plt.show()

# Main Function
def main():
    if choose_implementation():
        gui_implementation()
    else:
        args = parse_arguments()
        run_filter(args.numtaps, args.bands, args.desired, args.weights, args.mu, args.num_iterations, args.input_freq, args.noise_level)

if __name__ == "__main__":
    main()
