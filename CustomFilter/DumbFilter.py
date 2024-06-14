import argparse
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog, messagebox
import time
import math
import random

# Step 1: Function to ask for implementation choice
def choose_implementation():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    choice = messagebox.askquestion("Choose Implementation", "Do you want a GUI implementation?")
    return choice == 'yes'

# Step 2: Define Function to Parse Command Line Arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Adaptive filter using simple operations")
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

# Step 3: Define Function to Generate Signals
def generate_signals(num_iterations, input_freq, noise_level, sampling_freq):
    t = list(range(num_iterations))
    desired_signal = [math.sin(2 * math.pi * input_freq * i / sampling_freq) for i in t]
    noise = [noise_level * (random.random() - 0.5) * 2 for _ in range(num_iterations)]
    input_signal = [desired_signal[i] + noise[i] for i in range(num_iterations)]
    return input_signal, desired_signal

# Step 4: Define Function for Initial Filter Design
def design_initial_filter(numtaps):
    initial_filter = [1.0 / numtaps for _ in range(numtaps)]
    return initial_filter

# Step 5: Define Function for Adaptive Filtering (LMS Algorithm)
def adaptive_filter_lms(initial_filter, input_signal, desired_signal, mu, num_iterations):
    filter_coeffs = initial_filter.copy()
    numtaps = len(initial_filter)
    output_signal = [0.0] * num_iterations
    error_signal = [0.0] * num_iterations

    for n in range(numtaps, num_iterations):
        x = input_signal[n - numtaps:n]
        y = sum(filter_coeffs[i] * x[i] for i in range(numtaps))
        e = desired_signal[n] - y
        for i in range(numtaps):
            filter_coeffs[i] += 2 * mu * e * x[i]
        output_signal[n] = y
        error_signal[n] = e

    return output_signal, error_signal, filter_coeffs

# Step 6: GUI Implementation
def gui_implementation():
    root = tk.Tk()
    root.title("Adaptive Filter Parameters")

    def get_params():
        samp_freq = int(samp_freq_entry.get())
        numtaps = int(numtaps_entry.get())
        mu = float(mu_entry.get())
        num_iterations = int(num_iterations_entry.get())
        input_freq = float(input_freq_entry.get())
        noise_level = float(noise_level_entry.get())

        root.destroy()
        run_filter(samp_freq, numtaps, mu, num_iterations, input_freq, noise_level)

    # Create and place widgets with default values
    tk.Label(root, text="Sampling Frequency:").grid(row=0)
    tk.Label(root, text="Number of Filter Taps:").grid(row=1)
    tk.Label(root, text="Step Size (mu):").grid(row=2)
    tk.Label(root, text="Number of Iterations:").grid(row=3)
    tk.Label(root, text="Input Frequency:").grid(row=4)
    tk.Label(root, text="Noise Level:").grid(row=5)

    samp_freq_entry = tk.Entry(root)
    samp_freq_entry.insert(0, "1024")
    numtaps_entry = tk.Entry(root)
    numtaps_entry.insert(0, "51")
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
    mu_entry.grid(row=2, column=1)
    num_iterations_entry.grid(row=3, column=1)
    input_freq_entry.grid(row=4, column=1)
    noise_level_entry.grid(row=5, column=1)

    submit_button = tk.Button(root, text="Submit", command=get_params)
    submit_button.grid(row=6, columnspan=2)

    root.mainloop()

# Step 7: Function to Visualize Filter Response
def visualize_filter_response(filter_coeffs):
    plt.figure()
    plt.plot(filter_coeffs)
    plt.title('Filter Coefficients')
    plt.xlabel('Coefficient Index')
    plt.ylabel('Value')
    plt.grid()
    plt.show()

# Step 8: Function to Run Filter
def run_filter(sampling_freq, numtaps, mu, num_iterations, input_freq, noise_level):
    # Generate Input and Desired Signals
    input_signal, desired_signal = generate_signals(num_iterations, input_freq, noise_level, sampling_freq)

    # Design Initial Filter
    initial_filter = design_initial_filter(numtaps)
    print("Initial Filter Coefficients:", initial_filter)

    # Visualize the Filter Response
    visualize_filter_response(initial_filter)

    # Perform Adaptive Filtering
    output_signal, error_signal, final_coeffs = adaptive_filter_lms(initial_filter, input_signal, desired_signal, mu, num_iterations)

    print("Final Adaptive Filter Coefficients:", final_coeffs)

    # Plot Results
    plt.figure()

    plt.subplot(2, 1, 1)
    plt.plot(input_signal, label='Input Signal')
    plt.plot(desired_signal, label='Desired Signal')
    plt.title('Input and Desired Signals')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(output_signal, label='Output Signal')
    plt.plot(error_signal, label='Error Signal')
    plt.title('Output and Error Signals')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Main Function
def main():
    if choose_implementation():
        gui_implementation()
    else:
        args = parse_arguments()
        run_filter(args.sampling_freq, args.numtaps, args.mu, args.num_iterations, args.input_freq, args.noise_level)

if __name__ == "__main__":
    main()
