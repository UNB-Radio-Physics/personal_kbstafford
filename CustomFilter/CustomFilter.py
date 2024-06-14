import argparse
import numpy as np
from scipy.signal import remez, lfilter
import matplotlib.pyplot as plt


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


# Step 6: Main Function
def main():
    args = parse_arguments()

    # Generate Input and Desired Signals
    input_signal, desired_signal = generate_signals(args.num_iterations, args.input_freq, args.noise_level)

    # Design Initial Filter
    initial_filter = design_initial_filter(args.numtaps, args.bands, args.desired, args.weights)
    print("Initial Filter Coefficients:", initial_filter)

    # Perform Adaptive Filtering
    output_signal, error_signal, final_coeffs = adaptive_filter_lms(initial_filter, input_signal, desired_signal,
                                                                    args.mu, args.num_iterations)

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


if __name__ == "__main__":
    main()

