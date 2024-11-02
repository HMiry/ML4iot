import tensorflow as tf
import numpy as np
from subprocess import Popen
from time import time, sleep
from preprocessing import Normalization, Spectrogram  # Importing custom classes for preprocessing

### Fix the CPU frequency to its maximum value (1.5 GHz)
# This command sets the CPU to its maximum performance mode to avoid any CPU throttling
# The 'performance' governor forces the CPU to run at its maximum frequency (1.5 GHz)
Popen('sudo sh -c "echo performance >'
      '/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',
        shell=True).wait()

# Generate a random test audio sample of shape (16000,)
# This simulates 1 second of audio sampled at 16 kHz
x_test = tf.random.normal((16000,))

# Create instances of Normalization and Spectrogram classes
# Normalization normalizes audio to the `int16` format
# Spectrogram extracts spectrogram features from the normalized audio
normalization = Normalization(tf.int16)
feature_processor = Spectrogram(16000, 0.04, 0.02)  # Spectrogram with 16000 Hz sample rate, 40ms frame length, 20ms frame step

# Initialize a list to store latencies for processing (normalized audio + spectrogram feature extraction)
latencies = []

# Process the test audio 200 times, recording latencies for each operation
for i in range(200):
    start = time()  # Start time before feature extraction

    # Normalize the audio data
    x_normalized = normalization.normalize_audio(x_test)

    # Generate the spectrogram from the normalized audio data
    x_features = feature_processor.get_spectrogram(x_normalized)

    end = time()  # End time after processing
    
    # Collect latency values after the first 100 iterations (to avoid including warm-up times)
    if i >= 100:
        latencies.append(end - start)  # Add time difference (in seconds) to the latencies list

    # Pause for 0.1 seconds between iterations to simulate real-time operation
    sleep(0.1)

# Convert the recorded latencies to milliseconds (1 second = 1000 milliseconds)
latencies = np.array(latencies) * 1000  # Latencies were recorded in seconds, convert them to milliseconds

# Calculate the median latency from the collected data (gives a robust central value)
median_latency = np.median(latencies)

# Calculate the standard deviation of the latencies (measures the variability in latencies)
std_latency = np.std(latencies)

# Print the median latency and the standard deviation in milliseconds
print(f'Feature Extraction Latency: {median_latency:.1f} +/- {std_latency:.1f}ms')
