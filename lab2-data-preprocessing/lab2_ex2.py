import tensorflow as tf
import numpy as np
from subprocess import Popen
from time import time, sleep
from preprocessing import Normalization, Spectrogram


### Fix the CPU frequency to its maximum value (1.5 GHz)
Popen('sudo sh -c "echo performance >'
      '/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',
        shell=True).wait()


x_test = tf.random.normal((16000,))

normalization = Normalization(tf.int16)
feature_processor = Spectrogram(16000, 0.04, 0.02)
latencies = []
for i in range(200):
    start = time()
    x_normalized = normalization.normalize_audio(x_test)
    x_features = feature_processor.get_spectrogram(x_normalized)
    end = time()

    if i >= 100:
        latencies.append(end - start) 
    
    sleep(0.1)

latencies = np.array(latencies) * 1000
median_latency = np.median(latencies)
std_latency = np.std(latencies)

print(f'Feature Extraction Latency: {median_latency:.1f} +/- {std_latency:.1f}ms')
