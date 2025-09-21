import tensorflow as tf
import numpy as np
import zipfile
from subprocess import Popen
from time import time, sleep

# You may only modify the values of `MODEL_FILE_NAME` and `PREPROCESSING_ARGS`.
# Ensure that the PREPROCESSING_ARGS values match those used during training.
# No other modifications are permitted in this script

MODEL_FILE_PATH = './1730886043.tflite' # file extension can be .tflite or .zip

PREPROCESSING_ARGS = {
    'sampling_rate': 16000,
    'frame_length_in_s': 0.04,
    'frame_step_in_s': 0.02,
    'num_mel_bins': 40,
    'lower_frequency': 20,
    'upper_frequency': 4000,
    'num_coefficients': 0  # set num_coefficients to 0 if log-Mel Spectrogram features have been used
}


class Normalization():
    def __init__(self, bit_depth):
        self.max_range = bit_depth.max

    def normalize_audio(self, audio):
        audio_float32 = tf.cast(audio, tf.float32)
        audio_normalized = audio_float32 / self.max_range

        return audio_normalized

    def normalize(self, audio, label):
        audio_normalized = self.normalize_audio(audio)

        return audio_normalized, label

class Spectrogram():
    def __init__(self, sampling_rate, frame_length_in_s, frame_step_in_s):
        self.frame_length = int(frame_length_in_s * sampling_rate)
        self.frame_step = int(frame_step_in_s * sampling_rate)

    def get_spectrogram(self, audio):
        stft = tf.signal.stft(
            audio,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            fft_length=self.frame_length
        )
        spectrogram = tf.abs(stft)

        return spectrogram

    def get_spectrogram_and_label(self, audio, label):
        spectrogram = self.get_spectrogram(audio)

        return spectrogram, label


class MelSpectrogram():
    def __init__(
        self,
        sampling_rate,
        frame_length_in_s,
        frame_step_in_s,
        num_mel_bins,
        lower_frequency,
        upper_frequency
    ):
        self.spectrogram_processor = Spectrogram(sampling_rate, frame_length_in_s, frame_step_in_s)
        num_spectrogram_bins = self.spectrogram_processor.frame_length // 2 + 1

        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=num_mel_bins,
            num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=sampling_rate,
            lower_edge_hertz=lower_frequency,
            upper_edge_hertz=upper_frequency
        )

    def get_mel_spec(self, audio):
        spectrogram = self.spectrogram_processor.get_spectrogram(audio)
        mel_spectrogram = tf.matmul(spectrogram, self.linear_to_mel_weight_matrix)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)

        return log_mel_spectrogram

    def get_mel_spec_and_label(self, audio, label):
        log_mel_spectrogram = self.get_mel_spec(audio)

        return log_mel_spectrogram, label


class MFCC():
    def __init__(
        self,
        sampling_rate,
        frame_length_in_s,
        frame_step_in_s,
        num_mel_bins,
        lower_frequency,
        upper_frequency,
        num_coefficients
    ):
        self.mel_spec_processor = MelSpectrogram(
            sampling_rate, frame_length_in_s, frame_step_in_s, num_mel_bins, lower_frequency, upper_frequency
        )
        self.num_coefficients = num_coefficients

    def get_mfccs(self, audio):
        log_mel_spectrogram = self.mel_spec_processor.get_mel_spec(audio)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.num_coefficients]

        return mfccs

    def get_mfccs_and_label(self, audio, label):
        mfccs = self.get_mfccs(audio)

        return mfccs, label

### Fix the CPU frequency to its maximum value (1.5 GHz)
Popen('sudo sh -c "echo performance >'
      '/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',
        shell=True).wait()


x_test = tf.random.normal((16000,))

normalization = Normalization(tf.int16)

if PREPROCESSING_ARGS['num_coefficients'] == 0:
    PREPROCESSING_ARGS.pop('num_coefficients')
    feature_processor = MelSpectrogram(**PREPROCESSING_ARGS)
    feature_processor_fn = feature_processor.get_mel_spec
else:
    feature_processor = MFCC(**PREPROCESSING_ARGS)
    feature_processor_fn = feature_processor.get_mfccs


if MODEL_FILE_PATH.endswith('.zip'):
    with zipfile.ZipFile(MODEL_FILE_PATH, 'r') as fp:
        fp.extractall('/tmp/')
        model_filename = fp.namelist()[0]
        MODEL_FILE_PATH = '/tmp/' + model_filename

interpreter = tf.lite.Interpreter(model_path=MODEL_FILE_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()

preprocessing_latencies = []
model_latencies = []
tot_latencies = []
for i in range(200):
    start_preprocess = time()
    x_normalized = normalization.normalize_audio(x_test)
    x_features = feature_processor_fn(x_normalized)
    x_features = tf.expand_dims(x_features, 0)
    x_features = tf.expand_dims(x_features, -1)
    end_preprocess = time()

    interpreter.set_tensor(input_details[0]['index'], x_features)
    interpreter.invoke()

    end_inference = time()

    if i >= 100:
        preprocessing_latencies.append(end_preprocess - start_preprocess)
        model_latencies.append(end_inference - end_preprocess)
        tot_latencies.append(end_inference - start_preprocess)

    sleep(0.1)

preprocessing_latencies = 1000 * np.array(preprocessing_latencies)
median_preprocessing_latency = np.median(preprocessing_latencies)
std_preprocessing_latency = np.std(preprocessing_latencies)
model_latencies = 1000 * np.array(model_latencies)
median_model_latency = np.median(model_latencies)
std_model_latency = np.std(model_latencies)
tot_latencies = 1000 * np.array(tot_latencies)
median_total_latency = np.median(tot_latencies)
std_total_latency = np.std(tot_latencies)

print(f'Preprocessing Latency: {median_preprocessing_latency:.1f} +/- {std_preprocessing_latency:.1f}ms')
print(f'Model Latency: {median_model_latency:.1f} +/- {std_model_latency:.1f}ms')
print(f'Total Latency: {median_total_latency:.1f} +/- {std_total_latency:.1f}ms')
