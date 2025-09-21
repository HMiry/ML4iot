#!/usr/bin/env python
import tensorflow as tf
import numpy as np

import zipfile

import adafruit_dht # type: ignore
from board import D4 # type: ignore

import redis
import argparse

import sounddevice as sd # type: ignore
from scipy import signal # type: ignore
import time
from datetime import datetime

# Initialize states and timing
DATA_COLLECTION_ENABLED = False
LAST_TOGGLE_TIME = time.time()

# Configure audio input with 1 channel, 16-bit depth, and 48 kHz sampling rate
DEVICE_ID = 1  # USB microphone device ID (check this on your system if needed)
CHANNELS = 1
SAMPLE_RATE = 48000  # Sample rate for InputStream

# Constants
DATA_INTERVAL_SECONDS = 2                               # 2 seconds data acquisition interval
MAC_ADDRESS = "e4:5f:01:e8:9b:d0"                       # MAC address of the device

# Define timeseries names using the MAC address
DB_TEMP_NAME = MAC_ADDRESS + ":temperature"
DB_HUMO_NAME = MAC_ADDRESS + ":humidity"


# Function to create a timeseries on redis, only if it doesn't exist, with the default parameters
def create_timeseries_if_not_exists(client, key, retention_msecs = 30 * 24 * 60 * 60 * 1000, chunk_size=128, duplicate_policy="last"):
    try:
        client.ts().create(
            key,
            retention_msecs=retention_msecs,
            chunk_size=chunk_size,
            duplicate_policy=duplicate_policy
        )
    except redis.ResponseError as e:
        if "already exists" in str(e):
            print(f"Timeseries '{key}' already exists, skipping creation.")
        else:
            raise  # Raise any other error

# Function to create all the necessary databases
def create_needed_dbs(redis_client):
    # Create temperature and humidity timeseries with compression enabled
    try:
        # Base time series for temperature and humidity
        create_timeseries_if_not_exists(redis_client, DB_TEMP_NAME)
        create_timeseries_if_not_exists(redis_client, DB_HUMO_NAME)

    except redis.ResponseError as e:
        print(f'Error setting up timeseries or aggregation rules: {e}')
        return(-1)

# parse command line arguments and return the Redis client
def get_redis_client(args):
    redis_client = redis.Redis(
        # check if the host is provided, otherwise use the default host
        host=args.host,
        port=args.port,
        username=args.user,
        password=args.password
    )
    # Check connection to Redis
    try:
        if redis_client.ping():
            print('Redis Connected')
            return redis_client
    except redis.ConnectionError as e:
        print(f"Failed to connect to Redis: {e}")
        return(-1)

# Define the Spectrogram class for the VAD system
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

# Define the MelSpectrogram class
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

# Define the MFCC class
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
    
# Define the Normalization class
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

# Define the VAD class
class VAD:
    def __init__(self, sampling_rate, frame_length_in_s, frame_step_in_s, dBthres, duration_thres):
        self.frame_length_in_s = frame_length_in_s
        self.frame_step_in_s = frame_step_in_s
        self.spec_processor = Spectrogram(sampling_rate, frame_length_in_s, frame_step_in_s)
        self.dBthres = dBthres
        self.duration_thres = duration_thres

    def is_silence(self, audio):
        spectrogram = self.spec_processor.get_spectrogram(audio)
        dB = 20 * tf.math.log(spectrogram + 1.e-6)
        energy = tf.math.reduce_mean(dB, axis=1)
        min_energy = tf.reduce_min(energy)
        rel_energy = energy - min_energy
        non_silence = rel_energy > self.dBthres
        non_silence_frames = tf.math.reduce_sum(tf.cast(non_silence, tf.float32))
        non_silence_duration = self.frame_length_in_s + self.frame_step_in_s * (non_silence_frames - 1)
        
        return 0 if non_silence_duration > self.duration_thres else 1

# Temperature and Humidity Measurement
def measure_upload_temperature_humidity(redis_client, device, upload=True):
    try:
        temperature = device.temperature
        humidity = device.humidity
        timestamp = time.time()
        formatted_datetime = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        timestamp_ms = int(timestamp * 1000)
        print(f"{formatted_datetime}: Temperature: {temperature}°C, Humidity: {humidity}%")
        
        # upload the collected data to Redis
        if upload:
            redis_client.ts().add(DB_TEMP_NAME, timestamp_ms, temperature)
            redis_client.ts().add(DB_HUMO_NAME, timestamp_ms, humidity)

    except RuntimeError:
        print("Sensor failure: DHT sensor not found, check wiring")

# define the Voice User Interface (VUI) class
class VUI:
    def __init__(self, model_file_path, preprocessing_args, normalization_type=tf.int16):
        # order of the labels
        self.LABELS = ['down', 'up']
        self.model_file_path = model_file_path
        self.preprocessing_args = preprocessing_args
        self.normalization = Normalization(normalization_type)
        
        # Load and prepare the model
        self.load_model()
        
        # Set up feature processor
        self.setup_feature_processor()

    def load_model(self):
        if self.model_file_path.endswith('.zip'):
            with zipfile.ZipFile(self.model_file_path, 'r') as fp:
                fp.extractall('/tmp/')
                model_filename = fp.namelist()[0]
                self.model_file_path = '/tmp/' + model_filename

        self.interpreter = tf.lite.Interpreter(model_path=self.model_file_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def setup_feature_processor(self):
        if self.preprocessing_args['num_coefficients'] == 0:
            self.preprocessing_args.pop('num_coefficients')
            feature_processor = MelSpectrogram(**self.preprocessing_args)
            self.feature_processor_fn = feature_processor.get_mel_spec
        else:
            feature_processor = MFCC(**self.preprocessing_args)
            self.feature_processor_fn = feature_processor.get_mfccs

    def classify(self, audio_tensor):
        # Normalize and process features
        features = self.normalization.normalize_audio(audio_tensor)
        features = self.feature_processor_fn(features)
        features = tf.expand_dims(features, 0)
        features = tf.expand_dims(features, -1)

        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], features)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output
        
# Audio processing callback for VAD
def process_audio(indata, vad, classifier):
    global DATA_COLLECTION_ENABLED, LAST_TOGGLE_TIME

    # Downsample to 16 kHz from 48 kHz
    audio_data = indata.flatten().astype(np.float32)
    downsampling_factor = 3  # For 48 kHz to 16 kHz
    audio_resampled = signal.resample_poly(audio_data, up=1, down=downsampling_factor)

    # Convert the resampled audio back to a tf.Tensor, squeeze, and normalize
    audio_tensor = tf.convert_to_tensor(audio_resampled, dtype=tf.float32)
    audio_tensor = tf.squeeze(audio_tensor)
    max_range = np.iinfo(np.int16).max
    audio_normalized_vad = audio_tensor / max_range

    # Run VAD to check for voice activity
    is_silence = vad.is_silence(audio_normalized_vad)
    
    # If the VAD returns silence, maintain the current state.
    if is_silence:
        return
    else:

        LAST_TOGGLE_TIME = time.time()
        formatted_datetime = datetime.fromtimestamp(LAST_TOGGLE_TIME).strftime('%Y-%m-%d %H:%M:%S')
        # If the VAD returns non-silence, the recording is fed to the classification model for “up/down” spotting
        # Enable or disable data collection based on voice commands
        # audio_tensor is the resampled audio data from 48 kHz to 16 kHz
        # change the audio_tensor to the required format for the model
        output = classifier.classify(audio_tensor)

        # output is an array of probabilities of each label
        # print(f"output of classifier model is {output}")
        up_prob = output[0][1]
        down_prob = output[0][0]
        # If the predicted keyword is “up” with probability > 99%, enable data collection.
        if up_prob > 0.99:
            # only print the message when the `DATA_COLLECTION_ENABLED` is changing
            if not DATA_COLLECTION_ENABLED:
                print(F"{formatted_datetime}: Data collection enabled.")
            DATA_COLLECTION_ENABLED = True
        # If the predicted keyword is “down” with probability > 99%, stop data collection.
        elif down_prob > 0.99:
            # only print the message when the `DATA_COLLECTION_ENABLED` is changing
            if DATA_COLLECTION_ENABLED:
                print(F"{formatted_datetime}: Data collection disabled.")
            DATA_COLLECTION_ENABLED = False
        # If the top-1 probability (regardless of the predicted label) is ≤ 99%, remain in the current state.
        # else:
            # print(f"{formatted_datetime}: No action taken.")

def create_audio_callback(vad, classifier):
    def audio_callback(indata, frames, time, status):
        process_audio(indata, vad, classifier)
    return audio_callback

if __name__ == '__main__':
    # get server authentication details from the command line
    # python3 Ex1.py --host <host> --port <port> --user <user> --password <password>
    parser = argparse.ArgumentParser(description="Temperature & Humidity Monitoring System")
    parser.add_argument("--host", type=str, required=True, help="Redis Cloud host")
    parser.add_argument("--port", type=int, required=True, help="Redis Cloud port")
    parser.add_argument("--user", type=str, required=True, help="Redis Cloud username")
    parser.add_argument("--password", type=str, required=True, help="Redis Cloud password")

    args = parser.parse_args()

    # Connect to Redis
    redis_client = get_redis_client(args)

    if redis_client == -1:
        print("Failed to connect to Redis. Exiting.")
        exit()
    
    # Create the necessary databases
    create_needed_dbs(redis_client)

    # Initialize the VAD
    vad = VAD(sampling_rate=16000, frame_length_in_s=0.032, frame_step_in_s=0.016, dBthres=10, duration_thres=0.1)
    
    MODEL_FILE_PATH = "./tflite_models/Group06.tflite"

    PREPROCESSING_ARGS = {
        'sampling_rate': 16000,
        'frame_length_in_s': 0.04,
        'frame_step_in_s': 0.02,
        'num_mel_bins': 40,
        'lower_frequency': 20,
        'upper_frequency': 4000,
        'num_coefficients': 10  # set num_coefficients to 0 if log-Mel Spectrogram features have been used
    }

    # Initialize the VUI
    classifier = VUI(MODEL_FILE_PATH, PREPROCESSING_ARGS)

    # Create the callback function with the classifier instance
    audio_callback = create_audio_callback(vad, classifier)

    # Initialize  DHT11
    dht_device = adafruit_dht.DHT11(D4)

    # Audio recording configuration with InputStream
    with sd.InputStream(device=DEVICE_ID, channels=CHANNELS, dtype='int16', samplerate=SAMPLE_RATE, blocksize=SAMPLE_RATE, callback=audio_callback):
        print("Voice Activity Detection initialized. Press 'Ctrl+C' to stop.")
        last_measurement_time = time.time()

        try:
            # always runs in background and continuously records audio data
            while True:
                # perform data collection every 2 seconds
                if DATA_COLLECTION_ENABLED and (time.time() - last_measurement_time) >= DATA_INTERVAL_SECONDS:
                    measure_upload_temperature_humidity(redis_client, dht_device)
                    last_measurement_time = time.time()
                    
                time.sleep(1)  # Run the loop every second to check for voice commands

        except KeyboardInterrupt:
            print("Stopping voice activity detection and data collection.")
        