import sounddevice as sd
import numpy as np
import tensorflow as tf
from scipy import signal
import time
import adafruit_dht
from datetime import datetime
from board import D4

# Initialize states and timing
data_collection_enabled = False
last_toggle_time = time.time()

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

# Initialize VAD and DHT11

vad = VAD(sampling_rate=16000, frame_length_in_s=0.032, frame_step_in_s=0.016, dBthres=10, duration_thres=0.1)
dht_device = adafruit_dht.DHT11(D4)

# Temperature and Humidity Measurement
def measure_temperature_humidity():
    try:
        temperature = dht_device.temperature
        humidity = dht_device.humidity
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"{timestamp} - Temperature: {temperature}°C, Humidity: {humidity}%")
    except RuntimeError:
        print("Sensor failure: DHT sensor not found, check wiring")

# Audio processing callback for VAD
def process_audio(indata, frames, time_info, status):
    global data_collection_enabled, last_toggle_time

    # Downsample to 16 kHz from 48 kHz
    audio_data = indata.flatten().astype(np.float32)
    downsampling_factor = 3  # For 48 kHz to 16 kHz
    audio_resampled = signal.resample_poly(audio_data, up=1, down=downsampling_factor)

    # Convert the resampled audio back to a tf.Tensor, squeeze, and normalize
    audio_tensor = tf.convert_to_tensor(audio_resampled, dtype=tf.float32)
    audio_tensor = tf.squeeze(audio_tensor)
    max_range = np.iinfo(np.int16).max
    audio_normalized = audio_tensor / max_range

    # Run VAD to check for voice activity
    is_silence = vad.is_silence(audio_normalized)

    # Toggle data collection state if non-silence (voice command detected)
    if is_silence == 0 and (time.time() - last_toggle_time) >= 5:
        data_collection_enabled = not data_collection_enabled
        last_toggle_time = time.time()
        state = "enabled" if data_collection_enabled else "disabled"
        print(f"Data collection has been {state}.")


if __name__ == '__main__':
    # Configure audio input with 1 channel, 16-bit depth, and 48 kHz sampling rate
    device_id = 1  # USB microphone device ID (check this on your system if needed)
    channels = 1
    samplerate = 48000  # Sample rate for InputStream
    length_of_1s = samplerate  # Block size to capture 1 second of audio

    # Audio recording configuration with InputStream
    with sd.InputStream(device=device_id, channels=channels, dtype='int16', samplerate=samplerate, blocksize=length_of_1s, callback=process_audio):
        print("Voice Activity Detection initialized. Press 'Ctrl+C' to stop.")
        last_measurement_time = time.time()

        try:
            while True:
                # Check every 2 seconds if data collection is enabled
                if data_collection_enabled and (time.time() - last_measurement_time) >= 2:
                    measure_temperature_humidity()
                    last_measurement_time = time.time()
                    
                time.sleep(1)  # Run the loop every second to check for voice commands

        except KeyboardInterrupt:
            print("Stopping voice activity detection and data collection.")
