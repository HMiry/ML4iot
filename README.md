# Machine Learning for IoT (ML4IoT) - Group 06

This repository contains three projects for the Machine Learning for IoT course, each focusing on different aspects of IoT systems, machine learning, and data processing.

## 📁 Project Structure

```
ML4iot/
├── Project1/     # Voice Activity Detection
├── Project2/     # Keyword Spotting & Smart Hygrometer
├── Project3/     # MQTT/REST Communication
└── README.md     # This file
```

---

## 🎤 Project 1: Voice Activity Detection (VAD)

**Objective**: Implement and optimize Voice Activity Detection for IoT applications using audio signal processing.

### 📂 Files
- `ex1.py` - Temperature/Humidity monitoring with Redis integration
- `ex2.py` - Voice Activity Detection implementation
- `HW1_ Voice Activity Detection.ipynb` - Jupyter notebook with VAD analysis
- `redis.ipynb` - Redis data analysis and visualization
- `vad_latency.py` - VAD performance measurement
- `Group06_Homework1.pdf` - Project report

### 🚀 Key Features
- **VAD Implementation**: Real-time voice activity detection using spectrogram analysis
- **Sensor Integration**: DHT11 temperature and humidity sensor data collection
- **Redis Integration**: Time-series data storage with aggregation (min, max, avg)
- **Performance Optimization**: Latency measurement and optimization

### 📊 Technical Specifications
- Audio sampling rate: 48 kHz → 16 kHz (downsampling)
- Frame length: 32 ms, Frame step: 16 ms
- VAD threshold: 10 dB, Duration threshold: 100 ms
- Data collection interval: 2 seconds
- Redis retention: 30 days (raw), 1 year (aggregated)

### 🛠️ Usage
```bash
cd Project1
python ex1.py --host <redis_host> --port <redis_port> --user <username> --password <password>
```

---

## 🎯 Project 2: Keyword Spotting & Smart Hygrometer

**Objective**: Develop a voice-controlled smart hygrometer using "up/down" keyword spotting with TensorFlow Lite deployment.

### 📂 Files
- `ex1.py` - Main smart hygrometer implementation with VUI
- `training.ipynb` - Model training notebook (Deepnote)
- `Group06.tflite` - Trained TFLite model
- `hw2_latency.py` - Performance measurement script
- `Group06_Homework2.pdf` - Project report

### 🚀 Key Features
- **Keyword Spotting**: "up/down" voice commands with >99% accuracy
- **Smart Hygrometer**: Voice-controlled temperature/humidity monitoring
- **VUI Integration**: Voice User Interface with VAD and KWS
- **Redis Integration**: Automated data upload with voice control
- **Model Constraints**: <50KB size, <40ms latency

### 📊 Technical Specifications
- Model accuracy: >99.4% on test set
- TFLite model size: <50 KB
- Total latency: <40 ms on Raspberry Pi
- Preprocessing: MFCCs or Log-Mel Spectrograms
- Voice commands: "up" (enable), "down" (disable)
- Probability threshold: >99% for action

### 🛠️ Usage
```bash
cd Project2
python ex1.py --host <redis_host> --port <redis_port> --user <username> --password <password>
```

### 🎯 Model Performance
- **Accuracy**: >99.4%
- **Size**: <50 KB
- **Latency**: <40 ms
- **Features**: MFCCs with 10 coefficients
- **Architecture**: Optimized CNN for edge deployment

---

## 📡 Project 3: MQTT/REST Communication

**Objective**: Implement IoT data communication using MQTT pub/sub pattern and REST API for data visualization.

### 📂 Files
- `publisher.py` - MQTT publisher for sensor data
- `rest_server.ipynb` - REST API server implementation
- `rest_client.ipynb` - Client for data visualization and analysis
- `Group06_Homework3.pdf` - Project report

### 🚀 Key Features
- **MQTT Communication**: Publish/subscribe pattern for IoT data
- **REST API**: HTTP endpoints for data retrieval and analysis
- **Real-time Visualization**: Data plotting and analysis
- **Sensor Integration**: DHT11 temperature/humidity monitoring
- **Data Storage**: Redis time-series database integration

### 📊 Technical Specifications
- MQTT Broker: Eclipse Mosquitto (mqtt.eclipseprojects.io)
- Port: 1883
- Topic: s334033
- Data format: JSON with timestamp
- MAC Address: 0xe45f01e89bd0
- Collection interval: 2 seconds

### 🛠️ Usage

**Publisher (Raspberry Pi)**:
```bash
cd Project3
python publisher.py
```

**REST Server (Deepnote)**:
```bash
# Run rest_server.ipynb cells in Deepnote
```

**Data Visualization**:
```bash
# Run rest_client.ipynb cells for analysis
```

### 🌐 API Endpoints
- `GET /status` - Server health check
- `GET /data/{mac_address}/{sensor}` - Retrieve sensor data
- `GET /data/{mac_address}/{sensor}/last` - Get latest reading
- `GET /data/{mac_address}/{sensor}/range` - Get data range

---

## 🔧 Prerequisites

### Hardware
- Raspberry Pi with DHT11 sensor
- USB microphone
- Internet connection

### Software Dependencies
```bash
pip install tensorflow
pip install redis
pip install adafruit-circuitpython-dht
pip install sounddevice
pip install scipy
pip install paho-mqtt
pip install requests
pip install pandas
pip install matplotlib
```

### External Services
- **Redis Cloud**: For time-series data storage
- **Deepnote**: For model training and REST API hosting
- **Eclipse MQTT**: For pub/sub communication

---

## 📈 Key Achievements

### Project 1 - VAD Optimization
- ✅ Real-time voice activity detection
- ✅ Optimized latency for edge deployment
- ✅ Redis time-series integration
- ✅ Sensor data aggregation

### Project 2 - Keyword Spotting
- ✅ >99.4% accuracy on test set
- ✅ <50 KB model size
- ✅ <40 ms total latency
- ✅ Voice-controlled IoT system

### Project 3 - Communication Protocols
- ✅ MQTT pub/sub implementation
- ✅ REST API for data access
- ✅ Real-time data visualization
- ✅ Cross-platform communication

---

## 📄 License

This project is part of academic coursework and is intended for educational purposes only.
