#!/usr/bin/env python
import time
import json
import uuid
import adafruit_dht
from board import D4
import paho.mqtt.client as mqtt
from datetime import datetime

# Constants
BROKER = "mqtt.eclipseprojects.io"
PORT = 1883
TOPIC = "s334033"  
DATA_INTERVAL = 2  # seconds
MAC_ADDRESS = "0xe45f01e89bd0"

def get_sensor_data(dht_device):
    """Get temperature and humidity from DHT11 sensor"""
    try:
        temperature = dht_device.temperature
        humidity = dht_device.humidity
        return temperature, humidity
    except RuntimeError:
        # Error handling for sensor read failures
        print("Sensor failure. Check wiring.")
        return None, None
    except Exception as error:
        print(f"Other error occurred: {error}")
        return None, None

def create_message(temperature, humidity):
    """Create JSON message with sensor data"""
    if temperature is None or humidity is None:
        return None
        
    message = {
        "mac_address": MAC_ADDRESS,
        "timestamp": int(time.time() * 1000),  # Convert to milliseconds
        "temperature": int(temperature),
        "humidity": int(humidity)
    }
    return json.dumps(message)

def on_connect(client, userdata, flags, rc):
    """Callback for when the client receives a CONNACK response from the server"""
    if rc == 0:
        print("Connected to MQTT broker")
    else:
        print(f"Connection failed with code {rc}")

def main():
    # Initialize DHT11 sensor
    dht_device = adafruit_dht.DHT11(D4)
    
    # Initialize MQTT client
    client = mqtt.Client()
    client.on_connect = on_connect
    
    try:
        # Connect to MQTT broker
        print(f"Connecting to broker {BROKER}")
        client.connect(BROKER, PORT, 60)
        client.loop_start()
        
        # Main loop
        while True:
            # Get sensor readings
            temperature, humidity = get_sensor_data(dht_device)
            
            # Create and publish message
            message = create_message(temperature, humidity)
            if message:
                client.publish(TOPIC, message)
                print(f"{datetime.now()}: Published {message}")
            
            # Wait for next reading
            time.sleep(DATA_INTERVAL)
            
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        # Clean up
        client.loop_stop()
        client.disconnect()
        dht_device.exit()

if __name__ == "__main__":
    main()
