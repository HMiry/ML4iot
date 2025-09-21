import redis
import adafruit_dht
import time
import argparse
from datetime import datetime
from board import D4

# Constants
DATA_INTERVAL_SECONDS = 2          # 2 seconds data acquisition interval
THIRTY_DAYS_RETENTION_MS = 30 * 24 * 60 * 60 * 1000      # 30 days in milliseconds
ONE_YEAR_RETENTION_MS = 365 * 24 * 60 * 60 * 1000        # 365 days in milliseconds
BUCKET_SIZE_MS = 3600 * 1000                             # 1 hour in milliseconds for aggregation
MAC_ADDRESS = "e4:5f:01:e8:9b:d0"                        # MAC address of the device

# Define timeseries names using the MAC address
db_temp_name = MAC_ADDRESS + ":temperature"
db_humi_name = MAC_ADDRESS + ":humidity"
db_temp_min = MAC_ADDRESS + ":temperature_min"
db_temp_max = MAC_ADDRESS + ":temperature_max"
db_temp_avg = MAC_ADDRESS + ":temperature_avg"
db_humi_min = MAC_ADDRESS + ":humidity_min"
db_humi_max = MAC_ADDRESS + ":humidity_max"
db_humi_avg = MAC_ADDRESS + ":humidity_avg"

# Function to create a timeseries on redis, only if it doesn't exist
def create_timeseries_if_not_exists(client, key, retention_msecs, chunk_size=128, duplicate_policy="last"):
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
        create_timeseries_if_not_exists(redis_client, db_temp_name, THIRTY_DAYS_RETENTION_MS)
        create_timeseries_if_not_exists(redis_client, db_humi_name, THIRTY_DAYS_RETENTION_MS)

        # Aggregated time series for min, max, avg with 1-hour buckets
        create_timeseries_if_not_exists(redis_client, db_temp_min, ONE_YEAR_RETENTION_MS)
        create_timeseries_if_not_exists(redis_client, db_temp_max, ONE_YEAR_RETENTION_MS)
        create_timeseries_if_not_exists(redis_client, db_temp_avg, ONE_YEAR_RETENTION_MS)
        create_timeseries_if_not_exists(redis_client, db_humi_min, ONE_YEAR_RETENTION_MS)
        create_timeseries_if_not_exists(redis_client, db_humi_max, ONE_YEAR_RETENTION_MS)
        create_timeseries_if_not_exists(redis_client, db_humi_avg, ONE_YEAR_RETENTION_MS)

        # Aggregation rules for hourly min, max, avg
        redis_client.ts().createrule(db_temp_name, db_temp_min, 'min', bucket_size_msec=BUCKET_SIZE_MS)
        redis_client.ts().createrule(db_temp_name, db_temp_max, 'max', bucket_size_msec=BUCKET_SIZE_MS)
        redis_client.ts().createrule(db_temp_name, db_temp_avg, 'avg', bucket_size_msec=BUCKET_SIZE_MS)
        redis_client.ts().createrule(db_humi_name, db_humi_min, 'min', bucket_size_msec=BUCKET_SIZE_MS)
        redis_client.ts().createrule(db_humi_name, db_humi_max, 'max', bucket_size_msec=BUCKET_SIZE_MS)
        redis_client.ts().createrule(db_humi_name, db_humi_avg, 'avg', bucket_size_msec=BUCKET_SIZE_MS)

    except redis.ResponseError as e:
        print(f'Error setting up timeseries or aggregation rules: {e}')
        return(-1)

# parse command line arguments and return the Redis client
def get_redis_client(args):
    redis_client = redis.Redis(
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

    # Create the necessary databases
    create_needed_dbs(redis_client)

    # Initialize the DHT11 sensor
    dht_device = adafruit_dht.DHT11(D4)
    print('Collecting temperature and humidity data every 2 seconds...')

    # Collect and store sensor data every 2 seconds
    while True:
        timestamp_ms = int(time.time() * 1000)
        formatted_datetime = datetime.fromtimestamp(timestamp_ms / 1000).strftime('%Y-%m-%d %H:%M:%S.%f')

        try:
            # Read temperature and humidity from the DHT11 sensor
            temperature = dht_device.temperature
            humidity = dht_device.humidity

            # Print the readings
            print(f'{formatted_datetime} - {db_temp_name} = {temperature}')
            print(f'{formatted_datetime} - {db_humi_name} = {humidity}')

            # Add data points to Redis TimeSeries
            redis_client.ts().add(db_temp_name, timestamp_ms, temperature)
            redis_client.ts().add(db_humi_name, timestamp_ms, humidity)

        except RuntimeError:
            # Handle sensor errors
            print(f'{formatted_datetime} - Sensor failure: DHT sensor not found, check wiring')
            dht_device.exit()
            dht_device = adafruit_dht.DHT11(D4)  # Reinitialize the sensor

        # Wait for 2 seconds before the next reading
        time.sleep(DATA_INTERVAL_SECONDS)