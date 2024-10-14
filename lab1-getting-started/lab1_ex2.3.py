"""
This example shows how to create timeseries on Redis.
"""


import redis
from time import time


REDIS_HOST = 'your-host'
REDIS_PORT = 11938
REDIS_USERNAME = 'default'
REDIS_PASSWORD = 'your-password'


redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, username=REDIS_USERNAME, password=REDIS_PASSWORD)
is_connected = redis_client.ping()
print('Redis Connected:', is_connected)


# Create a timeseries named 'integers'
try:
    redis_client.ts().create('integers')
except redis.ResponseError:
    # Ignore error if the timeseries already exists
    pass

# Add a new value to the timeseries
timestamp_ms = int(time() * 1000)  # Redis TS requires the timestamp in ms
redis_client.ts().add('integers', timestamp_ms, 1)

# Read the last added values
last_timestamp_ms, last_value = redis_client.ts().get('integers')
print(last_timestamp_ms, last_value)

timestamp_ms = int(time() * 1000)
redis_client.ts().add('integers', timestamp_ms, 2)
last_timestamp_ms, last_value = redis_client.ts().get('integers')
print(last_timestamp_ms, last_value)
