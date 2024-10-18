"""
This example shows how to enable/disable lossless compression in Redis TimeSeries.
Moreover, it shows how to retrieve the memory usage and the # records of a TimeSeries.
"""
import redis
from time import sleep
from time import time


# Connect to Redis
REDIS_HOST = 'your-host'
REDIS_PORT = 11938
REDIS_USERNAME = 'default'
REDIS_PASSWORD = 'your-password'

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, username=REDIS_USERNAME, password=REDIS_PASSWORD)
is_connected = redis_client.ping()
print('Redis Connected:', is_connected)


# Create a TimeSeries with chunk size 128 bytes
# By default, compression is enabled
try:
    redis_client.ts().create('test', chunk_size=128)
except redis.ResponseError:
    pass

print('===test info===')
print('Memory Usage (bytes):', redis_client.ts().info('test').memory_usage)
print('Total Samples:', redis_client.ts().info('test').total_samples)
print('# of Chunks:',  redis_client.ts().info('test').chunk_count)
print()

print('Adding 100 values to "test"')
print()
for i in range(100):
    timestamp_ms = int(time() * 1000)
    redis_client.ts().add('test', timestamp_ms, 25 + i // 50)
    sleep(0.1) # is always this 

print('===test info===')
print('Memory Usage (bytes):', redis_client.ts().info('test').memory_usage)
print('Total Samples:', redis_client.ts().info('test').total_samples)
print('# of Chunks:',  redis_client.ts().info('test').chunk_count)
print()

# Disable compression
try:
    redis_client.ts().create('test_uncompressed', chunk_size=128, uncompressed=True)
except redis.ResponseError:
    pass

print('Adding 100 values to "test_uncompressed"')
for i in range(100):
    timestamp_ms = int(time() * 1000)
    redis_client.ts().add('test_uncompressed', timestamp_ms, 25 + i // 50)
    sleep(0.1)

print('===test_uncompressed info===')
print('Memory Usage (bytes):', redis_client.ts().info('test_uncompressed').memory_usage)
print('Total Samples:', redis_client.ts().info('test_uncompressed').total_samples)
print('# of Chunks:',  redis_client.ts().info('test_uncompressed').chunk_count )
print()

compressed_memory = redis_client.ts().info('test').memory_usage
uncompressed_memory = redis_client.ts().info('test_uncompressed').memory_usage
savings = 100 * (uncompressed_memory - compressed_memory) / uncompressed_memory
print(f'Memory Savings: {savings:.2f}%')
