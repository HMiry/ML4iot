"""
This example demonstrates how to set up a connection to Redis Cloud using Python 
and how to write/read a simple key-value pair to the database.

To connect to your personal database, update the host, port, username, and password 
with your own credentials.
To find this information: 
- sign in at https://cloud.redis.io
- navigate to Databases
- click on the name of your database.
"""


import redis


# Connect to Redis (not working with PoliTO-WiFi, use hotspot or personal WiFi)
REDIS_HOST = 'your-host' # strip :<port> from REDIS_HOST!
REDIS_PORT = 12345 # this is the right place for the port!
REDIS_USERNAME = 'default'
REDIS_PASSWORD = 'your-password'


redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, username=REDIS_USERNAME, password=REDIS_PASSWORD)
is_connected = redis_client.ping()
print('Redis Connected:', is_connected)

written = redis_client.set("message", "Hello World!")
print('Message written:', written)

msg = redis_client.get("message")
print(msg.decode())