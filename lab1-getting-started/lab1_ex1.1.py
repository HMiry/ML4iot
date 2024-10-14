"""
a. Create a new Python file (e.g., lab1_ex1.py) and write a script that uses the sounddevice package to record audio data. 
Stop the recording when the Q key is pressed.
"""


import sounddevice as sd

with sd.InputStream(device=1, channels=1, dtype='int32', samplerate=48000):
    while True:
        key = input()
        if key in ('q', 'Q'):
            print('Stop recording.')
            break
