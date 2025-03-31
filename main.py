# SpectroPi Python script for Raspberry Pi

try:
    import board
    import neopixel_spi
    import pyaudio
    import numpy as np
except: 
    print("Failed to import required libraries!")
    exit(1)


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

audio = pyaudio.PyAudio()

# TODO: Further script implementation