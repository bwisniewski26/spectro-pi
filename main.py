# SpectroPi Python script for Raspberry Pi

try:
    import sounddevice as sd
    import neopixel_spi
    import numpy as np
except: 
    print("Failed to import required libraries!")
    exit(1)


def audio_callback(indata, frames, time, status):
    samples = indata[:, 0] # mono audio
    spectrum = np.abs(np.fft.fft(samples))
    # map to LED matrix 32x8 to show spectrum from low to high
    spectrum = np.reshape(spectrum, (8, 32))
    print(spectrum)

stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=44100, blocksize=1024)
stream.start()

