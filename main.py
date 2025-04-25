import numpy as np
import subprocess

# konfiguracja
samplerate = 16000
frames_per_chunk = 1024
bytes_per_sample = 4  # S32_LE
channels = 1

cmd = [
    "arecord",
    "-D", "plughw:1,0",        # karta GoogleVoiceHat
    "-f", "S32_LE",
    "-c", str(channels),
    "-r", str(samplerate),
    "-t", "raw"                # raw output
]

# uruchom arecord jako proces
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=frames_per_chunk * bytes_per_sample)

try:
    while True:
        # czytaj próbki z arecord
        raw_audio = process.stdout.read(frames_per_chunk * bytes_per_sample)
        if not raw_audio:
            break

        samples = np.frombuffer(raw_audio, dtype=np.int32).astype(np.float32)
        samples /= np.max(np.abs(samples))  # normalizacja

        spectrum = np.abs(np.fft.fft(samples))[:256]
        spectrum = np.reshape(spectrum, (8, 32))
        print(spectrum)

except KeyboardInterrupt:
    print("Zatrzymano.")
    process.terminate()
