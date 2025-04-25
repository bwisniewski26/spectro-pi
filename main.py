import numpy as np
import subprocess
import os

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

        # Przygotuj obliczenie danych widma audio
        samples = np.frombuffer(raw_audio, dtype=np.int32).astype(np.float32)
        samples = samples - np.mean(samples)
        samples /= np.max(np.abs(samples))  # normalizacja

        # Oblicz widmo
        spectrum = np.abs(np.fft.fft(samples))[:frames_per_chunk // 2]

        # Podziel widmo na 8 części
        spectrum_chunks = np.array_split(spectrum, 8)

        # Wyświetl widmo
        os.system('clear')  # wyczyść konsolę
        for chunk in spectrum_chunks:
            for value in chunk:
                if value > 0.1:
                    print("█", end="")
                else:
                    print(" ", end="")
            print()

except KeyboardInterrupt:
    print("Zatrzymano.")
    process.terminate()
