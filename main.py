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
        # Przygotuj obliczenie danych widma audio
         
        samples = np.frombuffer(raw_audio, dtype=np.int32).astype(np.float32)
        samples = samples - np.mean(samples)
        samples /= np.max(np.abs(samples))  # normalizacja
        # Oblicz widmo
        spectrum = np.abs(np.fft.fftfreq(len(samples), samples))[:256]

        spectrum = np.reshape(spectrum, (8, 32))

        # Wyświetl na moc dźwięku o danych częstotliwościach w czytelnej formie
        print("Widmo audio:")
        for i in range(8):
            for j in range(32):
                if spectrum[i][j] > 0.1:
                    print(f"Pasmo {i*32+j}: {spectrum[i][j]:.2f}", end=" ")
                else:
                    print("Pasmo {i*32+j}: 0.00", end=" ")
            print()

except KeyboardInterrupt:
    print("Zatrzymano.")
    process.terminate()
