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
        spectrum = np.abs(np.fft.rfft(samples))  # używamy rfft, bo dane są rzeczywiste
        freqs = np.fft.rfftfreq(len(samples), d=1/samplerate)

        # Podziel widmo na 8 segmentów częstotliwości
        num_segments = 8
        max_freq = samplerate / 2  # maksymalna częstotliwość (Nyquist)
        segment_edges = np.linspace(0, max_freq, num_segments + 1)  # granice segmentów

        # Oblicz sumę amplitud w każdym segmencie
        segment_amplitudes = []
        for i in range(num_segments):
            segment_mask = (freqs >= segment_edges[i]) & (freqs < segment_edges[i + 1])
            segment_amplitudes.append(np.sum(spectrum[segment_mask]))

        # Zmniejsz wpływ niskich tonów (pierwszy segment)
        bass_reduction_factor = 0.5  # współczynnik redukcji niskich tonów
        segment_amplitudes[0] *= bass_reduction_factor  # zmniejszenie pierwszego segmentu

        # Normalizuj amplitudy do zakresu 0-1
        max_amplitude = max(segment_amplitudes) if max(segment_amplitudes) > 0 else 1
        normalized_amplitudes = [amp / max_amplitude for amp in segment_amplitudes]

        # Wyświetl widmo w konsoli
        os.system('clear')  # wyczyść konsolę
        for amplitude in normalized_amplitudes:
            bar_height = int(amplitude * 20)  # wysokość paska (max 20 znaków)
            print("█" * bar_height)

except KeyboardInterrupt:
    print("Zatrzymano.")
    process.terminate()