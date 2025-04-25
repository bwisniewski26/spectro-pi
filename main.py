try:
    import numpy as np
    import subprocess
    import os
    import board
    import neopixel
except ImportError as e:
    print("Nie można zaimportować wymaganych modułów. Upewnij się, że są zainstalowane.")
    exit(1)

# konfiguracja
samplerate = 16000
frames_per_chunk = 1024
bytes_per_sample = 4  # S32_LE
channels = 1
# Ustawienia diod LED
pixel_pin = board.D12
num_pixels = 256
brightness = 0.1

# Inicjalizacja diod LED
pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=brightness, auto_write=False)


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
        max_amplitude = max(segment_amplitudes) if max(segment_amplitudes) > 0 else 1  # Zabezpieczenie przed zerem
        normalized_amplitudes = [amp / max_amplitude for amp in segment_amplitudes]

        # Zabezpieczenie przed NaN w normalizowanych amplitudach
        normalized_amplitudes = [0 if np.isnan(amp) else amp for amp in normalized_amplitudes]

        # Wyświetl widmo w konsoli
        os.system('clear')  # wyczyść konsolę
        for amplitude in normalized_amplitudes:
            bar_height = int(amplitude * 20)  # wysokość paska (max 20 znaków)
            print("█" * bar_height)

        # Narysuj kolory na diodach LED
        for i in range(num_pixels):
            # Oblicz kolor na podstawie segmentu
            segment_index = int(i / (num_pixels / num_segments))
            if segment_index < num_segments:
                color_value = int(normalized_amplitudes[segment_index] * 255)
                pixels[i] = (color_value, 0, 255 - color_value)

        # Zaktualizuj diody LED
        pixels.show()
        

except KeyboardInterrupt:
    print("Zatrzymano.")
    process.terminate()