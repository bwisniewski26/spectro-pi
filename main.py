import numpy as np
import subprocess
import os
import board
import neopixel
import time

# konfiguracja
samplerate = 16000
frames_per_chunk = 1024
bytes_per_sample = 4  # S32_LE
channels = 1
# Ustawienia diod LED
pixel_pin = board.D12
num_pixels = 256
brightness = 0.05
i1 = 1
# Inicjalizacja diod LED
pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=brightness, auto_write=False)

# Metoda do mapowania indeksu kolumny do indeksu amplitudy
def map_index_to_amplitude(index, num_segments):
    # Mapa kolumny do segmentu
    segment_index = index // (num_pixels // num_segments)
    # Mapa segmentu do amplitudy
    return segment_index

# Metoda zwracająca indeksy diód LED w kolumnie
def get_led_indices(column_index, num_segments):
    # Oblicz indeksy diód LED w kolumnie
    start_index = column_index * (num_pixels // num_segments)
    end_index = start_index + (num_pixels // num_segments)
    return list(range(start_index, end_index))

# Metoda wyświetlająca amplitudy na diodach LED
def display_amplitudes(amplitudes):
    min_brightness = 10  # Minimalna jasność (0-255)
    for i in range(num_pixels):
         pixel_index = map_index_to_amplitude(i, len(amplitudes))
         led_indices = get_led_indices(pixel_index, len(amplitudes))
         print(led_indices)
         color_intensity = max(int(amplitudes[pixel_index] * 255), min_brightness)
         for led_index in range(num_pixels):
            pixels[led_index] = (color_intensity, color_intensity, color_intensity)
    pixels.show()

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

        # Podziel widmo na 32 segmentów częstotliwości
        num_segments = 32
        max_freq = samplerate / 2  # maksymalna częstotliwość (Nyquist)
        segment_edges = np.linspace(0, max_freq, num_segments + 1)  # granice segmentów

        # Oblicz sumę amplitud w każdym segmencie
        segment_amplitudes = []
        for i in range(num_segments):
            segment_mask = (freqs >= segment_edges[i]) & (freqs < segment_edges[i + 1])
            segment_amplitudes.append(np.sum(spectrum[segment_mask]))

        # Dodaj offset na niskie tony (np. zwiększ amplitudy w pierwszym segmencie)
        bass_boost_factor = 0.50  # współczynnik wzmocnienia niskich tonów
        segment_amplitudes[0] *= bass_boost_factor * bass_boost_factor  # wzmocnienie pierwszego segmentu
        for segment in segment_amplitudes:
            segment +=  segment*bass_boost_factor
        # Normalizuj amplitudy do zakresu 0-1
        max_amplitude = max(segment_amplitudes) if max(segment_amplitudes) > 0 else 1
        normalized_amplitudes = [amp / max_amplitude for amp in segment_amplitudes]

        # Wyświetl widmo w konsoli
        # os.system('clear')  # wyczyść konsolę
        # for amplitude in normalized_amplitudes:
               # bar_height = int(amplitude*50)
               # print("|" * bar_height)
        display_amplitudes(normalized_amplitudes)
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Zatrzymano.")
    process.terminate()
