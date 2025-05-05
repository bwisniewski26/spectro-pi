import numpy as np
import subprocess
import board
import neopixel
import time

# audio config
samplerate = 16000
frames_per_chunk = 1024
bytes_per_sample = 4  # S32_LE
channels = 1

# matrix config
matrix_width = 32
matrix_height = 32
num_pixels = matrix_width * matrix_height
brightness = 0.05
pixel_pin = board.D12

pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=brightness, auto_write=False)

# maps (x, y) to linear index assuming serpentine layout
def xy_to_index(x, y):
    if y % 2 == 0:
        return y * matrix_width + x
    else:
        return y * matrix_width + (matrix_width - 1 - x)

# render fft amplitudes as vertical bars
def display_amplitudes(amplitudes):
    pixels.fill((0, 0, 0))
    for x, amp in enumerate(amplitudes):
        height = int(amp * matrix_height)
        for y in range(height):
            idx = xy_to_index(x, matrix_height - 1 - y)
            pixels[idx] = (0, 255, 0)  # green
    pixels.show()

# arecord process
cmd = [
    "arecord",
    "-D", "plughw:1,0",
    "-f", "S32_LE",
    "-c", str(channels),
    "-r", str(samplerate),
    "-t", "raw"
]

process = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=frames_per_chunk * bytes_per_sample)

try:
    while True:
        raw_audio = process.stdout.read(frames_per_chunk * bytes_per_sample)
        if not raw_audio:
            break

        samples = np.frombuffer(raw_audio, dtype=np.int32).astype(np.float32)
        samples -= np.mean(samples)
        samples /= np.max(np.abs(samples)) + 1e-6

        spectrum = np.abs(np.fft.rfft(samples))
        freqs = np.fft.rfftfreq(len(samples), d=1/samplerate)

        num_segments = matrix_width
        max_freq = samplerate / 2
        segment_edges = np.linspace(0, max_freq, num_segments + 1)

        amplitudes = []
        for i in range(num_segments):
            mask = (freqs >= segment_edges[i]) & (freqs < segment_edges[i + 1])
            amp = np.sum(spectrum[mask])
            amplitudes.append(amp)

        # mild bass boost
        amplitudes[0] *= 0.5
        max_amp = max(amplitudes) or 1
        normalized = [a / max_amp for a in amplitudes]

        display_amplitudes(normalized)
        time.sleep(0.05)

except KeyboardInterrupt:
    process.terminate()
    print("stopped.")
