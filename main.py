import numpy as np
import subprocess
import board
import neopixel
import time

# audio input config
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

# neopixel init (serpentine layout assumed, GRB order common for WS2812)
pixels = neopixel.NeoPixel(
    pixel_pin,
    num_pixels,
    brightness=brightness,
    auto_write=False,
    pixel_order=neopixel.GRB
)

# serpentine mapping fn
def xy_to_index(x, y):
    if y % 2 == 0:
        return y * matrix_width + x
    else:
        return y * matrix_width + (matrix_width - 1 - x)

# draw vertical bars based on amplitudes (len == matrix_width)
def display_amplitudes(amplitudes):
    pixels.fill((0, 0, 0))  # clear
    for x, amp in enumerate(amplitudes):
        height = int(amp * matrix_height)
        for y in range(height):
            idx = xy_to_index(x, matrix_height - 1 - y)
            if idx < num_pixels:
                pixels[idx] = (0, 255, 0)  # green bar
    pixels.show()

# start audio capture process
cmd = [
    "arecord",
    "-D", "plughw:1,0",  # adjust if needed
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

        # unpack and normalize
        samples = np.frombuffer(raw_audio, dtype=np.int32).astype(np.float32)
        samples -= np.mean(samples)
        max_abs = np.max(np.abs(samples)) + 1e-6
        samples /= max_abs

        # compute fft
        spectrum = np.abs(np.fft.rfft(samples))
        freqs = np.fft.rfftfreq(len(samples), d=1/samplerate)

        # split into columns
        num_segments = matrix_width
        max_freq = samplerate / 2
        segment_edges = np.linspace(0, max_freq, num_segments + 1)

        amplitudes = []
        for i in range(num_segments):
            mask = (freqs >= segment_edges[i]) & (freqs < segment_edges[i + 1])
            val = np.sum(spectrum[mask])
            amplitudes.append(val)

        # bass boost
        amplitudes[0] *= 3.0
        amplitudes[1] *= 2.0

        # static gain factor
        gain = 0.005
        normalized = [min(a * gain, 1.0) for a in amplitudes]

        display_amplitudes(normalized)
        time.sleep(0.05)

except KeyboardInterrupt:
    process.terminate()
    print("bye")
