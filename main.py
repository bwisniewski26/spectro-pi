import numpy as np
import subprocess
import board
import neopixel
import time

# matrix config
matrix_width = 32
matrix_height = 32
num_pixels = matrix_width * matrix_height
brightness = 0.1
pixel_pin = board.D18  # or D12 if yours is wired there

pixels = neopixel.NeoPixel(
    pixel_pin,
    num_pixels,
    brightness=brightness,
    auto_write=False,
    pixel_order=neopixel.GRB
)

# xy to index with serpentine layout
def xy_to_index(x, y):
    if y % 2 == 0:
        return y * matrix_width + x
    else:
        return y * matrix_width + (matrix_width - 1 - x)

def display_amplitudes(amplitudes):
    pixels.fill((0, 0, 0))  # clear matrix
    for x in range(matrix_width):
        col_height = int(amplitudes[x] * matrix_height)
        for y in range(col_height):
            idx = xy_to_index(x, matrix_height - 1 - y)
            pixels[idx] = (0, 255, 0)  # green
    pixels.show()

# arecord config
samplerate = 16000
frames_per_chunk = 1024
bytes_per_sample = 4
channels = 1
cmd = [
    "arecord",
    "-D", "plughw:1,0",  # change if needed
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

        # audio to float32 normalized
        samples = np.frombuffer(raw_audio, dtype=np.int32).astype(np.float32)
        samples -= np.mean(samples)
        max_abs = np.max(np.abs(samples)) + 1e-9
        samples /= max_abs

        # fft
        spectrum = np.abs(np.fft.rfft(samples))
        freqs = np.fft.rfftfreq(len(samples), d=1/samplerate)

        # split into matrix_width bins
        segment_edges = np.linspace(0, samplerate/2, matrix_width + 1)
        amplitudes = []
        for i in range(matrix_width):
            mask = (freqs >= segment_edges[i]) & (freqs < segment_edges[i+1])
            value = np.sum(spectrum[mask])
            amplitudes.append(value)

        # bass boost
        amplitudes[0] *= 4.0
        amplitudes[1] *= 2.5
        amplitudes[2] *= 1.5

        # gain and clamp
        gain = 0.005
        normalized = [min(a * gain, 1.0) for a in amplitudes]

        display_amplitudes(normalized)
        time.sleep(0.02)

except KeyboardInterrupt:
    process.terminate()
    pixels.fill((0, 0, 0))
    pixels.show()
    print("done")
