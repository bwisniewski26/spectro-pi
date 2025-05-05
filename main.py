import numpy as np
import subprocess
import board
import neopixel
import time

# Matrix config - CORRECTED FOR 8x32 MATRIX
matrix_width = 32
matrix_height = 8  # Changed from 32 to 8
num_pixels = matrix_width * matrix_height
brightness = 0.1
pixel_pin = board.D12
pixels = neopixel.NeoPixel(
    pixel_pin,
    num_pixels,
    brightness=brightness,
    auto_write=False,
    pixel_order=neopixel.GRB
)

# XY to index with serpentine layout
def xy_to_index(x, y):
    if y % 2 == 0:
        return y * matrix_width + x
    else:
        return y * matrix_width + (matrix_width - 1 - x)

# Alternative serpentine mapping function - try this if the first one doesn't work
def alt_xy_to_index(x, y):
    if x % 2 == 0:
        return x * matrix_height + y
    else:
        return x * matrix_height + (matrix_height - 1 - y)

# Debug function to test the matrix layout
def test_matrix_layout():
    print("Testing matrix layout...")
    pixels.fill((0, 0, 0))
    pixels.show()
    time.sleep(1)

    # Test each row
    for y in range(matrix_height):
        pixels.fill((0, 0, 0))
        for x in range(matrix_width):
            idx = xy_to_index(x, y)
            if idx < num_pixels:
                pixels[idx] = (0, 255, 0)  # Green
        pixels.show()
        time.sleep(0.5)

    # Test each column
    for x in range(matrix_width):
        pixels.fill((0, 0, 0))
        for y in range(matrix_height):
            idx = xy_to_index(x, y)
            if idx < num_pixels:
                pixels[idx] = (0, 0, 255)  # Blue
        pixels.show()
        time.sleep(0.1)

    pixels.fill((0, 0, 0))
    pixels.show()

def display_amplitudes(amplitudes):
    pixels.fill((0, 0, 0))  # Clear matrix

    # Since we have only 8 rows but 32 frequency bins,
    # we need to compress the frequency data
    # Option 1: Take every 4th value (32 / 8 = 4)
    # compressed_amplitudes = [amplitudes[i] for i in range(0, len(amplitudes), 4)]

    # Option 2: Group data into 8 sections
    compressed_amplitudes = []
    section_size = len(amplitudes) // matrix_height
    for i in range(matrix_height):
        start_idx = i * section_size
        end_idx = start_idx + section_size
        section_avg = sum(amplitudes[start_idx:end_idx]) / section_size
        compressed_amplitudes.append(section_avg)

    # Display as horizontal bars (rotated visualization)
    for y in range(matrix_height):
        bar_width = int(compressed_amplitudes[y] * matrix_width)
        for x in range(bar_width):
            idx = xy_to_index(x, y)
            if idx < num_pixels:
                # Create a rainbow effect based on amplitude
                hue = (y / matrix_height) * 0.8  # Value between 0 and 0.8
                r, g, b = hsv_to_rgb(hue, 1.0, 1.0)
                pixels[idx] = (int(r * 255), int(g * 255), int(b * 255))

    pixels.show()

def vertical_display_amplitudes(amplitudes):
    pixels.fill((0, 0, 0))  # Clear matrix

    # For vertical bars, we use the full frequency range
    # but we need to fit it into our 32 columns
    scaling_factor = len(amplitudes) / matrix_width

    for x in range(matrix_width):
        # Calculate which frequency bin this column represents
        freq_idx = int(x * scaling_factor)
        if freq_idx < len(amplitudes):
            # Calculate height of the bar
            col_height = int(amplitudes[freq_idx] * matrix_height)
            for y in range(col_height):
                y_pos = matrix_height - 1 - y  # Start from bottom
                idx = xy_to_index(x, y_pos)
                if idx < num_pixels:
                    # Create color based on frequency (column position)
                    hue = (x / matrix_width) * 0.8  # Value between 0 and 0.8
                    r, g, b = hsv_to_rgb(hue, 1.0, 1.0)
                    pixels[idx] = (int(r * 255), int(g * 255), int(b * 255))

    pixels.show()

# HSV to RGB conversion for colorful visualization
def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v

    i = int(h * 6)
    f = (h * 6) - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    i %= 6

    if i == 0:
        return v, t, p
    elif i == 1:
        return q, v, p
    elif i == 2:
        return p, v, t
    elif i == 3:
        return p, q, v
    elif i == 4:
        return t, p, v
    else:
        return v, p, q

# Audio processing config
samplerate = 16000
frames_per_chunk = 1024
bytes_per_sample = 4
channels = 1

# Uncomment the next line to test the matrix layout without audio
# test_matrix_layout()

# Start audio capture
cmd = [
    "arecord",
    "-D", "plughw:1,0",  # Change if needed
    "-f", "S32_LE",
    "-c", str(channels),
    "-r", str(samplerate),
    "-t", "raw"
]

try:
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=frames_per_chunk * bytes_per_sample)

    # Improve visualization by adding smoothing
    prev_amplitudes = [0] * 32
    smoothing_factor = 0.7  # Higher values = more smoothing

    while True:
        raw_audio = process.stdout.read(frames_per_chunk * bytes_per_sample)
        if not raw_audio:
            break

        # Convert audio to float32 normalized
        samples = np.frombuffer(raw_audio, dtype=np.int32).astype(np.float32)
        samples -= np.mean(samples)
        max_abs = np.max(np.abs(samples)) + 1e-9
        samples /= max_abs

        # Perform FFT
        spectrum = np.abs(np.fft.rfft(samples))
        freqs = np.fft.rfftfreq(len(samples), d=1/samplerate)

        # Split into matrix_width bins
        segment_edges = np.linspace(20, 5000, 33)  # Focus on audible range
        amplitudes = []
        for i in range(32):
            mask = (freqs >= segment_edges[i]) & (freqs < segment_edges[i+1])
            value = np.sum(spectrum[mask])
            amplitudes.append(value)

        # Apply frequency-dependent boosting
        # Bass boost
        amplitudes[0] *= 4.0
        amplitudes[1] *= 2.5
        amplitudes[2] *= 1.5

        # Mid boost
        for i in range(3, 10):
            amplitudes[i] *= 1.2

        # Apply smoothing
        for i in range(len(amplitudes)):
            amplitudes[i] = smoothing_factor * prev_amplitudes[i] + (1 - smoothing_factor) * amplitudes[i]
        prev_amplitudes = amplitudes.copy()

        # Apply gain and clamp
        gain = 0.01  # Increased from 0.005
        normalized = [min(a * gain, 1.0) for a in amplitudes]

        # Choose which display function to use
        vertical_display_amplitudes(normalized)  # For vertical bars
        # display_amplitudes(normalized)  # For horizontal bars

        time.sleep(0.02)  # 50 fps

except KeyboardInterrupt:
    process.terminate()
    pixels.fill((0, 0, 0))
    pixels.show()
    print("Program terminated by user")
except Exception as e:
    if 'process' in locals():
        process.terminate()
    pixels.fill((0, 0, 0))
    pixels.show()
    print(f"Error: {e}")
