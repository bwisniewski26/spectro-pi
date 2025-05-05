import numpy as np
import subprocess
import board
import neopixel
import time
import sys

# Matrix config
matrix_width = 32
matrix_height = 8
num_pixels = matrix_width * matrix_height
brightness = 0.2  # Increased slightly
pixel_pin = board.D12

# Initialize the matrix with direct indexing (no serpentine layout)
pixels = neopixel.NeoPixel(
    pixel_pin,
    num_pixels,
    brightness=brightness,
    auto_write=False,
    pixel_order=neopixel.GRB
)

# Audio processing config
samplerate = 16000
frames_per_chunk = 1024
bytes_per_sample = 4
channels = 1

def clear_matrix():
    """Clear the entire matrix"""
    pixels.fill((0, 0, 0))
    pixels.show()

def test_pattern():
    """Run a simple test pattern to verify matrix works"""
    # Color wipe
    for color in [(255, 0, 0), (0, 255, 0), (0, 0, 255)]:
        for i in range(num_pixels):
            pixels[i] = color
            pixels.show()
            time.sleep(0.01)

    # Clear
    clear_matrix()
    time.sleep(0.5)

    # Column test
    for x in range(matrix_width):
        clear_matrix()
        for y in range(matrix_height):
            # Direct linear mapping - no serpentine
            idx = y * matrix_width + x
            pixels[idx] = (255, 255, 0)
        pixels.show()
        time.sleep(0.05)

    clear_matrix()

def run_visualizer():
    """Run the audio visualizer"""
    # Start audio capture
    cmd = [
        "arecord",
        "-D", "plughw:1,0",
        "-f", "S32_LE",
        "-c", str(channels),
        "-r", str(samplerate),
        "-t", "raw"
    ]

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=frames_per_chunk * bytes_per_sample)

        # Initialize variables for smoothing
        prev_amplitudes = [0] * matrix_width
        smoothing_factor = 0.7

        print("Audio visualizer started. Press Ctrl+C to exit.")

        while True:
            raw_audio = process.stdout.read(frames_per_chunk * bytes_per_sample)
            if not raw_audio:
                break

            # Process audio
            samples = np.frombuffer(raw_audio, dtype=np.int32).astype(np.float32)
            samples -= np.mean(samples)
            max_abs = np.max(np.abs(samples)) + 1e-9
            samples /= max_abs

            # Perform FFT
            spectrum = np.abs(np.fft.rfft(samples))
            freqs = np.fft.rfftfreq(len(samples), d=1/samplerate)

            # Create 32 frequency bins
            segment_edges = np.logspace(np.log10(20), np.log10(5000), matrix_width + 1)
            amplitudes = []
            for i in range(matrix_width):
                mask = (freqs >= segment_edges[i]) & (freqs < segment_edges[i+1])
                value = np.sum(spectrum[mask])
                amplitudes.append(value)

            # Apply frequency boosts
            amplitudes[0] *= 5.0   # Bass boost
            amplitudes[1] *= 3.0
            amplitudes[2] *= 2.0

            # Apply smoothing
            for i in range(len(amplitudes)):
                amplitudes[i] = smoothing_factor * prev_amplitudes[i] + (1 - smoothing_factor) * amplitudes[i]
            prev_amplitudes = amplitudes.copy()

            # Apply gain and clamp
            gain = 0.02  # Increased gain
            normalized = [min(a * gain, 1.0) for a in amplitudes]

            # Clear the matrix
            pixels.fill((0, 0, 0))

            # DIRECT DISPLAY - NO SERPENTINE LOGIC
            for x in range(matrix_width):
                height = int(normalized[x] * matrix_height)
                for y in range(height):
                    y_pos = matrix_height - 1 - y  # Start from bottom

                    # Direct linear indexing (no serpentine)
                    idx = y_pos * matrix_width + x

                    # Color based on frequency
                    if x < 8:        # Low frequencies (bass) - red
                        color = (255, 0, 0)
                    elif x < 16:     # Low-mid - yellow
                        color = (255, 255, 0)
                    elif x < 24:     # Mid-high - green
                        color = (0, 255, 0)
                    else:            # High frequencies - blue
                        color = (0, 0, 255)

                    pixels[idx] = color

            # Update the display
            pixels.show()

            # Small delay
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("Program terminated by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        if 'process' in locals():
            process.terminate()
        clear_matrix()
        print("Matrix cleared")

if __name__ == "__main__":
    # Check for command line arguments
    test_mode = "--test" in sys.argv

    try:
        if test_mode:
            print("Running test pattern...")
            test_pattern()
            print("Test completed.")
        else:
            run_visualizer()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        clear_matrix()
