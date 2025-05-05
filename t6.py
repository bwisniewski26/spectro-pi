import numpy as np
import subprocess
import board
import neopixel
import time
import sys
import os

# Matrix config
matrix_width = 32
matrix_height = 8
num_pixels = matrix_width * matrix_height
brightness = 0.3  # Increased brightness
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

def find_audio_device():
    """Find available audio input devices"""
    try:
        # List audio devices and get their details
        result = subprocess.run(['arecord', '-l'], capture_output=True, text=True)
        print("Available audio devices:")
        print(result.stdout)

        # Check if any capture devices are found
        if "card" not in result.stdout:
            print("No audio capture devices found!")
            return None

        return True
    except Exception as e:
        print(f"Error finding audio devices: {e}")
        return None

def test_audio_capture(device="plughw:1,0"):
    """Test if audio can be captured from the specified device"""
    cmd = [
        "arecord",
        "-D", device,
        "-d", "1",  # Record for 1 second
        "-f", "S32_LE",
        "-c", str(channels),
        "-r", str(samplerate),
        "/dev/null"
    ]

    try:
        print(f"Testing audio capture on device {device}...")
        process = subprocess.run(cmd, capture_output=True, text=True)
        if process.returncode == 0:
            print("Audio capture test successful!")
            return True
        else:
            print(f"Audio capture test failed: {process.stderr}")
            return False
    except Exception as e:
        print(f"Error testing audio capture: {e}")
        return False

def run_visualizer(device="plughw:1,0", debug=False):
    """Run the audio visualizer"""
    # Start audio capture
    cmd = [
        "arecord",
        "-D", device,
        "-f", "S32_LE",
        "-c", str(channels),
        "-r", str(samplerate),
        "-t", "raw"
    ]

    try:
        print(f"Starting audio capture from device {device}...")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=frames_per_chunk * bytes_per_sample)

        # Initialize variables for smoothing
        prev_amplitudes = [0] * matrix_width
        smoothing_factor = 0.6  # Slightly reduced for faster response

        # Variables for debug info
        last_debug_time = time.time()
        frame_count = 0
        has_displayed_data = False

        print("Audio visualizer started. Press Ctrl+C to exit.")

        while True:
            # Non-blocking read to check stderr for errors
            stderr_data = process.stderr.read(1024) if process.stderr.readable() and process.poll() is None else None
            if stderr_data:
                print(f"arecord error: {stderr_data.decode().strip()}")

            # Read audio data
            raw_audio = process.stdout.read(frames_per_chunk * bytes_per_sample)
            if not raw_audio:
                print("No audio data received. Make sure audio input is working.")
                time.sleep(1)
                continue

            # Process audio
            samples = np.frombuffer(raw_audio, dtype=np.int32).astype(np.float32)

            # Debug info
            if debug and (time.time() - last_debug_time > 2.0):
                print(f"Audio frame stats: min={np.min(samples):.2f}, max={np.max(samples):.2f}, mean={np.mean(samples):.2f}")
                last_debug_time = time.time()
                frame_count = 0

            frame_count += 1

            # Skip processing if audio is too quiet
            if np.max(np.abs(samples)) < 100:
                if debug and not has_displayed_data:
                    print("Audio input seems very quiet. Check your microphone or audio source.")
                # Still show a minimal visualization to indicate it's working
                clear_matrix()
                # Display a single pixel at the bottom to show it's alive
                idx = (matrix_height - 1) * matrix_width + 0
                pixels[idx] = (0, 0, 64)  # Dim blue
                pixels.show()
                time.sleep(0.1)
                continue

            has_displayed_data = True

            # Center the samples around zero
            samples -= np.mean(samples)
            max_abs = np.max(np.abs(samples)) + 1e-9
            samples /= max_abs

            # Perform FFT
            spectrum = np.abs(np.fft.rfft(samples))
            freqs = np.fft.rfftfreq(len(samples), d=1/samplerate)

            # Create frequency bins with logarithmic scaling
            segment_edges = np.logspace(np.log10(20), np.log10(5000), matrix_width + 1)
            amplitudes = []
            for i in range(matrix_width):
                mask = (freqs >= segment_edges[i]) & (freqs < segment_edges[i+1])
                value = np.sum(spectrum[mask])
                amplitudes.append(value)

            # Apply frequency boosts
            amplitudes[0] *= 6.0   # Bass boost
            amplitudes[1] *= 4.0
            amplitudes[2] *= 3.0

            # Apply smoothing
            for i in range(len(amplitudes)):
                amplitudes[i] = smoothing_factor * prev_amplitudes[i] + (1 - smoothing_factor) * amplitudes[i]
            prev_amplitudes = amplitudes.copy()

            # Apply gain and clamp
            gain = 0.05  # Increased gain significantly
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
        print(f"Error in visualizer: {e}")
    finally:
        # Clean up
        if 'process' in locals():
            process.terminate()
        clear_matrix()
        print("Matrix cleared")

def list_available_devices():
    """List all available audio input devices"""
    print("\n--- Available Audio Devices ---")
    # Try arecord -l (Linux)
    try:
        subprocess.run(['arecord', '-l'], check=False)
    except:
        pass

    print("\n--- Audio Device Details ---")
    # Try arecord -L (Linux)
    try:
        subprocess.run(['arecord', '-L'], check=False)
    except:
        pass

if __name__ == "__main__":
    # Check for command line arguments
    test_mode = "--test" in sys.argv
    debug_mode = "--debug" in sys.argv
    device_mode = "--list-devices" in sys.argv

    # Get audio device from arguments if provided
    audio_device = "plughw:1,0"  # Default device
    for arg in sys.argv:
        if arg.startswith("--device="):
            audio_device = arg.split("=")[1]

    try:
        if device_mode:
            list_available_devices()
        elif test_mode:
            print("Running test pattern...")
            test_pattern()
            print("Test completed.")
        else:
            # Test audio before starting visualizer
            find_audio_device()
            if test_audio_capture(audio_device):
                run_visualizer(audio_device, debug_mode)
            else:
                print("Audio test failed. Try running with --list-devices to see available audio devices")
                print("Then run with --device=YOUR_DEVICE to specify a different audio device.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        clear_matrix()
