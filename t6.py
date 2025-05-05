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
    """Test if audio can be captured from the specified device and contains actual data"""
    # First test if device exists and is accessible
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
        if process.returncode != 0:
            print(f"Audio capture test failed: {process.stderr}")
            return False

        # Now test if we actually get audio data with signal
        print("Testing for actual audio signal...")
        cmd = [
            "arecord",
            "-D", device,
            "-d", "3",  # Record for 3 seconds to get enough data
            "-f", "S32_LE",
            "-c", str(channels),
            "-r", str(samplerate),
            "-t", "raw"
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        audio_data = process.stdout.read(frames_per_chunk * bytes_per_sample * 10)  # Read several frames
        process.terminate()

        if len(audio_data) < frames_per_chunk * bytes_per_sample:
            print("No audio data received. Device might be muted or disconnected.")
            return False

        # Convert to numpy array and check if there's actual signal
        samples = np.frombuffer(audio_data, dtype=np.int32)
        signal_max = np.max(np.abs(samples))

        print(f"Maximum audio signal: {signal_max}")
        if signal_max < 1000:  # Arbitrary threshold for a very quiet signal
            print("WARNING: Audio signal is very weak. Check your microphone or input source.")
            print("The visualizer may not show much activity.")
            # Still return True but with a warning
            return True

        print("Audio capture test successful with good signal levels!")
        return True

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
        "-t", "raw",
        "--buffer-size=16384"  # Explicit buffer size to prevent underruns
    ]

    try:
        print(f"Starting audio capture from device {device}...")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=frames_per_chunk * bytes_per_sample)

        # Verify we're getting data immediately
        print("Waiting for initial audio data...")
        start_time = time.time()
        initial_data = None

        # Try for up to 5 seconds to get initial data
        while time.time() - start_time < 5 and initial_data is None:
            initial_data = process.stdout.read(frames_per_chunk * bytes_per_sample)
            if not initial_data:
                print("Waiting for audio data...")
                time.sleep(0.5)

        if not initial_data:
            print("ERROR: No audio data received after 5 seconds!")
            print("Possible issues:")
            print("1. Wrong audio device selected")
            print("2. Input source is muted")
            print("3. Permission problems with audio device")
            print("4. Audio subsystem issues")
            return

        print(f"Successfully receiving audio data! ({len(initial_data)} bytes)")

        # Put the data back by seeking (if possible) or just use it as first frame
        samples = np.frombuffer(initial_data, dtype=np.int32).astype(np.float32)
        print(f"Initial audio levels: min={np.min(samples):.2f}, max={np.max(samples):.2f}")

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
            audio_max = np.max(np.abs(samples))
            if debug and frame_count % 100 == 0:
                print(f"Current audio level: {audio_max}")

            if audio_max < 100:
                if debug and not has_displayed_data:
                    print("Audio input seems very quiet. Check your microphone or audio source.")
                    print("If you're sure audio is playing, try increasing input volume or gain.")
                # Still show a minimal visualization to indicate it's working
                clear_matrix()
                # Display pulsing pixels at the bottom row to show it's alive
                t = time.time() % 2  # 2-second cycle
                intensity = int(30 + 30 * np.sin(t * np.pi))  # Pulse between 30-60
                for i in range(0, matrix_width, 4):  # Every fourth pixel
                    idx = (matrix_height - 1) * matrix_width + i
                    pixels[idx] = (0, 0, intensity)  # Dim pulsing blue
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
    force_mode = "--force" in sys.argv  # Skip audio testing
    check_mode = "--check-audio" in sys.argv  # Only check audio

    # Get audio device from arguments if provided
    audio_device = "plughw:1,0"  # Default device
    for arg in sys.argv:
        if arg.startswith("--device="):
            audio_device = arg.split("=")[1]

    try:
        print("Audio Visualizer for LED Matrix")
        print("===============================")

        if device_mode:
            list_available_devices()
        elif test_mode:
            print("Running test pattern...")
            test_pattern()
            print("Test completed.")
        elif check_mode:
            # Only test audio and exit
            find_audio_device()
            test_audio_capture(audio_device)
            print("Audio check completed.")
        else:
            # Normal operation
            print(f"Using audio device: {audio_device}")
            print("Use --list-devices to see all available devices")
            print("Use --device=YOUR_DEVICE to select a specific device")
            print("Use --check-audio to only test audio input")
            print("Use --debug for more verbose output")
            print("Use --force to skip audio testing")

            # Show test pattern first to confirm display works
            print("\nRunning quick display test...")
            clear_matrix()
            # Just flash colors briefly
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            for color in colors:
                pixels.fill(color)
                pixels.show()
                time.sleep(0.3)
            clear_matrix()
            print("Display test completed.")

            # Test audio before starting visualizer (unless forced)
            if force_mode:
                print("\nSkipping audio tests (--force mode)")
                run_visualizer(audio_device, debug_mode)
            else:
                print("\nTesting audio input...")
                find_audio_device()
                if test_audio_capture(audio_device):
                    print("\nStarting visualizer...")
                    run_visualizer(audio_device, debug_mode)
                else:
                    print("\nAudio test failed. You can:")
                    print("1. Try running with --list-devices to see available audio devices")
                    print("2. Run with --device=YOUR_DEVICE to specify a different audio device")
                    print("3. Run with --force to skip audio testing and try anyway")
                    print("4. Check if your microphone/line-in is properly connected and not muted")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        clear_matrix()
