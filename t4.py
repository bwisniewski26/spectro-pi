import board
import neopixel
import time

# Matrix config
matrix_width = 32
matrix_height = 8
num_pixels = matrix_width * matrix_height
brightness = 0.1

# Use D12 pin that you verified works
pixel_pin = board.D12

# Initialize the matrix
pixels = neopixel.NeoPixel(
    pixel_pin,
    num_pixels,
    brightness=brightness,
    auto_write=False,
    pixel_order=neopixel.GRB
)

# Clear the matrix
def clear_matrix():
    pixels.fill((0, 0, 0))
    pixels.show()

# Simple animation test
def run_test():
    # Test 1: All red
    print("Test 1: All red")
    pixels.fill((255, 0, 0))
    pixels.show()
    time.sleep(3)

    # Test 2: All green
    print("Test 2: All green")
    pixels.fill((0, 255, 0))
    pixels.show()
    time.sleep(3)

    # Test 3: All blue
    print("Test 3: All blue")
    pixels.fill((0, 0, 255))
    pixels.show()
    time.sleep(3)

    # Test 4: Moving dot across the matrix
    print("Test 4: Moving dot")
    for i in range(num_pixels):
        pixels.fill((0, 0, 0))
        pixels[i] = (255, 255, 255)
        pixels.show()
        time.sleep(0.02)

    # Clear the matrix
    clear_matrix()

if __name__ == "__main__":
    try:
        print("Starting simple LED matrix test...")
        run_test()
        print("Test completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Always clear the matrix when exiting
        clear_matrix()
