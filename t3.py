import board
import neopixel
import time

pixels = neopixel.NeoPixel(board.D12, 256, brightness=0.1, auto_write=True)
pixels[0] = (10, 0, 0)  # Very dim red on first pixel
time.sleep(10)
pixels.fill((0, 0, 0))
