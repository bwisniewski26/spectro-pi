#!/bin/python3

import board
import neopixel
import time
pixels = neopixel.NeoPixel(board.D18, 256, brightness = 0.05)
while True:
	pixels[0] = (255, 0, 0)

	pixels.show()
#	time.sleep(1)

print("hello")
