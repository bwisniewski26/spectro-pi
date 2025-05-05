from rpi_ws281x import PixelStrip, Color

LED_COUNT = 256
LED_PIN = 12 
LED_FREQ_HZ = 800000
LED_DMA = 10
LED_BRIGHTNESS = 10
LED_INVERT = False
LED_CHANNEL = 0

strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
strip.begin()

for i in range (strip.numPixels()):
	strip.setPixelColor(i, Color(0,255,0))
strip.show()
