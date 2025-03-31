# SpectroPi

Device based on Raspberry Pi Zero 2W designed for 'Projektowanie i Produkcja' course at Nicolaus Copernicus University.

Its purpose is to visualize audio spectrum gathered through built in microphones on a 32x8 RGB LED matrix. 


### Requirements

- Raspberry Pi (tested only on model 2W)
- RaspberryPiOS
- Python 3

### Parts used

- Raspberry Pi Zero 2W
- SPH0645LM4H - MEMS I2S microphone
- 32x8 RGB WS2812B LED matrix 


### Script dependencies

- Board - Common container for board base pin names. These will vary from board to board so don’t expect portability when using this module.
- NeoPixel_SPI - Library used to control WS2812B matrix through SPI bus
- PyAudio & NumPy - libraries used for mathematical operations on audio input