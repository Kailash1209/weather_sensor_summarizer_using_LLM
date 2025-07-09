import Adafruit_DHT
import board
import adafruit_bmp280

# DHT11 Setup (GPIO4)
DHT_SENSOR = Adafruit_DHT.DHT11
DHT_PIN = 4

# BMP280 Setup (I2C)
i2c = board.I2C()
bmp280 = adafruit_bmp280.Adafruit_BMP280_I2C(i2c)

def read_environment():
    humidity, temperature = Adafruit_DHT.read(DHT_SENSOR, DHT_PIN)
    pressure = bmp280.pressure  # in hPa

    if humidity is not None and temperature is not None:
        return round(temperature, 1), round(humidity, 1), round(pressure, 1)
    else:
        return None
