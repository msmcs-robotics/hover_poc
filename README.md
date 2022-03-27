# Hover_PoC

> A proof of concept robot

## Stabilizer

> stabilization

An arduino is used to implement simple algorithms to keep the drone upright.
This process utilizes an MPU6050 to determine changes in the drone relative orientation.

**References**

- [dRehmFlight](https://github.com/nickrehm/dRehmFlight)

## Caput
> flight path

A raspberry pi is used to dictate instructions to the arduino to in turn allocate more or less powers to ESCs specified in said instructions.
This process uses the YOLO algorithm to identify objects as targets or obstacles.
This process then utilizes I2C for the raspberry pi to communicate with the arduino.

References:

- [Facial Detection 1](https://github.com/yeephycho/tensorflow-face-detection)
- [Facial Detection 2](https://github.com/parulnith/Face-Detection-in-Python-using-OpenCV)
- [Object Tracking](https://iot4beginners.com/object-tracking-camera-using-raspberry-pi-and-opencv/)
- [I2C Communication](https://radiostud.io/howto-i2c-communication-rpi/)