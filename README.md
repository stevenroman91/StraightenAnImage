# Image Deskewing Utility
This Python script provides a set of functions to perform image deskewing and rotation with aspect ratio preservation.

# Functions
rotate_w_proportinons(image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]) -> np.ndarray:
This function takes an input image, an angle of rotation, and a background color (either an integer or a tuple of RGB values).
It calculates the new dimensions of the rotated image, taking into account the aspect ratio, and performs the rotation using OpenCV's cv2.warpAffine() function.
The function returns the rotated image with the new dimensions.

getSkewAngle(cvImage) -> float:
This function calculates the skew angle of the input image.
It preprocesses the image by converting it to grayscale, applying Gaussian blur, and thresholding.
It then finds the largest contour in the image and determines the angle of the minimum area rectangle surrounding the contour.
The function returns the negative of the calculated angle, as it is meant to be used for deskewing the image.

deskew(cvImage):
This function takes an input image and deskews it by calling the getSkewAngle() function to determine the skew angle, and then calling the rotate_w_proportinons() function to rotate the image by the negative of the calculated angle.
The function returns the deskewed image.
Usage
To use these functions, you can import them into your Python script and call them with the appropriate arguments:

# Your code here
from PIL import Image

import pytesseract

import numpy as np

from typing import Tuple, Union

import math

import cv2

image = cv2.imread('image.jpg')
deskewed_image = deskew(image)
cv2.imwrite('deskewed_image.jpg', deskewed_image)


Make sure you have the required dependencies installed, such as Pillow, pytesseract, and OpenCV.

# Dependencies
Python 3.x

Pillow (PIL)

pytesseract

NumPy

OpenCV
