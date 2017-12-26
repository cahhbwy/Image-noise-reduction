# coding:utf-8
import sys
from PIL import Image
import numpy as np
import pybm3d
import cv2

noisy_img = np.array(Image.open(sys.argv[1]))

Image.fromarray(pybm3d.bm3d.bm3d(noisy_img, int(sys.argv[2]))).save("test/output_bm3d.jpg")
Image.fromarray(cv2.blur(noisy_img, (5, 5))).save("test/output_blur.jpg")
Image.fromarray(cv2.GaussianBlur(noisy_img, (5, 5), 0.8)).save("test/output_gaussian.jpg")
