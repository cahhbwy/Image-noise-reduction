# coding:utf-8
import sys
from PIL import Image
import numpy as np
import pybm3d
import cv2
import time

noisy_img = np.array(Image.open("test/noisy_image.jpg"))

start_time = time.time()
pybm3d.bm3d.bm3d(noisy_img, 16)
stop_time = time.time()
print("time_cost: %f" % (stop_time - start_time))

start_time = time.time()
cv2.blur(noisy_img, (5, 5))
stop_time = time.time()
print("time_cost: %f" % (stop_time - start_time))

start_time = time.time()
cv2.GaussianBlur(noisy_img, (5, 5), 0.8)
stop_time = time.time()
print("time_cost: %f" % (stop_time - start_time))

Image.fromarray(pybm3d.bm3d.bm3d(noisy_img, int(sys.argv[1]))).save("test/output_bm3d.jpg")
Image.fromarray(cv2.blur(noisy_img, (5, 5))).save("test/output_blur.jpg")
Image.fromarray(cv2.GaussianBlur(noisy_img, (5, 5), 0.8)).save("test/output_gaussian.jpg")
