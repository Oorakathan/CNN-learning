import numpy as np
import cv2

Gimg = cv2.imread("D:\\python\\CNN\\1_Foundational Math & Image Basics\\CNN\\datas\\grayscaled_image.jpg",cv2.IMREAD_GRAYSCALE)
Aimg = np.array(Gimg)
print(Aimg)
print(Aimg.shape)

Cimg = cv2.imread("D:\\python\\CNN\\1_Foundational Math & Image Basics\\CNN\\datas\\test image.jpg")
Aimg = np.array(Cimg)
print(Aimg)
print(Aimg.shape)