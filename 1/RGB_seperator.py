import cv2
import numpy as np

def rgb_extractor(image_path):
    '''
    argument:   image path
    return : 3 - 2d arrays. R array, G array, B array
    '''
    # try:
        # img = cv2.imread(image_path)
    # except FileNotFoundError:
        # print(f'No image in the path: {image_path}')
    img = cv2.imread(image_path)
    
    blue = img[:,:,0]
    green = img[:,:,1]
    red = img[:,:,2]
    
    return red,green,blue

image_path = "D:\\python\\CNN\\1_Foundational Math & Image Basics\\CNN\\datas\\test image.jpg"
 
channels = rgb_extractor(image_path)
red,green,blue = channels
# try:
    # cv2.imwrite('\\datas\\red.jpg',red)
    # cv2.imwrite('\\datas\\green.jpg',green)
    # cv2.imwrite('\\datas\\blue.jpg',blue)
# except FileNotFoundError:
        # print(f'No image in the path: {image_path}')

cv2.imwrite('D:\\python\\CNN\\1_Foundational Math & Image Basics\\CNN\\datas\\red.jpg',red)
cv2.imwrite('D:\\python\\CNN\\1_Foundational Math & Image Basics\\CNN\\datas\\green.jpg',green)
cv2.imwrite('D:\\python\\CNN\\1_Foundational Math & Image Basics\\CNN\\datas\\blue.jpg',blue)
        