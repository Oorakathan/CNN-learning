import numpy as np
import cv2

#without stride and padding
def cross_correlation(image,kernel):
    """
        args:   2d array of grayscale image
                2d array of the kernel matrix
        return: a 2d array of feature map
    """
    
    img_h,img_w = image.shape
    ker_h,ker_w = kernel.shape
    
    op_h = img_h - ker_h +1
    op_w = img_w - ker_w +1
    feature_map = np.zeros((op_h,op_w),dtype=np.float32)
    
    for y in range(op_h):
        for x in range(op_w):
            #get the image portion/region
            image_region = image[y:y+ker_h,x:x+ker_w]
            correlation = np.sum(image_region*kernel)
            feature_map[y,x] = correlation
    return feature_map

image_path = "D:\\python\\CNN\\1_Foundational Math & Image Basics\\CNN\\datas\\grayscaled_image.jpg"

save_path = "D:\\python\\CNN\\1_Foundational Math & Image Basics\\CNN\\datas\\cross_correlated_image.jpg"


kernel = np.array([[-1,-1,-1],[-1,8.5,-1],[-1,-1,-1]],dtype=np.float32)
try:
    Gimage = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    
    correlated_img = cross_correlation(Gimage,kernel)

    
    cv2.imwrite(save_path,correlated_img)
    print(f"correlated_img saved to {save_path}")
    
except FileNotFoundError:
    print(f"File {image_path} is not found")
            