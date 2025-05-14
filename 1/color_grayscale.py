import cv2

def color_to_grayscale(input_path,output_path):
	try:
		image = cv2.imread(input_path)
		grayscaled_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		cv2.imwrite(output_path,grayscaled_image)
		print(f"grayscale is saved {output_path}")
	except FileNotFoundError:
		print(f"file not found, try again")
	

input_path = "D:\\python\\CNN\\1_Foundational Math & Image Basics\\CNN\\datas\\test image.jpg"
output_path = "D:\\python\\CNN\\1_Foundational Math & Image Basics\\CNN\\datas\\grayscaled_image.jpg"
color_to_grayscale(input_path,output_path)