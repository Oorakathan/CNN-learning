import cv2
import numpy as np


# below multicommented is the structure i am going to implement in this file
'''

# Placeholder functions (you've likely implemented similar ones):
def extract_red(image):
    # ... your code to extract the red channel
    pass

def extract_green(image):
    # ... your code to extract the green channel
    pass

def extract_blue(image):
    # ... your code to extract the blue channel
    pass

def cross_correlation(input_channel, kernel):
    # ... your existing cross-correlation implementation
    pass

def apply_activation(feature_map, activation):
    # ... your activation function implementation (e.g., ReLU)
    pass

def apply_pooling(feature_map, size, stride):
    # ... your pooling implementation (e.g., max pooling)
    pass

def combine_channels(red, green, blue):
    # ... your code to combine the processed channels back into an RGB-like structure
    pass



def process_rgb_channel(rgb_image, kernel, activation_function, pooling_size, pooling_stride):
    """
    Processes an RGB image through convolution, activation, and pooling for each channel.
    """
    red_channel = extract_red(rgb_image)  # Assuming you have this
    green_channel = extract_green(rgb_image) # Assuming you have this
    blue_channel = extract_blue(rgb_image)  # Assuming you have this

    # Convolution for each channel
    red_convolved = cross_correlation(red_channel, kernel)
    green_convolved = cross_correlation(green_channel, kernel)
    blue_convolved = cross_correlation(blue_channel, kernel)

    # Activation for each channel
    red_activated = apply_activation(red_convolved, activation_function)
    green_activated = apply_activation(green_convolved, activation_function)
    blue_activated = apply_activation(blue_convolved, activation_function)

    # Pooling for each channel
    red_pooled = apply_pooling(red_activated, pooling_size, pooling_stride)
    green_pooled = apply_pooling(green_activated, pooling_size, pooling_stride)
    blue_pooled = apply_pooling(blue_activated, pooling_size, pooling_stride)

    # Combine the processed channels
    output_image = combine_channels(red_pooled, green_pooled, blue_pooled) # Assuming you have this

    return output_image


# Example usage:
# my_rgb_image = ... (your RGB image data)
# my_kernel = ... (your kernel)
# my_activation = "relu"
# pooling_size = (2, 2)
# pooling_stride = (2, 2)

# processed_image = process_rgb_channel(my_rgb_image, my_kernel, my_activation, pooling_size, pooling_stride)
# print(processed_image.shape)


'''




def get_img(img_path):
    """
    Reads an image from the given path using OpenCV.

    Args:
        img_path (str): Path to the image file.

    Returns:
        numpy.ndarray: The image read by OpenCV in BGR format.  Returns None on error.
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        return img
    except Exception as e:
        print(f"Error reading image: {e}")
        return None  # Important: Return None on error

def extractor(img):
    """
    Extracts the red, green, and blue channels from an image.

    Args:
        img (numpy.ndarray): The input image (BGR format).

    Returns:
        tuple: (red_channel, green_channel, blue_channel)
    """
    blue = img[:, :, 0]
    green = img[:, :, 1]
    red = img[:, :, 2]
    return red, green, blue

def set_padding(red, green, blue, padding=0):
    """
    Applies padding to the red, green, and blue channels of an image.

    Args:
        red (numpy.ndarray): The red channel.
        green (numpy.ndarray): The green channel.
        blue (numpy.ndarray): The blue channel.
        padding (int or tuple, optional): Padding size. If int, applies to all sides.
            If tuple (top, bottom, left, right), specifies padding for each side.
            Defaults to 0 (no padding).

    Returns:
        tuple: (padded_red, padded_green, padded_blue)
    """
    if isinstance(padding, int):
        pad_top = pad_bottom = pad_left = pad_right = padding
    elif isinstance(padding, tuple) and len(padding) == 4:
        pad_top, pad_bottom, pad_left, pad_right = padding
    else:
        raise ValueError("Padding must be an integer or a tuple of 4 integers.")

    padded_r = cv2.copyMakeBorder(red, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)
    padded_g = cv2.copyMakeBorder(green, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)
    padded_b = cv2.copyMakeBorder(blue, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)
    return padded_r, padded_g, padded_b

def cross_correlation(img, stride=1, activation_function=None):
    """
    Performs 2D cross-correlation with a specified stride on an image that is
    assumed to be already padded.  The kernel is fixed.

    Args:
        img (numpy.ndarray): The input 2D grayscale image (already padded).
        stride (int or tuple, optional): Stride for the kernel.
            Defaults to 1.
        activation_function (str, optional): The name of the activation
            function ('relu', 'sigmoid', 'tanh', etc.). Defaults to None.

    Returns:
        numpy.ndarray: The output feature map after cross-correlation.
    """
    kernel = np.array([[-1, -1, -1], [-1, 8.5, -1], [-1, -1, -1]], dtype=np.float32)
    img_h, img_w = img.shape
    ker_h, ker_w = kernel.shape

    # Handle stride
    if isinstance(stride, int):
        stride_h = stride_w = stride
    elif isinstance(stride, tuple) and len(stride) == 2:
        stride_h, stride_w = stride
    else:
        raise ValueError("Stride must be an integer or a tuple of 2 integers.")

    # Calculate output dimensions correctly, handling stride
    op_h = (img_h - ker_h) // stride_h + 1
    op_w = (img_w - ker_w) // stride_w + 1
    feature_map = np.zeros((op_h, op_w), dtype=np.float32)

    for y in range(op_h):
        for x in range(op_w):
            # Calculate the starting position of the kernel in the image
            start_y = y * stride_h
            start_x = x * stride_w
            # Extract the image region
            image_region = img[start_y:start_y + ker_h, start_x:start_x + ker_w]
            correlation = np.sum(image_region * kernel)
            feature_map[y, x] = correlation

    if activation_function:
        feature_map = apply_activation(feature_map, activation_function)  # Apply activation

    return feature_map

def apply_activation(feature_map, activation_function='relu'):
    """
    Applies an activation function to the feature map.

    Args:
        feature_map (numpy.ndarray): The input feature map.
        activation_function (str, optional): The name of the activation function
            ('relu', 'sigmoid', 'tanh', etc.). Defaults to 'relu'.

    Returns:
        numpy.ndarray: The activated feature map.
    """
    if activation_function == 'relu':
        return np.maximum(0, feature_map)
    elif activation_function == 'sigmoid':
        return 1 / (1 + np.exp(-feature_map))
    elif activation_function == 'tanh':
        return np.tanh(feature_map)
    else:
        raise ValueError(
            f"Activation function '{activation_function}' not supported."
        )

def apply_pooling(feature_map, pool_size=(2, 2), stride=(2, 2), pool_type='max'):
    """
    Applies max or average pooling to the input feature map.

    Args:
        feature_map (numpy.ndarray): The input 2D feature map.
        pool_size (int or tuple, optional): The size of the pooling window (height, width).
            Defaults to (2, 2).
        stride (int or tuple, optional): The stride of the pooling window (height, width).
            Defaults to (2, 2).  Can be different from convolution stride.
        pool_type (str, optional): 'max' for max pooling, 'avg' for average pooling.
            Defaults to 'max'.

    Returns:
        numpy.ndarray: The pooled output.
    """
    img_h, img_w = feature_map.shape

    # Handle pool_size
    if isinstance(pool_size, int):
        pool_h = pool_w = pool_size
    elif isinstance(pool_size, tuple) and len(pool_size) == 2:
        pool_h, pool_w = pool_size
    else:
        raise ValueError("Pool size must be an integer or a tuple of 2 integers.")

    # Handle stride
    if isinstance(stride, int):
        stride_h = stride_w = stride
    elif isinstance(stride, tuple) and len(stride) == 2:
        stride_h, stride_w = stride
    else:
        raise ValueError("Stride must be an integer or a tuple of 2 integers.")

    # Calculate output dimensions
    output_h = (img_h - pool_h) // stride_h + 1
    output_w = (img_w - pool_w) // stride_w + 1
    output = np.zeros((output_h, output_w), dtype=np.float32)

    for y in range(output_h):
        for x in range(output_w):
            start_y = y * stride_h
            start_x = x * stride_w
            window = feature_map[start_y:start_y + pool_h, start_x:start_x + pool_w]
            if pool_type == 'max':
                output[y, x] = np.max(window)
            elif pool_type == 'avg':
                output[y, x] = np.mean(window)
            else:
                raise ValueError("Invalid pool_type. Choose 'max' or 'avg'.")
    return output

def merge_rgb_channels(red_channel, green_channel, blue_channel):
    """
    Merges three 2D arrays (red, green, blue) into a single 3D RGB image.

    Args:
        red_channel (numpy.ndarray): 2D array representing the red channel.
        green_channel (numpy.ndarray): 2D array representing the green channel.
        blue_channel (numpy.ndarray): 2D array representing the blue channel.

    Returns:
        numpy.ndarray: A 3D array representing the merged RGB image in BGR format.
    """
    # Ensure the channels have the same dimensions
    height, width = red_channel.shape
    if (green_channel.shape != (height, width) or blue_channel.shape != (height, width)):
        raise ValueError("All channels must have the same dimensions.")

    # Stack the channels in BGR order
    rgb_image = np.stack([blue_channel, green_channel, red_channel], axis=-1)
    return rgb_image



def full(img_path, save_path, stride=1, padding=0, activation_function='relu',
         pool_size=(2, 2), pool_stride=(2, 2), pool_type='max'):  # Added pool_stride
    """
    Processes an image through a simplified convolutional neural network pipeline:
    reads the image, extracts RGB channels, applies padding, performs cross-correlation,
    applies an activation function, performs pooling, and saves the result.

    Args:
        img_path (str): Path to the input image.
        save_path (str): Path to save the processed image.
        stride (int or tuple, optional): Stride for convolution. Defaults to 1.
        padding (int or tuple, optional): Padding size. Defaults to 0.
        activation_function (str, optional): Activation function. Defaults to 'relu'.
        pool_size (int or tuple, optional): Size of the pooling window. Defaults to (2,2).
        pool_stride (int or tuple, optional): Stride of the pooling window. Defaults to (2,2).
        pool_type (str, optional): Pooling type ('max' or 'avg'). Defaults to 'max'.
    """
    img = get_img(img_path)
    if img is None:
        return  # Handle the error by returning early

    red, green, blue = extractor(img)
    padded_r, padded_g, padded_b = set_padding(red, green, blue, padding)

    # Apply convolution, activation, and pooling to each channel
    convolved_r = cross_correlation(padded_r, stride, activation_function)
    convolved_g = cross_correlation(padded_g, stride, activation_function)
    convolved_b = cross_correlation(padded_b, stride, activation_function)

    pooled_r = apply_pooling(convolved_r, pool_size, pool_stride, pool_type)  # Use pool_stride
    pooled_g = apply_pooling(convolved_g, pool_size, pool_stride, pool_type)  # Use pool_stride
    pooled_b = apply_pooling(convolved_b, pool_size, pool_stride, pool_type)  # Use pool_stride

    # Merge the processed channels
    op = merge_rgb_channels(pooled_r, pooled_g, pooled_b)
    save_img(save_path, op)



def save_img(save_path, img):
    """
    Saves an image to the specified path using OpenCV.

    Args:
        save_path (str): Path to save the image.
        img (numpy.ndarray): The image to save.
    """
    try:
        cv2.imwrite(save_path, img)
        print(f"Image saved to {save_path}")
    except Exception as e:
        print(f"Error saving image: {e}")



if __name__ == '__main__':
    # Example usage:
    img_path = "D:\\python\\CNN\\1_Foundational Math & Image Basics\\CNN\\datas\\test image.jpg"
    save_path = "D:\\python\\CNN\\1_Foundational Math & Image Basics\\CNN\\datas\\full_processed_image.jpg" # Changed save_path
    stride = 1
    padding = 1
    activation_function = 'relu'
    pool_size = (2, 2)
    pool_stride = (2, 2)  # Added pool_stride
    pooling_type = 'max'

    final = full(img_path, save_path, stride, padding, activation_function, pool_size, pool_stride, pooling_type) # Added pool_stride
    if final is not None:
        print("Processing complete.")  # Only print if processing was successful
    else:
        print("Processing failed.")
