import os
import cv2
import numpy as np
from PIL import Image

def add_gaussian_noise(image_in, noise_sigma):
    temp_image = np.float64(np.copy(image_in))
    h, w, _ = temp_image.shape
    noise = np.random.randn(h, w) * noise_sigma
    noisy_image = np.zeros(temp_image.shape, np.float64)
    if len(image_in.shape) == 2:
        noisy_image = temp_image + noise
    else:
        noisy_image[:,:,0] = temp_image[:,:,0] + noise
        noisy_image[:,:,1] = temp_image[:,:,1] + noise
        noisy_image[:,:,2] = temp_image[:,:,2] + noise
    return cv2.convertScaleAbs(noisy_image)

def compress_image(image, jpeg_quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    if result:
        decimg = cv2.imdecode(encimg, 1)
        return decimg
    return None

def downsample_image(image, factor):
    return cv2.resize(image, (image.shape[1]//factor, image.shape[0]//factor))

def upsample_image(image, factor):
    return cv2.resize(image, (image.shape[1]*factor, image.shape[0]*factor))

def process_images(input_folder, output_folder, blur_sigma, noise_sigma, jpeg_quality):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for img_name in os.listdir(input_folder):
        if img_name.endswith('.jpg'):
            img_path = os.path.join(input_folder, img_name)
            img = cv2.imread(img_path)

            # img = cv2.GaussianBlur(img, (blur_sigma, blur_sigma), 0)
            # img = add_gaussian_noise(img, noise_sigma)
            img = compress_image(img, jpeg_quality)
            # img = downsample_image(img, 2)
            # img = upsample_image(img, 2)

            output_path = os.path.join(output_folder, img_name)
            cv2.imwrite(output_path, img)

path_to_input_folder = "./input"
path_to_output_folder = "./jc"
process_images(path_to_input_folder, path_to_output_folder, 5, 10, 60)
