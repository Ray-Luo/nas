'''
resize image to 35cm
'''
if 0:
    import cv2

    # Load the image
    image = cv2.imread('/home/luoleyouluole/nas/data_collection/10cm.png')

    # Specify the new dimensions (width, height)
    new_width = 246
    new_height = 502

    # Resize the image using bilinear interpolation
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # If you want to save the resized image
    cv2.imwrite('./resized_image.png', resized_image)

if 0:
    import cv2
    import numpy as np

    # Load the two images you want to register
    image1 = cv2.imread('./resized_image.png', cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread('./35cm.png', cv2.IMREAD_GRAYSCALE)

    image = cv2.imread('/home/luoleyouluole/nas/data_collection/resized_image.png')

    # Initialize the ORB (Oriented FAST and Rotated BRIEF) detector
    orb = cv2.ORB_create()

    # Find the keypoints and descriptors with ORB
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

    # Create a Brute Force Matcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Sort them in ascending order of distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract the matched keypoints
    src_points = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find the homography matrix
    H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    # Warp image1 onto image2
    result = cv2.warpPerspective(image, H, (image2.shape[1], image2.shape[0]))

    cv2.imwrite('./matched_image.png', result)


if 0:
    # Camera sensor's focal length in millimeters
    focal_length = 2.080  # Example focal length

    # Aperture size (f-number)
    f_number = 2.20  # Example f-number

    sensor_size = 3.646 # camera sensor's size (diagonal) in millimeters

    # Circle of confusion (1/1000th of sensor size)
    circle_of_confusion = 0.001 * sensor_size  # Adjust based on your sensor size

    # Calculate near and far limits
    near_limit = (focal_length**2) / (f_number * circle_of_confusion)
    far_limit = 2 * near_limit

    # Calculate the best distance
    best_distance = (near_limit + far_limit) / 2

    # Print the results
    print("Near Limit (N):", near_limit, "millimeters")
    print("Far Limit (F):", far_limit, "millimeters")
    print("Best Distance (D):", best_distance, "millimeters")


if 1:
    import OpenEXR
    import Imath
    import numpy as np

    # Open the input .exr file
    input_file = '/home/luoleyouluole/nas/data_collection/Route_66_Museum.exr'
    input_exr = OpenEXR.InputFile(input_file)

    # Get the original image's header to obtain metadata
    original_header = input_exr.header()
    original_width = int(original_header['dataWindow'].max.x - original_header['dataWindow'].min.x + 1)
    original_height = int(original_header['dataWindow'].max.y - original_header['dataWindow'].min.y + 1)
    channels = original_header['channels']
    data_format = original_header['channels']['R'].type

    # Crop parameters
    crop_x = 0  # Starting x-coordinate of the crop
    crop_y = 0  # Starting y-coordinate of the crop
    crop_width = 2328  # Width of the crop
    crop_height = 1748  # Height of the crop

    # Read the image data
    image_data = [input_exr.channel(channel_name, data_format) for channel_name in channels]

    # Crop the image data
    cropped_data = {}
    for i, channel_name in enumerate(channels):
        # Convert bytes to a NumPy array and then apply slicing
        channel_data = np.frombuffer(image_data[i], dtype=np.float32)
        channel_data = channel_data.reshape(original_height, original_width)
        cropped_channel = channel_data[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
        cropped_data[channel_name] = cropped_channel.tobytes()

    # Create a new header for the cropped image
    header = OpenEXR.Header(crop_width, crop_height)
    header['channels'] = original_header['channels']  # Copy channels metadata

    # Create a new .exr file for the cropped image
    output_file = 'Route_66_cropped_image.exr'
    output_exr = OpenEXR.OutputFile(output_file, header)

    # Write the cropped image data to the new file
    output_exr.writePixels(cropped_data)

    # Close the files
    input_exr.close()
    output_exr.close()

    print(f'Cropped image saved as {output_file}')


if 0:
    import cv2
    from process_hdr import save_exr
    import numpy as np

    # Load the image
    image = cv2.imread("/home/luoleyouluole/nas/data_collection/Route_66_Museum_raw_GT.hdr", -1).astype(np.float32)
    image /= 4000.0

    print(np.max(image), np.min(image), np.mean(image), "**********")

    save_exr(image, "./", "Route_66_Museum_raw_GT.exr")

    # If you want to save the resized image
    # cv2.imwrite('./resized_image.png', image)

if 0:
    import cv2
    import numpy as np
    import os
    from process_hdr import save_exr
    os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

    input_file = '/home/luoleyouluole/nas/data_collection/rit_hdr4000/Route_66_Museum.hdr'
    # Load the image
    image = cv2.imread(input_file, -1).astype(np.float32)

    print(np.max(image), np.min(image), np.mean(image), "**********")

    save_exr(image, "./", "Route_66_Museum.exr")

    # cv2.imwrite("./test.png", image)


if 0:
    import OpenEXR
    import Imath
    import numpy as np

    # Open the input .exr file
    input_file = '/home/luoleyouluole/nas/data_collection/output_cropped_image.exr'
    input_exr = OpenEXR.InputFile(input_file)

    # Get image metadata (width, height)
    header = input_exr.header()
    width = int(header['dataWindow'].max.x - header['dataWindow'].min.x + 1)
    height = int(header['dataWindow'].max.y - header['dataWindow'].min.y + 1)

    # Read the RGB image data
    data_format = header['channels']['R'].type
    R_channel = np.frombuffer(input_exr.channel('R', data_format), dtype=np.float32)
    G_channel = np.frombuffer(input_exr.channel('G', data_format), dtype=np.float32)
    B_channel = np.frombuffer(input_exr.channel('B', data_format), dtype=np.float32)

    # Create an empty array for the luminance image
    luminance_image = np.zeros((height, width, 3), dtype=np.float32)

    # Calculate the luminance for each pixel
    luminance_image[..., 0] = 0.2126 * R_channel.reshape(height, width)
    luminance_image[..., 1] = 0.7152 * G_channel.reshape(height, width)
    luminance_image[..., 2] = 0.0722 * B_channel.reshape(height, width)

    # Create a new header for the luminance image
    output_header = OpenEXR.Header(width, height)
    output_header['channels'] = {'R': Imath.Channel(data_format), 'G': Imath.Channel(data_format), 'B': Imath.Channel(data_format)}

    # Create a new .exr file for the luminance image
    output_file = 'output_luminance_image.exr'
    output_exr = OpenEXR.OutputFile(output_file, output_header)

    # Write the luminance image data to the new file
    output_exr.writePixels({'R': luminance_image[:, :, 0], 'G': luminance_image[:, :, 1], 'B': luminance_image[:, :, 2]})

    # Close the files
    input_exr.close()
    output_exr.close()


    print(f'Luminance image saved as {output_file}')


if 0:
    import OpenEXR
    import Imath
    import numpy as np

    # Open the input .exr file
    input_file = '/home/luoleyouluole/nas/data_collection/output_cropped_image.exr'
    input_exr = OpenEXR.InputFile(input_file)

    # Get image metadata (width, height)
    header = input_exr.header()
    width = int(header['dataWindow'].max.x - header['dataWindow'].min.x + 1)
    height = int(header['dataWindow'].max.y - header['dataWindow'].min.y + 1)

    # Read the RGB image data
    data_format = header['channels']['R'].type
    R_channel = np.frombuffer(input_exr.channel('R', data_format), dtype=np.float32)
    G_channel = np.frombuffer(input_exr.channel('G', data_format), dtype=np.float32)
    B_channel = np.frombuffer(input_exr.channel('B', data_format), dtype=np.float32)

    # Calculate the Y, U, and V components
    Y_channel = 0.299 * R_channel.reshape(height, width) + 0.587 * G_channel.reshape(height, width) + 0.114 * B_channel.reshape(height, width)
    U_channel = 0.492 * (B_channel.reshape(height, width) - Y_channel)
    V_channel = 0.877 * (R_channel.reshape(height, width) - Y_channel)

    # Downsample U and V channels (by a factor of 2 in both dimensions)
    U_channel = U_channel[::2, ::2]
    V_channel = V_channel[::2, ::2]

    print(Y_channel.shape, U_channel.shape, V_channel.shape, "******")

    # Create a 3-channel YUV420 image
    YUV420_image = np.zeros((height, width, 3), dtype=np.float32)
    YUV420_image[:, :, 0] = Y_channel
    YUV420_image[::2, ::2, 1] = U_channel
    YUV420_image[::2, ::2, 2] = V_channel

    # Create a new header for the YUV420 image
    output_header = OpenEXR.Header(width, height)
    output_header['channels'] = {'Y': Imath.Channel(data_format), 'U': Imath.Channel(data_format), 'V': Imath.Channel(data_format)}

    # Create a new .exr file for the YUV420 image
    output_file = 'output_yuv420_image.exr'
    output_exr = OpenEXR.OutputFile(output_file, output_header)

    # Write the YUV420 image data to the new file
    # output_exr.writePixels({'Y': YUV420_image[:, :, 0], 'U': YUV420_image[:, :, 1], 'V': YUV420_image[:, :, 2]})
    output_exr.writePixels({'Y': Y_channel.tobytes(), 'U': U_channel.tobytes(), 'V': V_channel.tobytes()})

    # Close the files
    input_exr.close()
    output_exr.close()

    print(f'YUV420 image saved as {output_file}')
