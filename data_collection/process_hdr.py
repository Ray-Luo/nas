import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from types import SimpleNamespace
import OpenEXR, Imath
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

SEED = 12345
NUMBER_AUG = 1
FOCUS_ORDER = 3

cfg = {
    "enable_dpc": False,
    "bayer_binning": False,
    "NOISE_MODE": "VAR", #option: CDF, VAR
    "CROP_HEIGHT": 512,
    "CROP_WIDTH": 512,
    "RAW_HEIGHT": 3024,
    "RAW_WIDTH" : 4038,
    "RAW_BITS" : 10,
    "meta_padding": 8076,
    "AUG_WHITE_LEVEL": 65535,
    "AUG_BLACK_LEVEL": 4096,
    "BLACK_LEVEL": 256,
    "WHITE_LEVEL": 4095,
    "BLACK_OFFSET": 16,
    "BAYER_PATTERN": "GRBG",
    "NOISE_ORDER": 1.0
}
cfg = SimpleNamespace(**cfg)

class RandomNoiseAdder:
    def __init__(
        self,
        iso_range_start,
        iso_range_end,
        min_gain,
        max_gain,
        shot_noise_slope,
        shot_noise_intercept,
        shot_noise_stderr,
        read_noise_slope,
        read_noise_intercept,
        read_noise_stderr,
    ) -> None:
        self.shot_noise_slope = torch.tensor(shot_noise_slope)
        self.shot_noise_intercept = torch.tensor(shot_noise_intercept)
        self.shot_noise_stderr = torch.tensor(shot_noise_stderr)
        self.read_noise_slope = torch.tensor(read_noise_slope)
        self.read_noise_intercept = torch.tensor(read_noise_intercept)
        self.read_noise_stderr = torch.tensor(read_noise_stderr)
        min_gain_subrange = min_gain + iso_range_start * (max_gain - min_gain)
        max_gain_subrange = min_gain + iso_range_end * (max_gain - min_gain)
        self.uniform = torch.distributions.uniform.Uniform(
            min_gain_subrange,
            max_gain_subrange,
        )

    def _generate_noise(
        self,
        image,
        shot_noise,
        read_noise,
        clip_noise: bool = True,
    ):
        """Adds random shot (proportional to image) and read (independent) noise."""
        variance = image * shot_noise[:, None, None] + read_noise[:, None, None]
        stdev = torch.sqrt(variance)
        noise = torch.randn(image.shape, dtype=torch.float, device=image.device)
        noise = torch.mul(noise, stdev)

        noisy_image = noise + image
        if clip_noise:
            noisy_image = torch.clamp(noisy_image, min=0.0, max=1.0)

        return noisy_image

    def __call__(self, img):
        gain = self.uniform.sample()
        reported_shot_noise = self.shot_noise_slope * gain + self.shot_noise_intercept
        reported_read_noise = self.read_noise_slope * gain + self.read_noise_intercept
        shot_noise = reported_shot_noise + np.random.normal(
            0.0, self.shot_noise_stderr
        ).astype(np.float32)
        read_noise = reported_read_noise + np.random.normal(
            0.0, self.read_noise_stderr
        ).astype(np.float32)
        noisy_img = self._generate_noise(img, shot_noise, read_noise)

        return noisy_img, reported_shot_noise, reported_read_noise


def derivativeV(img, mode="g"):
    if mode == "all":
        derivative = (
            abs(
                np.roll(img, [-1, -1], axis=[0, 1])
                - np.roll(img, [+1, -1], axis=[0, 1])
            )
            * 2
            + abs(np.roll(img, [-2, 0], axis=[0, 1]) - img) * 2
            + abs(img - np.roll(img, [+2, 0], axis=[0, 1])) * 2
            + abs(
                np.roll(img, [-1, +1], axis=[0, 1])
                - np.roll(img, [+1, +1], axis=[0, 1])
            )
            * 2
            + abs(
                np.roll(img, [-2, -1], axis=[0, 1]) - np.roll(img, [0, -1], axis=[0, 1])
            )
            + abs(
                np.roll(img, [0, -1], axis=[0, 1]) - np.roll(img, [+2, -1], axis=[0, 1])
            )
            + abs(
                np.roll(img, [-1, 0], axis=[0, 1]) - np.roll(img, [+1, 0], axis=[0, 1])
            )
            * 4
            + abs(
                np.roll(img, [-2, +1], axis=[0, 1]) - np.roll(img, [0, +1], axis=[0, 1])
            )
            + abs(
                np.roll(img, [0, +1], axis=[0, 1]) - np.roll(img, [-2, +1], axis=[0, 1])
            )
        )
        derivative = derivative / 16
    elif mode == "g":
        derivative = (
            abs(
                np.roll(img, [-2, -1], axis=[0, 1]) - np.roll(img, [0, -1], axis=[0, 1])
            )
            + abs(
                np.roll(img, [0, -1], axis=[0, 1]) - np.roll(img, [+2, -1], axis=[0, 1])
            )
            + abs(
                np.roll(img, [-1, 0], axis=[0, 1]) - np.roll(img, [+1, 0], axis=[0, 1])
            )
            * 4
            + abs(
                np.roll(img, [-2, +1], axis=[0, 1]) - np.roll(img, [0, +1], axis=[0, 1])
            )
            + abs(
                np.roll(img, [0, +1], axis=[0, 1]) - np.roll(img, [-2, +1], axis=[0, 1])
            )
        )
        derivative = derivative / 8
    else:
        raise Exception("unexpected derivative mode")
    return derivative

def derivativeH(img, mode="g"):
    return np.transpose(derivativeV(np.transpose(img), mode))

def interpolateH(img):
    return (np.roll(img, [0, -1], axis=[0, 1]) + np.roll(img, [0, 1], axis=[0, 1])) / 2

def interpolateV(img):
    return (np.roll(img, [-1, 0], axis=[0, 1]) + np.roll(img, [1, 0], axis=[0, 1])) / 2

def interpolatePlus(img):
    return (
        np.roll(img, [-1, 0], axis=[0, 1])
        + np.roll(img, [1, 0], axis=[0, 1])
        + np.roll(img, [0, -1], axis=[0, 1])
        + np.roll(img, [0, 1], axis=[0, 1])
    ) / 4

def interpolateCross(img):
    return (
        np.roll(img, [-1, -1], axis=[0, 1])
        + np.roll(img, [+1, +1], axis=[0, 1])
        + np.roll(img, [+1, -1], axis=[0, 1])
        + np.roll(img, [-1, +1], axis=[0, 1])
    ) / 4

def bayer2RGB(bayer, thres=10, interpolateChroma=True, deriMode="all"):
    rgbImg = np.zeros((bayer.shape[0], bayer.shape[1], 3))

    rgbImg[0::2, 0::2, 1] = bayer[0::2, 0::2]  # gb
    rgbImg[0::2, 1::2, 0] = bayer[0::2, 1::2]  # b
    rgbImg[1::2, 0::2, 2] = bayer[1::2, 0::2]  # r
    rgbImg[1::2, 1::2, 1] = bayer[1::2, 1::2]  # gr

    # interpolate green channel
    derH = derivativeH(bayer, deriMode)
    derV = derivativeV(bayer, deriMode)
    derMax = np.maximum(derH, derV)
    weight = (np.clip(derMax - thres, -8, 0) + 8) / 8
    driction = derH > derV

    pixelH = interpolateH(rgbImg[::, ::, 1])
    pixelV = interpolateV(rgbImg[::, ::, 1])
    pixelDir = pixelH
    pixelDir[driction] = pixelV[driction]
    pixelPlus = interpolatePlus(rgbImg[::, ::, 1])
    pixelBlend = (pixelDir - pixelPlus) * weight + pixelPlus

    rgbImg[0::2, 1::2, 1] = pixelBlend[0::2, 1::2]
    rgbImg[1::2, 0::2, 1] = pixelBlend[1::2, 0::2]

    # chroma interpolation
    greenPlane = (
        rgbImg[::, ::, 1]
        if interpolateChroma
        else np.zeros((rgbImg.shape[0], rgbImg.shape[1]))
    )

    # complete red/blue cross
    rgbImg[0::2, 1::2, 2] = (
        interpolateCross(rgbImg[::, ::, 2] - greenPlane) + greenPlane
    )[0::2, 1::2]
    rgbImg[1::2, 0::2, 0] = (
        interpolateCross(rgbImg[::, ::, 0] - greenPlane) + greenPlane
    )[1::2, 0::2]

    # complete blue plus
    rgbImg[0::2, 0::2, 0] = (interpolateH(rgbImg[::, ::, 0] - greenPlane) + greenPlane)[
        0::2, 0::2
    ]
    rgbImg[1::2, 1::2, 0] = (interpolateV(rgbImg[::, ::, 0] - greenPlane) + greenPlane)[
        1::2, 1::2
    ]

    # complete red plus
    rgbImg[0::2, 0::2, 2] = (interpolateV(rgbImg[::, ::, 2] - greenPlane) + greenPlane)[
        0::2, 0::2
    ]
    rgbImg[1::2, 1::2, 2] = (interpolateH(rgbImg[::, ::, 2] - greenPlane) + greenPlane)[
        1::2, 1::2
    ]

    return rgbImg


def bayer2rgb(raw: np.ndarray) -> np.ndarray:
    """
    Create linear RGB by converting 2x2 Bayer pattern RGGB to 3-channel RGB
    """
    raw = raw.permute(1,2,0).squeeze(-1)
    height = int(raw.shape[0] / 2)
    width = int(raw.shape[1] / 2)
    # print(raw.shape, height, width)
    out = np.empty((height, width, 3), dtype=int)
    # Use slicing to generate RGB colocated image
    # R value
    out[:, :, 0] = raw[0::2, 0::2]
    # B value
    out[:, :, 2] = raw[1::2, 1::2]
    # G value
    if False:
        out[:, :, 1] = (raw[0::2, 1::2] + raw[1::2, 0::2] + 1) / 2
    else:  # use Gr only
        out[:, :, 1] = raw[0::2, 1::2]
    return out

def normalizeRaw(in_image):
    in_image = in_image.astype(np.float32)
    in_image = in_image / (
        (cfg.AUG_WHITE_LEVEL + 1) / (cfg.WHITE_LEVEL + 1)
    )
    quant_numerator = np.clip(
        in_image - cfg.BLACK_LEVEL,
        -cfg.BLACK_LEVEL / 32,
        cfg.WHITE_LEVEL,
    )
    quant_denominator = np.sqrt(
        np.clip(
            in_image - cfg.BLACK_LEVEL + cfg.BLACK_OFFSET,
            cfg.BLACK_OFFSET,
            cfg.WHITE_LEVEL,
        )
    )
    quant_result = np.divide(
        np.divide(quant_numerator, quant_denominator),
        np.sqrt(cfg.WHITE_LEVEL),
    )

    return quant_result


def noiseFromCDF(gt_bayer):
    pass

def noiseFromVariance(profile, gt_bayer):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))
    shape = gt_bayer.shape
    var = profile[gt_bayer]
    noise = var * np.random.randn(*shape)
    return noise

def addNoise(profile, gt_bayer):
    if cfg.NOISE_MODE == "VAR":
        noise = noiseFromVariance(profile, gt_bayer)
    elif cfg.NOISE_MODE == "CDF":
        noise = noiseFromCDF(gt_bayer)
    else:
        raise NotImplementedError

    noise_order = cfg.NOISE_ORDER - (
        cfg.NOISE_ORDER - 1.0
    ) * np.power(
        np.clip(
            (gt_bayer.astype(np.float32) - cfg.AUG_BLACK_LEVEL)
            / cfg.AUG_WHITE_LEVEL,
            0,
            1.0,
        ),
        0.2,
    )
    noisy_bayer = np.clip(
        noise * noise_order + gt_bayer.astype(np.float32),
        0,
        cfg.AUG_WHITE_LEVEL,
    )
    return noisy_bayer

def getRandomMultiplier():
    np.random.seed()
    dice = np.random.randint(10)
    max_multf = (
        cfg.AUG_WHITE_LEVEL - cfg.AUG_BLACK_LEVEL
    ) * 1.0 / 65535.0 - 1
    multf = max_multf + 1
    if dice == 1:
        multf = 1 + max_multf * np.random.random_sample()
    elif dice > 1:
        multf = 1 + max_multf * (np.random.random_sample() ** FOCUS_ORDER)
    return multf

def getRandomGain():
    multf = getRandomMultiplier()
    r_gain = random.uniform(1.0, 3.0)
    b_gain = random.uniform(1.0, 3.0)
    if cfg.BAYER_PATTERN == "MONO":
        r_gain = 1.0
        b_gain = 1.0
    return multf, r_gain, b_gain

def applyGain(rgb_tensor, multf, r_gain, b_gain):
    wb_gain = np.zeros((2, 2), dtype=np.float32)
    T, H, W, C = rgb_tensor.shape
    rc = np.clip(
        rgb_tensor[:, :, :, 0] * multf / r_gain + cfg.AUG_BLACK_LEVEL,
        0,
        cfg.AUG_WHITE_LEVEL,
    )
    gc = np.clip(
        rgb_tensor[:, :, :, 1] * multf + cfg.AUG_BLACK_LEVEL,
        0,
        cfg.AUG_WHITE_LEVEL,
    )
    bc = np.clip(
        rgb_tensor[:, :, :, 2] * multf / b_gain + cfg.AUG_BLACK_LEVEL,
        0,
        cfg.AUG_WHITE_LEVEL,
    )

    bayer_tensor = np.zeros((T, 1, H, W), dtype=np.int32)

    if cfg.BAYER_PATTERN == "MONO":
        bayer_tensor[:, 0, :, :] = (rc * 0.299 + gc * 0.587 + bc * 0.114).astype(
            np.uint32
        )
    elif cfg.BAYER_PATTERN == "RGGB":
        bayer_tensor[:, 0, 0:H:2, 0:W:2] = rc[:, 0:H:2, 0:W:2]
        bayer_tensor[:, 0, 0:H:2, 1:W:2] = gc[:, 0:H:2, 1:W:2]
        bayer_tensor[:, 0, 1:H:2, 0:W:2] = gc[:, 1:H:2, 0:W:2]
        bayer_tensor[:, 0, 1:H:2, 1:W:2] = bc[:, 1:H:2, 1:W:2]
        wb_gain = np.array([[r_gain, 1.0], [1.0, b_gain]])
    elif cfg.BAYER_PATTERN == "BGGR":
        bayer_tensor[:, 0, 0:H:2, 0:W:2] = bc[:, 0:H:2, 0:W:2]
        bayer_tensor[:, 0, 0:H:2, 1:W:2] = gc[:, 0:H:2, 1:W:2]
        bayer_tensor[:, 0, 1:H:2, 0:W:2] = gc[:, 1:H:2, 0:W:2]
        bayer_tensor[:, 0, 1:H:2, 1:W:2] = rc[:, 1:H:2, 1:W:2]
        wb_gain = np.array([[b_gain, 1.0], [1.0, r_gain]])
    elif cfg.BAYER_PATTERN == "GRBG":
        bayer_tensor[:, 0, 0:H:2, 0:W:2] = gc[:, 0:H:2, 0:W:2]
        bayer_tensor[:, 0, 0:H:2, 1:W:2] = rc[:, 0:H:2, 1:W:2]
        bayer_tensor[:, 0, 1:H:2, 0:W:2] = bc[:, 1:H:2, 0:W:2]
        bayer_tensor[:, 0, 1:H:2, 1:W:2] = gc[:, 1:H:2, 1:W:2]
        wb_gain = np.array([[1.0, r_gain], [b_gain, 1.0]])
    elif cfg.BAYER_PATTERN == "GBRG":
        bayer_tensor[:, 0, 0:H:2, 0:W:2] = gc[:, 0:H:2, 0:W:2]
        bayer_tensor[:, 0, 0:H:2, 1:W:2] = bc[:, 0:H:2, 1:W:2]
        bayer_tensor[:, 0, 1:H:2, 0:W:2] = rc[:, 1:H:2, 0:W:2]
        bayer_tensor[:, 0, 1:H:2, 1:W:2] = gc[:, 1:H:2, 1:W:2]
        wb_gain = np.array([[1.0, b_gain], [r_gain, 1.0]])
    else:
        raise NotImplementedError

    return bayer_tensor, wb_gain

def rgb2bayer(rgb_tensor):
    multf, r_gain, b_gain = getRandomGain()
    return applyGain(rgb_tensor, multf, r_gain, b_gain)

def print_min_max(path: str, name: str):
    # Read the image using OpenCV
    img = cv2.imread(path, -1).astype(np.float32)

    # Get the minimum and maximum pixel values
    min_val = np.min(img)
    max_val = np.max(img)

    # Print the results
    print(min_val, "  ", max_val, "  ", img.shape, "  ", name)


def print_min_max(img: np.array):

    # Get the minimum and maximum pixel values
    min_val = np.min(img)
    max_val = np.max(img)
    mean_val = np.mean(img)
    std_val = np.std(img)

    # Print the results
    print("min: {}, max: {}, mean: {}, std: {}, shape: {}".format(min_val, max_val, mean_val, std_val, img.shape))


def save_hdr(img: np.array, img_folder: str, name: str):
    # print(os.path.join(img_folder, name))
    cv2.imwrite(os.path.join(img_folder, name), img, [cv2.IMWRITE_HDR_COMPRESSION, 0])


def augment(img_folder: str, name: str):
    img = cv2.imread(os.path.join(img_folder, name), -1).astype(np.float32)

    seen = set()
    random.seed(SEED)
    rand_num = random.uniform(0.1, 0.9)

    while len(seen) < NUMBER_AUG:
        if rand_num not in seen:
            seen.add(rand_num)
            img *= rand_num
            processed_img = process_rit(img)
            new_name = name.split('.')[0] + '_aug_' + str(len(seen)) + '.hdr'
            save_hdr(processed_img, img_folder, new_name)
            rand_num = random.uniform(0.1, 0.9)


def process_rit(img: np.array):
    print_min_max(img)
    img_g = img[:,:,1]
    value_at_99_percentile = np.percentile(img_g, 99)
    img /= value_at_99_percentile
    img *= 4000.0
    img = np.clip(img, 0.05, 4000.0)

    print_min_max(img)
    print("**********")

    return img


def process_save(img_folder: str, name: str, save_path: str):
    img = cv2.imread(os.path.join(img_folder, name), -1).astype(np.float32)
    print_min_max(img)
    # img_g = img[:,:,1]
    # value_at_99_percentile = np.percentile(img_g, 99)
    # img /= value_at_99_percentile
    # img *= 4000.0
    img = np.clip(img, 0.05, 4000.0)

    print_min_max(img)
    print("**********")

    new_name = name
    # new_name = name.split('.')[0] + "_processed.hdr"

    save_hdr(img, save_path, new_name)


def downsample2x(img_folder: str, name: str):
    img = cv2.imread(os.path.join(img_folder, name), -1).astype(np.float32)
    downscaled = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    new_name = name.split('.')[0] + "_2x.hdr"
    save_hdr(downscaled, img_folder, new_name)


def downsample4x(img_folder: str, name: str, save_path: str):
    img = cv2.imread(os.path.join(img_folder, name), -1).astype(np.float32)
    assert np.min(img) >= 0.0, print(name)
    downscaled = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
    new_name = name.split('.')[0] + "_4x.hdr"
    save_hdr(downscaled, save_path, new_name)
    print_min_max(downscaled)

def downsample4x(img):
    downscaled = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
    return downscaled

def print_info(img_folder: str, name: str):
    img = cv2.imread(os.path.join(img_folder, name), -1).astype(np.float32)
    print_min_max(img)


def compare_content(a_folder, b_folder):
    for file_name in file_list:
        print(file_name, os.path.exists(os.path.join(b_folder, file_name.replace('4x_', ''))))

def draw_histogram(array, mode, save_path):
    array = torch.log(array + torch.ones_like(array) * 1e-5)
    array = array.squeeze(0).cpu().permute(1,2,0).detach().numpy()
    fig, ax = plt.subplots()
    sns.distplot(array.flatten(), bins=100, kde=False)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Log Histogram of {} Prediction'.format(mode))
    plt.savefig(os.path.join(save_path + '{}_prediction.png'.format(mode)))

def exr2hdr(img):
    green = img[:,:,1]
    p = np.percentile(green, 99)
    img = img / p
    img = img * 4000.0
    img = np.clip(img, 0.05, 4000.0)
    return img

def save_exr(img, folder, name):
    header = OpenEXR.Header(img.shape[1], img.shape[0])
    header['channels'] = dict([(c, Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))) for c in "RGB"])

    # Create an OpenEXR file
    file = OpenEXR.OutputFile(os.path.join(folder, name), header)

    # Convert the numpy array data into a string
    red = (img[:,:,2].astype(np.float32)).tobytes()
    green = (img[:,:,1].astype(np.float32)).tobytes()
    blue = (img[:,:,0].astype(np.float32)).tobytes()

    # Write the image data to the exr file
    file.writePixels({'R': red, 'G': green, 'B': blue})



# folder_path = "/home/luoleyouluole/Image-Restoration-Experiments/data/hdr_data/test" # replace with the path to your image folder
# save_path = "/home/luoleyouluole/Image-Restoration-Experiments/data/hdr_data/test_d_4x"
# file_list = os.listdir(folder_path)
# file_list.sort()

# for file_name in file_list:
#     downsample4x(folder_path, file_name, save_path)

# img = cv2.imread("/home/luoleyouluole/Image-Restoration-Experiments/data/rit_hdr4000/Ahwahnee_Great_Lounge.hdr", -1).astype(np.float32)
# img = torch.tensor(img)

# img /= np.max(img)
# img = np.power(img, 1/2.2)
# save_hdr(img, "/home/luoleyouluole/Image-Restoration-Experiments/", "GT.hdr")


# draw_histogram(img, "GT", "./")
