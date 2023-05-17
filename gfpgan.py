import math
import random

import torch
# from iopath.common.file_io import PathManager
# from iopath.fb.manifold import ManifoldPathHandler
from torch import nn
from torch.nn import functional as F

from stylegan2 import ResBlock, ScaledLeakyReLU, StyleGAN2Generator


class FacialComponentDiscriminator(nn.Module):
    # Facial component (eyes, mouth, noise) discriminator

    def __init__(self):
        super(FacialComponentDiscriminator, self).__init__()
        # It now uses a VGG-style architectrue with fixed model size
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.final_conv = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=0)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.slrelu = ScaledLeakyReLU()

    def forward(self, x, return_feats=False, **kwargs):
        # return_feats (bool): Whether to return intermediate features.
        feat = self.lrelu(self.conv1(x))
        feat = self.lrelu(self.conv2(feat))
        feat = self.lrelu(self.conv3(feat))
        rlt_feats = []
        if return_feats:
            rlt_feats.append(feat.clone())
        feat = self.lrelu(self.conv4(feat))
        feat = self.lrelu(self.conv5(feat))
        if return_feats:
            rlt_feats.append(feat.clone())
        out = self.slrelu(self.final_conv(feat))

        if return_feats:
            return out, rlt_feats
        else:
            return out, None


class StyleGAN2GeneratorSFT(StyleGAN2Generator):
    def __init__(
        self,
        out_size,
        num_style_feat=512,
        num_mlp=8,
        channel_multiplier=2,
        narrow=1,
        sft_half=False,
    ):
        super(StyleGAN2GeneratorSFT, self).__init__(
            out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
            narrow=narrow,
        )
        self.sft_half = sft_half

    def forward(
        self,
        styles,
        conditions,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
        truncation=1,
        truncation_latent=None,
        inject_index=None,
        return_latents=False,
    ):

        # style codes -> latents with Style MLP layer
        if not input_is_latent:
            styles = [self.style_mlp(s) for s in styles]
        # noises
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers  # for each style conv layer
            else:  # use the stored noise
                noise = [
                    getattr(self.noises, f"noise{i}") for i in range(self.num_layers)
                ]
        # style truncation
        if truncation < 1:
            style_truncation = []
            for style in styles:
                style_truncation.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )
            styles = style_truncation
        # get style latents with injection
        if len(styles) == 1:
            inject_index = self.num_latent

            if styles[0].ndim < 3:
                # repeat latent code for all the layers
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:  # used for encoder with different latent code for each layer
                latent = styles[0]
        elif len(styles) == 2:  # mixing noises
            if inject_index is None:
                inject_index = random.randint(1, self.num_latent - 1)
            latent1 = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = (
                styles[1].unsqueeze(1).repeat(1, self.num_latent - inject_index, 1)
            )
            latent = torch.cat([latent1, latent2], 1)

        # main generation
        out = self.constant_input(latent.shape[0])
        out = self.style_conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.style_convs[::2],
            self.style_convs[1::2],
            noise[1::2],
            noise[2::2],
            self.to_rgbs,
        ):
            out = conv1(out, latent[:, i], noise=noise1)

            # the conditions may have fewer levels
            if i < len(conditions):
                # SFT part to combine the conditions
                if self.sft_half:  # only apply SFT to half of the channels
                    out_same, out_sft = torch.split(out, int(out.size(1) // 2), dim=1)
                    out_sft = out_sft * conditions[i - 1] + conditions[i]
                    out = torch.cat([out_same, out_sft], dim=1)
                else:  # apply SFT to all the channels
                    out = out * conditions[i - 1] + conditions[i]

            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)  # feature back to the rgb space
            i += 2

        image = skip

        if return_latents:
            return image, latent
        else:
            return image, None


class GFPGAN(nn.Module):
    def __init__(
        self,
        out_size=512,
        num_style_feat=512,
        channel_multiplier=2,
        decoder_load_path=None,
        fix_decoder=False,
        # for stylegan decoder
        num_mlp=8,
        input_is_latent=True,
        different_w=True,
        narrow=1,
        sft_half=True,
    ):

        super(GFPGAN, self).__init__()
        self.input_is_latent = input_is_latent
        self.different_w = different_w
        self.num_style_feat = num_style_feat

        unet_narrow = narrow * 0.5  # by default, use a half of input channels
        channels = {
            "4": int(512 * unet_narrow),
            "8": int(512 * unet_narrow),
            "16": int(512 * unet_narrow),
            "32": int(512 * unet_narrow),
            "64": int(256 * channel_multiplier * unet_narrow),
            "128": int(128 * channel_multiplier * unet_narrow),
            "256": int(64 * channel_multiplier * unet_narrow),
            "512": int(32 * channel_multiplier * unet_narrow),
            "1024": int(16 * channel_multiplier * unet_narrow),
        }

        self.log_size = int(math.log(out_size, 2))
        first_out_size = 2 ** (int(math.log(out_size, 2)))

        self.conv_body_first = nn.Conv2d(3, channels[f"{first_out_size}"], 1)

        # downsample
        in_channels = channels[f"{first_out_size}"]
        self.conv_body_down = nn.ModuleList()
        for i in range(self.log_size, 2, -1):
            out_channels = channels[f"{2**(i - 1)}"]
            self.conv_body_down.append(ResBlock(in_channels, out_channels, mode="down"))
            in_channels = out_channels

        self.final_conv = nn.Conv2d(in_channels, channels["4"], 3, 1, 1)

        # upsample
        in_channels = channels["4"]
        self.conv_body_up = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f"{2**i}"]
            self.conv_body_up.append(ResBlock(in_channels, out_channels, mode="up"))
            in_channels = out_channels

        # to RGB
        self.toRGB = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            self.toRGB.append(nn.Conv2d(channels[f"{2**i}"], 3, 1))

        if different_w:
            linear_out_channel = (int(math.log(out_size, 2)) * 2 - 2) * num_style_feat
        else:
            linear_out_channel = num_style_feat

        self.final_linear = nn.Linear(channels["4"] * 4 * 4, linear_out_channel)

        # the decoder: stylegan2 generator with SFT modulations
        self.stylegan_decoder = StyleGAN2GeneratorSFT(
            out_size=out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
            narrow=narrow,
            sft_half=sft_half,
        )

        # load pre-trained stylegan2 model if necessary
        # if decoder_load_path:
        #     pathmgr = PathManager()
        #     pathmgr.register_handler(ManifoldPathHandler())
        #     self.stylegan_decoder.load_state_dict(
        #         torch.load(
        #             pathmgr.get_local_path(decoder_load_path),
        #             map_location=lambda storage, loc: storage,
        #         )
        #     )
        # fix decoder without updating params
        if fix_decoder:
            for _, param in self.stylegan_decoder.named_parameters():
                param.requires_grad = False

        # for SFT modulations (scale and shift)
        self.condition_scale = nn.ModuleList()
        self.condition_shift = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f"{2**i}"]
            if sft_half:
                sft_out_channels = out_channels
            else:
                sft_out_channels = out_channels * 2
            self.condition_scale.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_channels, sft_out_channels, 3, 1, 1),
                )
            )
            self.condition_shift.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_channels, sft_out_channels, 3, 1, 1),
                )
            )

    def preprocess(self, input):
        input = input / 255.0
        from torchvision.transforms.functional import normalize
        input = normalize(input, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        return input

    def post_process(self, tensor, rgb2bgr=True, min_max=(-1, 1)):
        output = tensor.clamp_(*min_max)
        output = (output - min_max[0]) / (min_max[1] - min_max[0]) * 255
        return output

    def forward(
        self, x, return_latents=False, return_rgb=True, randomize_noise=True, **kwargs
    ):

        conditions = []
        unet_skips = []
        out_rgbs = []

        x = self.preprocess(x)

        # encoder
        feat = F.leaky_relu_(self.conv_body_first(x), negative_slope=0.2)
        for i in range(self.log_size - 2):
            feat = self.conv_body_down[i](feat)
            unet_skips.insert(0, feat)
        feat = F.leaky_relu_(self.final_conv(feat), negative_slope=0.2)

        # style code
        style_code = self.final_linear(feat.view(feat.size(0), -1))
        if self.different_w:
            style_code = style_code.view(style_code.size(0), -1, self.num_style_feat)

        # decode
        for i in range(self.log_size - 2):
            # add unet skip
            feat = feat + unet_skips[i]
            # ResUpLayer
            feat = self.conv_body_up[i](feat)
            # generate scale and shift for SFT layers
            scale = self.condition_scale[i](feat)
            conditions.append(scale.clone())
            shift = self.condition_shift[i](feat)
            conditions.append(shift.clone())
            # generate rgb images
            if return_rgb:
                out_rgbs.append(self.toRGB[i](feat))

        # decoder
        image, _ = self.stylegan_decoder(
            [style_code],
            conditions,
            return_latents=return_latents,
            input_is_latent=self.input_is_latent,
            randomize_noise=randomize_noise,
        )

        image = self.post_process(image)

        return image
