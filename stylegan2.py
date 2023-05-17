import math
import random

import torch
from basicsr.archs.arch_util import default_init_weights
from torch import nn
from torch.nn import functional as F


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super(ScaledLeakyReLU, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        out = F.leaky_relu(x, negative_slope=self.negative_slope)
        return out * math.sqrt(2)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mode="down"):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if mode == "down":
            self.scale_factor = 0.5
        elif mode == "up":
            self.scale_factor = 2

    def forward(self, x):
        out = F.leaky_relu_(self.conv1(x), negative_slope=0.2)
        # upsample/downsample
        out = F.interpolate(
            out, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        )
        out = F.leaky_relu_(self.conv2(out), negative_slope=0.2)
        # skip
        x = F.interpolate(
            x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        )
        skip = self.skip(x)
        out = out + skip
        return out


class NormStyleCode(nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        num_style_feat,
        demodulate=True,
        sample_mode=None,
        eps=1e-8,
    ):
        super(ModulatedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.sample_mode = sample_mode
        self.eps = eps

        # modulation inside each modulated conv
        self.modulation = nn.Linear(num_style_feat, in_channels, bias=True)
        # initialization
        default_init_weights(
            self.modulation,
            scale=1,
            bias_fill=1,
            a=0,
            mode="fan_in",
            nonlinearity="linear",
        )

        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
            / math.sqrt(in_channels * kernel_size**2)
        )
        self.padding = kernel_size // 2

    def forward(self, x, style):

        b, c, h, w = x.shape  # c = c_in
        # weight modulation
        style = self.modulation(style).view(b, 1, c, 1, 1)
        # self.weight: (1, c_out, c_in, k, k); style: (b, 1, c, 1, 1)
        weight = self.weight * style  # (b, c_out, c_in, k, k)

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(b, self.out_channels, 1, 1, 1)

        weight = weight.view(
            b * self.out_channels, c, self.kernel_size, self.kernel_size
        )

        # upsample or downsample if necessary
        if self.sample_mode == "upsample":
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        elif self.sample_mode == "downsample":
            x = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)

        b, c, h, w = x.shape
        x = x.view(1, b * c, h, w)
        # weight: (b*c_out, c_in, k, k), groups=b
        out = F.conv2d(x, weight, padding=self.padding, groups=b)
        out = out.view(b, self.out_channels, *out.shape[2:4])

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, demodulate={self.demodulate}, sample_mode={self.sample_mode})"
        )


class StyleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        num_style_feat,
        demodulate=True,
        sample_mode=None,
    ):
        super(StyleConv, self).__init__()
        self.modulated_conv = ModulatedConv2d(
            in_channels,
            out_channels,
            kernel_size,
            num_style_feat,
            demodulate=demodulate,
            sample_mode=sample_mode,
        )
        self.weight = nn.Parameter(torch.zeros(1))  # for noise injection
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, style, noise=None):
        # modulate
        out = self.modulated_conv(x, style) * 2**0.5  # for conversion
        # noise injection
        if noise is None:
            b, _, h, w = out.shape
            noise = out.new_empty(b, 1, h, w).normal_()
        out = out + self.weight * noise
        # add bias
        out = out + self.bias
        # activation
        out = self.activate(out)
        return out


class ToRGB(nn.Module):
    def __init__(self, in_channels, num_style_feat, upsample=True):
        super(ToRGB, self).__init__()
        self.upsample = upsample
        self.modulated_conv = ModulatedConv2d(
            in_channels,
            3,
            kernel_size=1,
            num_style_feat=num_style_feat,
            demodulate=False,
            sample_mode=None,
        )
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x, style, skip=None):

        out = self.modulated_conv(x, style)
        out = out + self.bias
        if skip is not None:
            if self.upsample:
                skip = F.interpolate(
                    skip, scale_factor=2, mode="bilinear", align_corners=False
                )
            out = out + skip
        return out


class ConstantInput(nn.Module):
    def __init__(self, num_channel, size):
        super(ConstantInput, self).__init__()
        self.weight = nn.Parameter(torch.randn(1, num_channel, size, size))

    def forward(self, batch):
        out = self.weight.repeat(batch, 1, 1, 1)
        return out


class StyleGAN2Generator(nn.Module):
    def __init__(
        self, out_size, num_style_feat=512, num_mlp=8, channel_multiplier=2, narrow=1
    ):
        super(StyleGAN2Generator, self).__init__()
        # Style MLP layers
        self.num_style_feat = num_style_feat
        style_mlp_layers = [NormStyleCode()]
        for _ in range(num_mlp):
            style_mlp_layers.extend(
                [
                    nn.Linear(num_style_feat, num_style_feat, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                ]
            )
        self.style_mlp = nn.Sequential(*style_mlp_layers)
        # initialization
        default_init_weights(
            self.style_mlp,
            scale=1,
            bias_fill=0,
            a=0.2,
            mode="fan_in",
            nonlinearity="leaky_relu",
        )

        # channel list
        channels = {
            "4": int(512 * narrow),
            "8": int(512 * narrow),
            "16": int(512 * narrow),
            "32": int(512 * narrow),
            "64": int(256 * channel_multiplier * narrow),
            "128": int(128 * channel_multiplier * narrow),
            "256": int(64 * channel_multiplier * narrow),
            "512": int(32 * channel_multiplier * narrow),
            "1024": int(16 * channel_multiplier * narrow),
        }
        self.channels = channels

        self.constant_input = ConstantInput(channels["4"], size=4)
        self.style_conv1 = StyleConv(
            channels["4"],
            channels["4"],
            kernel_size=3,
            num_style_feat=num_style_feat,
            demodulate=True,
            sample_mode=None,
        )
        self.to_rgb1 = ToRGB(channels["4"], num_style_feat, upsample=False)

        self.log_size = int(math.log(out_size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1
        self.num_latent = self.log_size * 2 - 2

        self.style_convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channels = channels["4"]
        # noise
        for layer_idx in range(self.num_layers):
            resolution = 2 ** ((layer_idx + 5) // 2)
            shape = [1, 1, resolution, resolution]
            self.noises.register_buffer(f"noise{layer_idx}", torch.randn(*shape))
        # style convs and to_rgbs
        for i in range(3, self.log_size + 1):
            out_channels = channels[f"{2**i}"]
            self.style_convs.append(
                StyleConv(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    num_style_feat=num_style_feat,
                    demodulate=True,
                    sample_mode="upsample",
                )
            )
            self.style_convs.append(
                StyleConv(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    num_style_feat=num_style_feat,
                    demodulate=True,
                    sample_mode=None,
                )
            )
            self.to_rgbs.append(ToRGB(out_channels, num_style_feat, upsample=True))
            in_channels = out_channels

    def make_noise(self):
        """Make noise for noise injection."""
        device = self.constant_input.weight.device
        noises = [torch.randn(1, 1, 4, 4, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2**i, 2**i, device=device))

        return noises

    def get_latent(self, x):
        return self.style_mlp(x)

    def mean_latent(self, num_latent):
        latent_in = torch.randn(
            num_latent, self.num_style_feat, device=self.constant_input.weight.device
        )
        latent = self.style_mlp(latent_in).mean(0, keepdim=True)
        return latent

    def forward(
        self,
        styles,
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
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)  # feature back to the rgb space
            i += 2

        image = skip

        if return_latents:
            return image, latent
        else:
            return image, None


class StyleGAN2Discriminator(nn.Module):
    def __init__(
        self,
        out_size,
        channel_multiplier=2,
        stddev_group=4,
        narrow=1,
    ):
        super(StyleGAN2Discriminator, self).__init__()

        channels = {
            "4": int(512 * narrow),
            "8": int(512 * narrow),
            "16": int(512 * narrow),
            "32": int(512 * narrow),
            "64": int(256 * channel_multiplier * narrow),
            "128": int(128 * channel_multiplier * narrow),
            "256": int(64 * channel_multiplier * narrow),
            "512": int(32 * channel_multiplier * narrow),
            "1024": int(16 * channel_multiplier * narrow),
        }

        log_size = int(math.log(out_size, 2))

        self.first_conv = nn.Conv2d(3, channels[f"{out_size}"], 1)
        self.conv_body = []

        in_channels = channels[f"{out_size}"]
        for i in range(log_size, 2, -1):
            out_channels = channels[f"{2**(i - 1)}"]
            self.conv_body.append(ResBlock(in_channels, out_channels, mode="down"))
            in_channels = out_channels
        self.conv_body = nn.Sequential(*self.conv_body)

        self.final_conv = nn.Conv2d(
            in_channels + 1, channels["4"], kernel_size=3, stride=1, padding=1
        )
        self.final_linear = nn.Sequential(
            nn.Linear(
                channels["4"] * 4 * 4,
                channels["4"],
                bias=True,
            ),
            ScaledLeakyReLU(0.2),
            nn.Linear(channels["4"], 1, bias=True),
        )
        self.stddev_group = stddev_group
        self.stddev_feat = 1

    def forward(self, x):
        out = F.leaky_relu_(self.first_conv(x), negative_slope=0.2)
        out = self.conv_body(out)

        b, c, h, w = out.shape
        # concatenate a group stddev statistics to out
        group = min(
            b, self.stddev_group
        )  # Minibatch must be divisible by (or smaller than) group_size
        stddev = out.view(group, -1, self.stddev_feat, c // self.stddev_feat, h, w)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, h, w)
        out = torch.cat([out, stddev], 1)

        out = F.leaky_relu_(self.final_conv(out), negative_slope=0.2)
        out = out.view(b, -1)
        out = self.final_linear(out)

        return out
