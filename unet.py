import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder
        self.encoder_conv1 = self.conv_block(in_channels, 64)
        self.encoder_conv2 = self.conv_block(64, 128)
        self.encoder_conv3 = self.conv_block(128, 256)
        self.encoder_conv4 = self.conv_block(256, 512)

        # Max-pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.decoder_upconv3 = self.upconv_block(512, 256)
        self.decoder_conv3 = self.conv_block(512, 256)
        self.decoder_upconv2 = self.upconv_block(256, 128)
        self.decoder_conv2 = self.conv_block(256, 128)
        self.decoder_upconv1 = self.upconv_block(128, 64)
        self.decoder_conv1 = self.conv_block(128, 64)

        # Output layer
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder_conv1(x)
        enc2 = self.encoder_conv2(self.pool(enc1))
        enc3 = self.encoder_conv3(self.pool(enc2))
        enc4 = self.encoder_conv4(self.pool(enc3))

        # Decoder
        dec3 = self.decoder_upconv3(enc4)
        dec3 = self.decoder_conv3(torch.cat((enc3, dec3), 1))
        dec2 = self.decoder_upconv2(dec3)
        dec2 = self.decoder_conv2(torch.cat((enc2, dec2), 1))
        dec1 = self.decoder_upconv1(dec2)
        dec1 = self.decoder_conv1(torch.cat((enc1, dec1), 1))

        # Output
        return self.out(dec1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

# Instantiate the U-Net model
in_channels = 3 # Example: number of input channels for an RGB image
out_channels = 2 # Example: number of output channels for binary segmentation
model = UNet(in_channels, out_channels)
