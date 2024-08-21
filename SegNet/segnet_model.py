import torch
import torch.nn as nn
import torch.nn.functional as F

class SegNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=11):
        super(SegNet, self).__init__()

        # Encoder layers
        self.enc1 = self._encoder_block(in_channels, 64)
        self.enc2 = self._encoder_block(64, 128)
        self.enc3 = self._encoder_block(128, 256, num_layers=3)
        self.enc4 = self._encoder_block(256, 512, num_layers=3)
        self.enc5 = self._encoder_block(512, 512, num_layers=3)

        # Decoder layers
        self.dec5 = self._decoder_block(512, 512, num_layers=3)
        self.dec4 = self._decoder_block(512, 512, num_layers=3)
        self.dec3 = self._decoder_block(512, 256, num_layers=3)
        self.dec2 = self._decoder_block(256, 128)
        self.dec1 = self._decoder_block(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def _encoder_block(self, in_channels, out_channels, num_layers=2):
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _decoder_block(self, in_channels, out_channels, num_layers=2):
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        x1, ind1 = F.max_pool2d(self.enc1(x), kernel_size=2, stride=2, return_indices=True)
        x2, ind2 = F.max_pool2d(self.enc2(x1), kernel_size=2, stride=2, return_indices=True)
        x3, ind3 = F.max_pool2d(self.enc3(x2), kernel_size=2, stride=2, return_indices=True)
        x4, ind4 = F.max_pool2d(self.enc4(x3), kernel_size=2, stride=2, return_indices=True)
        x5, ind5 = F.max_pool2d(self.enc5(x4), kernel_size=2, stride=2, return_indices=True)

        # Decoder
        x5d = F.max_unpool2d(self.dec5(x5), ind5, kernel_size=2, stride=2, output_size=x4.size())
        x4d = F.max_unpool2d(self.dec4(x5d), ind4, kernel_size=2, stride=2, output_size=x3.size())
        x3d = F.max_unpool2d(self.dec3(x4d), ind3, kernel_size=2, stride=2, output_size=x2.size())
        x2d = F.max_unpool2d(self.dec2(x3d), ind2, kernel_size=2, stride=2, output_size=x1.size())
        x1d = F.max_unpool2d(self.dec1(x2d), ind1, kernel_size=2, stride=2, output_size=x.size())

        x_final = self.final_conv(x1d)
        return x_final

# Example usage:
if __name__ == "__main__":
    model = SegNet(in_channels=3, out_channels=11)  # Adjust the number of output channels as per your use case
    x = torch.randn((1, 3, 512, 512))
    preds = model(x)
    print("Final output shape: ", preds.shape)  # Expected output shape should be [1, 11, 512, 512]
