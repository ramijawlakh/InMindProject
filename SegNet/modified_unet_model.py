
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        return self.conv(x) + self.residual(x)


class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=None):
        super(UNET, self).__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsampling part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Upsampling part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
            skip_connections = []
            for down in self.downs:
                x = down(x)
                skip_connections.append(x)
                x = self.pool(x)

            x = self.bottleneck(x)
            skip_connections = skip_connections[::-1]

            deep_outputs = []

            for idx in range(0, len(self.ups), 2):
                x = self.ups[idx](x)  # ConvTranspose2d upsampling
                skip_connection = skip_connections[idx // 2]

                if x.size() != skip_connection.size():
                    x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)

                concat_skip = torch.cat((skip_connection, x), dim=1)
                x = self.ups[idx + 1](concat_skip)  # DoubleConv

                deep_outputs.append(x)

            # Ensure all tensors in deep_outputs have the same size before summing
            min_height = min([x.size(2) for x in deep_outputs])
            min_width = min([x.size(3) for x in deep_outputs])
            deep_outputs = [F.interpolate(x, size=(min_height, min_width), mode='bilinear', align_corners=False) for x in deep_outputs]

            output = sum(deep_outputs) / len(deep_outputs)
            return self.final_conv(output)
    

def test():
    x = torch.randn((3, 1, 160, 160))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    assert preds.shape == x.shape


if __name__ == "__main__":
    test()
