import torch
import torch.nn as nn

class SegNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=11, features=[64, 128, 256, 512]):
        super(SegNet, self).__init__()
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        # Encoder (Downsampling path)
        for feature in features:
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = feature

        # Decoder (Upsampling path)
        for feature in reversed(features):
            self.decoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True)
                )
            )
            self.decoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True)
                )
            )

        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        indices_list = []
        size_list = []

        # Encoder
        for layer in self.encoder_layers:
            x = layer(x)
            size_list.append(x.size())
            x, indices = self.pool(x)
            indices_list.append(indices)

        # Decoder
        for i in range(0, len(self.decoder_layers), 2):
            indices = indices_list.pop()
            size = size_list.pop()
            x = self.unpool(x, indices, output_size=size)
            x = self.decoder_layers[i](x)
            x = self.decoder_layers[i + 1](x)

        return self.final_conv(x)

# Example usage
#if __name__ == "__main__":
   # model = SegNet(in_channels=3, out_channels=11)  # 11 classes as defined in the mapping
    #x = torch.randn((1, 3, 256, 256))
    #preds = model(x)
    #print(preds.shape)  # Should output [1, 11, 256, 256]
