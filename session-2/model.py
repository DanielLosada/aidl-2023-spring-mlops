import torch.nn as nn
from convblock import ConvBlock
from linearblock import LinearBlock

class MyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = ConvBlock(num_inp_channels=1, num_out_fmaps=6, kernel_size=5)
        self.conv2 = ConvBlock(num_inp_channels=6, num_out_fmaps=6, kernel_size=5)
        self.conv3 = ConvBlock(num_inp_channels=6, num_out_fmaps=6, kernel_size=5)

        self.mlp = nn.Sequential(
            LinearBlock(120, 84),
            LinearBlock(84, 10),
            nn.LogSoftmax(-1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        print("Shape: ", x.shape)
        bsz, nch, height, width = x.shape
        x = x.reshape(-1, nch * height * width)
        #x = x.view(x.size(0), -1)  # flatten the output for the MLP
        print("Shape after: ", x.shape)
        x = self.mlp(x)
        return x
