from pyslownet.convolutional_layer import Convolution


class ResNetBlock:

    def __init__(self, in_channel: int):

        bottleneck = in_channel // 2

        self.conv_1 = Convolution(
                np.random.randn(bottleneck, in_channel, 1, 1), 
                np.random.randn(), 
                stride=1, pad=1)
        self.conv_2 = Convolution(
                np.random.randn(in_channel, bottleneck, 3, 3),
                np.random.randn(), 
                stride=1, pad=1)

    def forward(self, x):
        x = self.conv_1.forward(x)
        out = self.conv_2.forward(x)
        return out

    def backward(self, dout):
        dout = self.conv_1.backward(dout)
        dout = self.conv_2.backward(dout)
        return dout
