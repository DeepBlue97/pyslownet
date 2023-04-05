
import numpy as np


class ConvolutionalLayer:
    def __init__(self, in_channel: int, out_channel: int, kernel_size: int,
        padding: int, stride: int
    ):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.weights = 

    def forward():
        pass

    def backward():
        pass

    def out_height():
        pass

    def out_width():
        pass

    def get_images():
        pass

    def get_delta():
        pass

    def resize():
        pass

if __name__ == "__main__":
    # conv = ConvolutionalLayer(3, 16, (3, 3),  (3, 3),  (3, 3))
    conv = ConvolutionalLayer(3, 16, 3, 3, 3)
