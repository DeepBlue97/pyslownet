from pyslownet import *

from pyslownet.convolutional_layer import Convolution


if __name__ == "__main__":

    img = np.random.randn(2, 4, 28, 28).astype(np.float32)

    conv = Convolution(np.random.randn(6,4,3,3), np.random.randn(6) , stride=1, pad=1)

    out = conv.forward(img)
    d_img = conv.backward(out)
    print(print(out))
    print(out.shape)
    print(d_img.shape) # (2, 4, 28, 28)
