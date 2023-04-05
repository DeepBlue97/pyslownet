from pyslownet import *
from pyslownet.utils import im2col, col2im


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 中间数据（在反向传播时使用）
        self.x = None
        self.col = None
        self.col_W = None

        # 权重和偏置参数的梯度
        self.dW = None
        self.db = None

    def forward(self, x):
        """
        将前向传播来的特征图进行卷积操作，得到新的特征图给到下一层。
        """
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)  # 卷积ready列
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W  # 本层的梯度，其实就是本层的权重

        return out

    def backward(self, dout):
        """
        将反向传播来的梯度乘上本层的梯度，得出本层对最终损失的梯度。
        """
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        # 调整本层权重用的梯度
        self.dW = np.dot(self.col.T, dout)  # 本层权重变动对最终损失的影响，调整权重就靠这个
        self.dW = self.dW.transpose((1, 0)).reshape((FN, C, FH, FW))
        # 给上一层准备的
        dcol = np.dot(dout, self.col_W.T)  # 图像（也就是上层的输出）变动对最终损失的影响，卷积ready列
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)  # 梯度转换为图像形状

        return dx


if __name__ == "__main__":

    conv = Convolution(np.ndarray((6,4,3,3)), np.ndarray(6,3,3))



















# # import numpy as np
# import cupy as cp


# class ConvolutionalLayer:
#     def __init__(self, in_channel: int, out_channel: int, kernel_size: int = 3, padding: int = 1, stride: int = 1):
#         self.in_channel = in_channel
#         self.out_channel = out_channel
#         self.kernel_size = kernel_size
#         self.padding = padding
#         self.stride = stride

#         self.weights = cupy.random.rand(self.in_channel * self.out_channel * self.kernel_size ** 2
#         ).reshape(self.out_channel, self.in_channel, self.kernel_size, self.kernel_size)

#     def forward(x):
#         """
#         x: b, c, h, w
#         """
#         y = scipy.signal.convolve2d(x, self.kernel, mode='same')
#         return y

#     @property
#     def kernel(self):
#         return self.weights

#     def backward():
#         pass

#     def out_height():
#         pass

#     def out_width():
#         pass

#     def get_images():
#         pass

#     def get_delta():
#         pass

#     def resize():
#         pass

# if __name__ == "__main__":
#     # conv = ConvolutionalLayer(3, 16, (3, 3),  (3, 3),  (3, 3))
#     conv = ConvolutionalLayer(3, 16, 3, 3, 3)
