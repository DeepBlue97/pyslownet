# coding: utf-8
import numpy as np
from common.functions import *
from common.util import im2col, col2im


class Dropout:
    """
    https://arxiv.org/abs/1207.0580
    """

    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h  # 2
        self.pool_w = pool_w  # 2
        self.stride = stride  # 2
        self.pad = pad  # 0

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)  # (14400, 120)
        col = col.reshape(-1, self.pool_h * self.pool_w)  # (432000, 4)

        arg_max = np.argmax(col, axis=1)  # (432000,) 保存最大值的索引
        out = np.max(col, axis=1)  # (432000,) 保存最大值
        out = out.reshape((N, out_h, out_w, C)).transpose(0, 3, 1, 2)  # (100, 30, 12, 12)

        self.x = x  # (100, 30, 24, 24)
        self.arg_max = arg_max  # ↑

        return out  # (100, 30, 12, 12)

    def backward(self, dout):  # (100, 30, 12, 12)
        dout = dout.transpose(0, 2, 3, 1)  # (100, 12, 12, 30)

        pool_size = self.pool_h * self.pool_w  # 2*2 = 4
        dmax = np.zeros((dout.size, pool_size))  # (432000, 4)
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()  # (432000,)
        dmax = dmax.reshape(dout.shape + (pool_size,))  # (100, 12, 12, 30, 4)

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)  # (14400, 120) = 1728000
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)  # (100, 30, 24, 24) = 1728000

        return dx
