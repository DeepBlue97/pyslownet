
class Affine:
    def __init__(self, W, b):
        self.W = W  # (100, 10) or (4320, 100)
        self.b = b  # (10,)

        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分 权重、偏置参数的微分
        self.dW = None
        self.db = None

    def forward(self, x):
        # テンソル対応 张量对应
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x  # (100, 100) or (100, 4320)

        out = np.dot(self.x, self.W) + self.b  # (100, 10) or (100, 100)

        return out

    def backward(self, dout):  # (100, 10)
        dx = np.dot(dout, self.W.T)  # (100, 10).dot((10, 100)) == (100, 100)
        self.dW = np.dot(self.x.T, dout)  # (100, 100).dot((100, 10)) == (100, 10)
        self.db = np.sum(dout, axis=0)  # (10,)

        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す 恢复输入数据的形状（テンソル対応）张量对应
        return dx  # (100, 10) or (4320, 100)
