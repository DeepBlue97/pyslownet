from pyslownet import *
from pyslownet.functions import sigmoid


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        # 根据求导公式可得出
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0  # 小于0的地方梯度为0，其它地方原样输出
        dx = dout

        return dx


class Softmax:
    def __init__(self):
        self.y = None
        self.t = None
        self.x = None

    def forward(self, x, t):
        self.t = t
        self.x = x
        self.y = softmax(x)
        return self.y

    def backward(self, dout):
        """
        分两种情况求偏导，可得到梯度为：
        当输入输出标号相同时：si * (1 - si)
        当输入输出标号不同时：-si * sj
        """
        batch_size = self.t.shape[0]
        # jacobian.shape == (batch, 输出length, 输入length)
        jacobian = np.zeros(batch_size, self.y.shape[1], self.x.shape[1])
        # 梯度矩阵赋值
        for bi in range(batch_size):
            for ri in range(jacobian.shape[1]):
                for ci in range(jacobian.shape[2]):
                    if ri == ci:  # 在角线上
                        jacobian[bi][ri][ci] = self.x[bi][ci] * (1 - self.x[bi][ci])
                    else:
                        jacobian[bi][ri][ci] = -self.x[bi][ci] * self.y[bi][ri]
        # # 对角线赋值
        # for bi in jacobian:
        #     for diagi in range(jacobian.shape[1]):
        #         jacobian[bi, diagi, diagi] = self.x[bi, diagi, diagi] * (1 - self.x[bi, diagi, diagi])

        # dx = np.dot(dout, jacobian.T) / batch_size
        dx = np.zeros((batch_size, self.x.shape[1]))
        for bi in range(batch_size):
            dx[bi] = dout.dot(jacobian[bi])

        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None  # softmaxの出力
        self.t = None  # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        """
        以下梯度公式有严格的推理过程，最终结论为si-yi（输入减去标签值）
        """
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 教師データがone-hot-vectorの場合
            dx: np.ndarray = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1  # 使得正确标号对应的输入其偏导为负（使得该处输出值越大损失值越小）
            dx: np.ndarray = dx / batch_size  # 减小偏导的绝对值的简便方法

        return dx  # (100, 10)
