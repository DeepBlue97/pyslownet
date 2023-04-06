class Metadata:
    def __init__(self, file: str):
        pass


class Layer:
    pass


class Tree():
    def __init__(self, filename: str) -> None:

        self.leaf = 0
        self.n = 0
        self.parent = 0
        self.child = 0
        self.group = 0
        self.name = ''

        self.groups = 0
        self.group_size = 0
        self.group_offset = 0


ACTIVATION = set(
    'LOGISTIC', 'RELU', 'RELIE', 'LINEAR', 'RAMP', 'TANH', 'PLSE',
    'LEAKY', 'ELU', 'LOGGY', 'STAIR', 'HARDTAN', 'LHTAN', 'SELU'
)

IMTYPE = set('PNG', 'BMP', 'TGA', 'JPG')

BINARY_ACTIVATION = set('MULT', 'ADD', 'SUB', 'DIV')

LAYER_TYPE = set(
    'CONVOLUTIONAL',
    'DECONVOLUTIONAL',
    'CONNECTED',
    'MAXPOOL',
    'SOFTMAX',
    'DETECTION',
    'DROPOUT',
    'CROP',
    'ROUTE',
    'COST',
    'NORMALIZATION',
    'AVGPOOL',
    'LOCAL',
    'SHORTCUT',
    'ACTIVE',
    'RNN',
    'GRU',
    'LSTM',
    'CRNN',
    'BATCHNORM',
    'NETWORK',
    'XNOR',
    'REGION',
    'YOLO',
    'ISEG',
    'REORG',
    'UPSAMPLE',
    'LOGXENT',
    'L2NORM',
    'BLANK'
)

COST_TYPE = set('SSE', 'MASKED', 'L1', 'SEG', 'SMOOTH', 'WGAN')

update_args = {
    'batch': 0,
    'learning_rate': 0.,
    'momentum': 0.,
    'decay': 0.,
    'adam': 0,
    'B1': 0.,
    'B2': 0.,
    'eps': 0.,
    't': 0
}

learning_rate_policy = set('CONSTANT', 'STEP', 'EXP', 'POLY', 'STEPS', 'SIG', 'RANDOM')

class Network:
    pass
