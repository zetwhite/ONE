from enum import Enum

# Optimizer
class Optimizer(dict):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

class SGD(Optimizer):
    name = ['sgd', 'stocasticgradientdescent', 'stocastic gradient descent']

    def __str__(self):
        return self.name[0]

    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)

class Adam(Optimizer):
    name = ['adam']

    def __str__(self):
        return self.name[0]

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-07):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

# Loss
class LossReduction(Enum):
    SUM_OVER_BATCH_SIZE = 1,
    SUM = 2

    @staticmethod
    def name(loss_reduction):
        name = {
            LossReduction.SUM_OVER_BATCH_SIZE : ['sum over batch size', 'sumoverbatchsize'],
            LossReduction.SUM : ['sum']
        }
        return name[loss_reduction]

class Loss:
    def __init__(self, reduction=LossReduction.SUM_OVER_BATCH_SIZE):
        self.reduction = reduction

class SparseCategoricalCrossentropy(Loss):
    name = [
        'sparse categorical crossentropy', 'sparsecategoricalcrossentropy', 'sparsecce'
    ]

    def __str__(self):
        return self.name[0]

    def __init__(self, reduction=LossReduction.SUM_OVER_BATCH_SIZE, from_logits=False):
        super().__init__(reduction)
        self.from_logits = from_logits

class CategoricalCrossentropy(Loss):
    name = ['categorical crossentropy', 'categoricalcrossentropy', 'cce']

    def __str__(self):
        return self.name[0]

    def __init__(self, reduction=LossReduction.SUM_OVER_BATCH_SIZE, from_logits=False):
        super().__init__(reduction)
        self.from_logits = from_logits

class MeanSquaredError(Loss):
    name = ['mean squared error', 'mse', 'meansquarederror']

    def __str__(self):
        return self.name[0]

    def __init__(self, reduction=LossReduction.SUM_OVER_BATCH_SIZE):
        super().__init__(reduction)

# Training Information
class TrainingInfo:
    def __init__(self, optimizer: Optimizer, loss: Loss, batch_size: int):
        self.optimizer = optimizer
        self.loss = loss
        self.batch_size = batch_size
