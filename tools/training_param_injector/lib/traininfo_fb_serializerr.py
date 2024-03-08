import flatbuffers

from lib.traininfo import *
from circle_schema.circle_traininfo_generated import *


class TinfoFBSerializer:
    '''
    Training Inforamtion FlatBuffer Serializer
    '''
    TRAINING_FILE_IDENTIFIER = b"CTR0"

    def __init__(self, info: TrainingInfo, buf_size=0):
        self.builder = flatbuffers.Builder(buf_size)
        train_info = self.__build_train_info(info)
        self.builder.Finish(train_info, self.TRAINING_FILE_IDENTIFIER)

    def __build_optimizer(self, optimizer: Optimizer):

        if isinstance(optimizer, SGD):
            SGDOptionsStart(self.builder)
            SGDOptionsAddLearningRate(self.builder, optimizer.learning_rate)
            sgd_option = SGDOptionsEnd(self.builder)
            return Optimizer.SGD, ctr.OptimizerOptions.SGDOptions, sgd_option

        elif isinstance(optimizer, Adam):
            AdamOptionsStart(self.builder)
            AdamOptionsAddBeta1(self.builder, optimizer.beta1)
            AdamOptionsAddBeta2(self.builder, optimizer.beta2)
            AdamOptionsAddEpsilon(self.builder, optimizer.epsilon)
            AdamOptionsAddLearningRate(self.builder, optimizer.learning_rate)
            adam_option = AdamOptionsEnd(self.builder)
            return Optimizer.ADAM, ctr.OptimizerOptions.AdamOptions, adam_option

        else:
            raise ValueError(f"unknown optimizer: {type(optimizer)}")

    def __build_loss(self, loss: Loss):
        if isinstance(loss, SparseCategoricalCrossentropy):
            SparseCategoricalCrossentropyOptionsStart(self.builder)
            SparseCategoricalCrossentropyOptionsAddFromLogits(
                self.builder, loss.from_logits)
            sparse_cce = SparseCategoricalCrossentropyOptionsEnd(self.builder)
            return LossFn.SPARSE_CATEGORICAL_CROSSENTROPY, ctr.LossFnOptions.SparseCategoricalCrossentropyOptions, sparse_cce

        elif isinstance(loss, CategoricalCrossentropy):
            CategoricalCrossentropyOptionsStart(self.builder)
            CategoricalCrossentropyOptionsAddFromLogits(self.builder,
                                                            loss.from_logits)
            cce = CategoricalCrossentropyOptionsEnd(self.builder)
            return LossFn.CATEGORICAL_CROSSENTROPY, ctr.LossFnOptions.CategoricalCrossentropyOptions, cce

        elif isinstance(loss, MeanSquaredError):
            MeanSquaredErrorOptionsStart(self.builder)
            mse = MeanSquaredErrorOptionsEnd(self.builder)
            return LossFn.MEAN_SQUARED_ERROR, ctr.LossFnOptions.MeanSquaredErrorOptions, mse

        else:
            raise ValueError(f"unknown loss: {type(loss)}")

    def __build_loss_rdt(self, loss_rdt: LossReduction):
        if loss_rdt == LossReduction.SUM_OVER_BATCH_SIZE:
            return LossReductionType.SumOverBatchSize
        elif loss_rdt == LossReduction.SUM:
            return LossReductionType.Sum
        else:
            raise ValueError(f"unkonw loss reduction: {loss_rdt}")

    def __build_train_info(self, info: TrainingInfo):

        optimizer_args = self.__build_optimizer(info.optimizer)
        loss_args = self.__build_loss(info.loss)
        loss_rdt = self.__build_loss_rdt(info.loss.reduction)

        ModelTrainingStart(self.builder)

        # optimizer
        optimizer, optimizer_opt_t, optimizer_opt = optimizer_args
        ModelTrainingAddOptimizer(self.builder, optimizer)
        ModelTrainingAddOptimizerOptType(self.builder, optimizer_opt_t)
        ModelTrainingAddOptimizerOpt(self.builder, optimizer_opt)

        # loss
        loss, loss_opt_t, loss_opt = loss_args
        ModelTrainingAddLossfn(self.builder, loss)
        ModelTrainingAddLossfnOptType(self.builder, loss_opt_t)
        ModelTrainingAddLossfnOpt(self.builder, loss_opt)

        # TODO: removed this line after Epoch removed from *.fbs
        ModelTrainingAddEpochs(self.builder, 0)

        # others
        ModelTrainingAddBatchSize(self.builder, info.batch_size)
        ModelTrainingAddLossReductionType(self.builder, loss_rdt)

        model_training = ModelTrainingEnd(self.builder)
        return model_training

    def get_buff(self):
        return self.builder.Output()
