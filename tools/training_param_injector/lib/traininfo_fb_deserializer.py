import flatbuffers

from lib.traininfo import *
from circle_schema import circle_traininfo_generated as fb #flatbuffer

class TinfoFBDeserializer:
  '''
  Training Information Flatbuffer Deserializer
  '''

  def __init__(self, buffer):
    '''
    buffer: serialized byte array of training information
    '''
    fb_obj = fb.ModelTraining.GetRootAs(buffer)

    fb_optimizer = fb.OptimizerOptionsCreator(fb_obj.OptimizerOptType(),fb_obj.OptimizerOpt())

    # optimizer_obj = self.__deserialize_opt(fb_obj.Optimizer(), fb_obj.OptimizerOpt())
    # loss_rdt = self.__deserialize_rdt(fb_obj.LossReductionType(), fb_obj.Loss)
    # loss_obj = self.__deserialize_loss(fb_obj.Lossfn(), fb_obj.LossfnOpt())
    # batch_size = fb_obj.BatchSize()
    
    # self.tinfo = TrainingInfo(optimizer_obj, loss_obj, batch_size)


  def __deserialize_opt:

  '''
  def __deserialize_opt 

  def __deserialize_opt(opt : fb.Optimizer, opt_option: fb.OptimizerOptions) -> Optimizer:
    if opt_option:
      fb_sgd = fb.SGDOptions.GetRootAs(opt_option)
      return SGD(fb_sgd.LearningRate()) 
    elif opt == fb.Optimizer.ADAM:
      fb_adam = fb.AdamOptions.GetRootAs(opt_option)
      return Adam(fb_adam.LearningRate(), fb_adam.Beta1(), fb_adam.Beta2(), fb_adam.Epsilon())
    else:
      raise ValueError(f'unknown optimizer :{opt}')

  def deserialize_rdt(rdt : fb.LossReductionType)-> LossReduction:
    if rdt == fb.LossReductionType.SumOverBatchSize:
      return LossReduction.SUM_OVER_BATCH_SIZE
    elif rdt == fb.LossReductionType.Sum:
      return LossReduction.SUM
    else:
      raise ValueError(f'unknown loss reduction type: {rdt}')

  def deserialize_loss(loss: fb.LossFn, loss_opt : fb.LossFnOptions) -> Loss:
    if loss == fb.LossFn.SPARSE_CATEGORICAL_CROSSENTROPY:
      fb_scc = fb.SparseCateogircalCrossEntropy()
      return SparseCategoricalCrossentropy(from_logits=loss_opt.)
    elif loss == fb.LossFn.CATEGORICAL_CROSSENTROPY:
    elif loss == fb.LossFn.MEAN_SQUARED_ERROR:
    else: 
      raise ValueError(f'unknown loss fn : {loss}')

  def get_obj(self):
    return self.tinfo
  ''' 
