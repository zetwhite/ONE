from schema.circle_traininfo_generated import *
from . import json_parser
# optimizers
class SGD(SGDOptionsT):
  name = ['sgd', 'stocasticgradientdescent']
  
  def __init__(self, learning_rate=0.01):
    super().__init__()
    self.learningRate = learning_rate
  
class Adam(AdamOptionsT):
  name = ['adam']
  
  def __init__(self, learning_rate = 0.0001, beta1 = 0.9, beta2 = 0.999, epsilon=1e-07):
    super().__init__()
    self.learningRate = learning_rate
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon

# Loss 
class SparseCategoricalCrossEntropy(SparseCategoricalCrossentropyOptionsT):
  name = ['sparse categorical crossentropy', 'sparsecategoricalcrossentropy', 'sparsecce']

  def __init__(self, from_logits=False):
    super().__init__()
    self.fromLogits = from_logits
    

class CategoricalCrossEntropy(CategoricalCrossentropyOptionsT):
  name = ['categorical crossentropy', 'categoricalcrossentropy', 'cce']
  
  def __init__(self, from_logits=False):
    super().__init__()
    self.fromLogits = from_logits

class MeanSqauaredError(MeanSquaredErrorOptionsT):
  name = ['mean squared error', 'mse']
    
  def __init__(self):
    super().__init__() 

# TrainInfo 
class TrainInfo(ModelTrainingT):
    def __init__(self, json_obj:dict):
      super().__init__()
      self.__load_from_json(json_obj) 

    def __load_from_json(self, json_obj:dict):
      opt, opt_type, opt_obj = json_parser.load_optimizer(json_obj["optimizer"])
      self.optimizer = opt
      self.optimizerOptType = opt_type
      self.optimizerOpt = opt_obj

      lossfn, lossfn_type, lossfn_obj = json_parser.load_loss(json_obj["loss"])
      self.lossfn = lossfn
      self.lossfnOptType = lossfn_type
      self.lossfnOpt = lossfn_obj

      self.batchSize = json_obj["batch_size"]

      if "reduction" in json_obj["loss"].keys():
        self.lossReductionType = json_parser.load_loss_rdt(json_obj["loss"]["reduction"] ) 

    def get_buff(self):
      builder = flatbuffers.Builder(0)
      builder.Finish(self.Pack(builder))
      return builder.Output()  
    