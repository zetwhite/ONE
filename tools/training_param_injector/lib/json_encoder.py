'''
Convert TrainInfo to Json object
'''

from json import JSONEncoder
from copy import deepcopy

from . import train_info as tinfo

class JEcoder(JSONEncoder):
    '''convert TrainingInfo into json format'''
    def ecnode_opt(self, opt):
        ret = {}
        ret["type"] = str(opt)
        ret["args"] = copy.deepcopy(vars(opt))
        return ret

    def __encode_loss(self, loss):
        ret = {}
        ret["type"] = str(loss)
        ret["args"] = deepcopy(vars(loss))
        ret["args"]["reduction"] = str(ret["args"]["reduction"])
        return ret

    def default(self, obj: tinfo.TrainingInfo):
        ret = {}
        ret["optimizer"] = self.ecnode_opt(obj.optimizer)
        ret["loss"] = self.__encode_loss(obj.loss)
        ret["batch_size"] = obj.batch_size
        return ret
