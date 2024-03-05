'''
Convert Json object into TrainInfo
'''

from json import JSONDecoder
from . import train_info as tinfo

class JDecoder(JSONDecoder):
    def __init__(self, *args, **kwargs):
        JSONDecoder.__init__(self, object_hook=self.__decode, *args, **kwargs)

    def __decode_optimizer(opt_obj: dict) -> tinfo.Optimizer:
        opt_type = opt_obj["type"]
        opt_args = opt_obj["args"]

        if (opt_type.lower() in tinfo.SGD.name):
            return tinfo.SGD(**opt_args)
        elif (opt_type.lower() in tinfo.Adam.name):
            return tinfo.Adam(**opt_args)
        else:
            raise ValueError(f"not supported optmizer.type={opt_type}")


    def __decode_loss_rdt(s: str) -> tinfo.LossReduction:
        if (s.lower() in tinfo.LossReduction.name(tinfo.LossReduction.SUM_OVER_BATCH_SIZE)):
            return tinfo.LossReduction.SUM_OVER_BATCH_SIZE
        elif (s.lower() in tinfo.LossReduction.name(tinfo.LossReduction.SUM)):
            return tinfo.SUM
        else:
            raise ValueError(f"not supported loss.args.reduction={s}")


    def __decode_loss(self, loss_obj: dict) -> tinfo.Loss:
        loss_type = loss_obj["type"]
        loss_args = loss_obj["args"]

        # update reduction string into corresponded enum
        if ("reduction" in loss_args.keys()):
            loss_args["reduction"] = self.__decode_loss_rdt(loss_args["reduction"])

        if (loss_type.lower() in tinfo.SparseCategoricalCrossentropy.name):
            return tinfo.SparseCategoricalCrossentropy(**loss_args)
        elif (loss_type.lower() in tinfo.CategoricalCrossentropy.name):
            return tinfo.CategoricalCrossentropy(**loss_args)
        elif (loss_type.lower() in tinfo.MeanSquaredError.name):
            return tinfo.MeanSquaredError(**loss_args)
        else:
            raise ValueError(f"not supported loss.type={loss_type}")

    def __decode(self, obj):
        optimizer = self.__decode_optimizer(obj["optimizer"])
        loss = self.__decode_loss(obj["loss"])
        batch_size = obj["batch_size"]
        return tinfo.TrainingInfo(optimizer, loss, batch_size)

