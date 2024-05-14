from lib.utils import *
from schema import circle_traininfo_generated as ctr_gen


def load_optimizer(opt_obj: dict):
    opt_type = opt_obj["type"]
    opt_args = opt_obj["args"]

    names_of = OptimizerNamer(all=True)
    supported_opt = [ctr_gen.SGDOptionsT, ctr_gen.AdamOptionsT]

    # convert opt_type(e.g. "sgd") to *OptionsT(e.g. SGDOptionsT)
    load_type = None
    for t in supported_opt:
        if opt_type.lower() in names_of(t):
            load_type = t
            break

    if load_type == None:
        raise ValueError(f"not supported optimizer.type={opt_type}")

    return generate_optimizer(load_type, opt_args)


def load_lossfn(loss_obj: dict):
    loss_type = loss_obj["type"]
    loss_args = loss_obj["args"].copy()
    loss_args.pop("reduction")

    names_of = LossNamer(all=True)
    supported_loss = [
        ctr_gen.SparseCategoricalCrossentropyOptionsT,
        ctr_gen.CategoricalCrossentropyOptionsT, ctr_gen.MeanSquaredErrorOptionsT
    ]

    # convert loss_type(e.g. "mean squared error")
    # to *OptionsT(e.g. ctr_gen.MeanSquaredErrorOptionsT)
    load_type = None
    for t in supported_loss:
        if loss_type.lower() in names_of(t):
            load_type = t
            break

    if load_type == None:
        raise ValueError(f"not supported loss.type={loss_type}")

    return generate_lossfn(load_type, loss_args)


def load_loss_reduction(s: str):

    names_of = LossReductionNamer(all=True)
    supported_rdt = [
        ctr_gen.LossReductionType.SumOverBatchSize, ctr_gen.LossReductionType.Sum
    ]

    load_type = None
    for t in supported_rdt:
        if s.lower() in names_of(t):
            load_type = t
            break

    if load_type == None:
        raise ValueError(f"not supported loss.args.reduction={s}")

    return load_type
