import re

from typing import Union, Type, Tuple
from schema import circle_traininfo_generated as ctr_gen


class OptimizerNamer:
    '''Return name(string) based on ModelTraining.OptimizerOpt'''
    names = {
        ctr_gen.SGDOptionsT: ['sgd', 'stocasticgradientdescent'],
        ctr_gen.AdamOptionsT: ['adam']
    }

    def __init__(self, all=False):
        ''' If all = Ture, return all possible names
            Otherwise, return one representative name
        '''
        self.all = all

    def __call__(self, opt):
        try:
            names = self.names[opt]
        except:
            print(f"unknown optimizer {type(opt)}")

        if self.all:
            return names
        else:
            return names[0]


class LossNamer:
    '''Return name(string) based on ModelTraining.LossfnOpt'''

    # yapf:disable
    names = {
        ctr_gen.SparseCategoricalCrossentropyOptionsT:
            ['sparse categorical crossentropy', 'sparsecategoricalcrossentropy', 'sparsecce'],
        ctr_gen.CategoricalCrossentropyOptionsT:
            ['categorical crossentropy', 'categoricalcrossentropy', 'cce'],
        ctr_gen.MeanSquaredErrorOptionsT:
            ['mean squared error', 'mse']
    }
    # yapf:eanble

    def __init__(self, all=False):
        self.all = all

    def __call__(self, lossfn):
        try:
            names = self.names[lossfn]
        except:
            print(f"unknown lossfn {type(lossfn)}")

        if self.all:
            return names
        else:
            return names[0]


class LossReductionNamer:
    '''Return name(string) based on ModelTraining.LossReductionType '''
    names = {
        ctr_gen.LossReductionType.SumOverBatchSize: ['sum over batch size', 'sumoverbatchsize'],
        ctr_gen.LossReductionType.Sum: ['sum'],
    }

    def __init__(self, all=False):
        self.all = all

    def __call__(self, rdt):
        try:
            names = self.names[rdt]
        except:
            print(f"unknown loss reduction type {rdt}")

        if self.all:
            return names
        else:
            return names[0]


OPT_OPTIONS_T = Union[Type[ctr_gen.SGDOptionsT], Type[ctr_gen.AdamOptionsT]]


# yapf:disable
def generate_optimizer(
        opt_type: OPT_OPTIONS_T,
        args: dict
        ) -> Tuple[ctr_gen.Optimizer, ctr_gen.OptimizerOptions, OPT_OPTIONS_T]:
    '''
    Generates objects for circle_traininfo_generated.ModelTrainingT.[optimizer, optimizerOptType, OptimizerOpt]
    '''
    #yapf:enable
    options_t_str: str = opt_type.__name__  #SGDOptionsT, AdamOptionsT
    options_str: str = options_t_str[:-1]  #SGDOptions, AdamOptions
    optimizer_str: str = options_str.replace("Options", "").upper()  #SGD, ADAM

    optimizer = getattr(ctr_gen.Optimizer, optimizer_str)
    optimizer_opt_type = getattr(ctr_gen.OptimizerOptions, options_str)

    optimizer_opt = opt_type()
    for (key, value) in args.items():
        setattr(optimizer_opt, key, value)

    return optimizer, optimizer_opt_type, optimizer_opt


# yapf:disable
LOSSFN_OPTIONS_T = Union[Type[ctr_gen.SparseCategoricalCrossentropyOptionsT],
                         Type[ctr_gen.CategoricalCrossentropyOptionsT],
                         Type[ctr_gen.MeanSquaredErrorOptionsT]]
# yapf:enable


# yapf:disable
def generate_lossfn(
        lossfn_type: LOSSFN_OPTIONS_T,
        args: dict
     ) -> Tuple[ctr_gen.LossFn, ctr_gen.LossFnOptions, LOSSFN_OPTIONS_T]:
    '''
    Generate objects for circle_traininfo_generated.ModelTrainingT.[lossfn, lossfnOptType, lossfnOpt]
    '''
    # yapf:eanble

    options_t_str: str = lossfn_type.__name__   # CategoricalCrossentropyOptionsT
    options_str: str = options_t_str[:-1]       # CategoricalCrossentropyOptions
    lossfn_camel: str = options_str.replace("Options","")  # CategoricalCrossentropy
    lossfn_str: str = re.sub(r'(?<!^)(?=[A-Z])', '_',lossfn_camel).upper()  # CATEGORICAL_CROSSENTROPY

    lossfn = getattr(ctr_gen.LossFn, lossfn_str)
    lossfn_opt_type = getattr(ctr_gen.LossFnOptions, options_str)

    lossfn_opt = lossfn_type()
    for (key, value) in args.items():
        setattr(lossfn_opt, key, value)

    return lossfn, lossfn_opt_type, lossfn_opt
