# How To Write Hyperparameters JSON file

The Hyperparameters JSON file is designed to store hyperparameters which must be configured in circle file for training. <br/>
For quicker understanding, please refer to the examples([#1](./example/train_tparam.json), [#2](./example/tparam_sgd_scce.json)) of the JSON file. 
<br/>

The json file consists of a single JSON object containing the following keys: 
- [optimizer](#optimizer)
- [loss](#loss)
- [batchSize](#batchsize)

## optimizer

An object describing optimization algorithm. This should include two keys : 

* `type` : a string to indiciate optimizer (e.g. `adam`) 
* `args` : additional arguments specific to the chosen optimizer. These may be varyinng depending on the optimizer type, but typically include 'learningRate' value. 

**Supported Optimizers:**

  | Supported Optimizer | Key                    |Data Type     | Example Values |
  |---------------------|-----------------       |--------      |----------------|
  | Adam                | type                   |string        | "adam"         |
  |                     | args                   |object        |                |
  |                     | &ensp; \- learningRate |number        | 0.01           |
  |                     | &ensp; \- beta1        |number        | 0.9            |
  |                     | &ensp; \- beta2        |number        | 0.999          |
  |                     | &ensp; \- epsilon      |number        | 1e-07          |
  |                                                                              |
  | SGD                 | type                   |string        | "sgd"          |
  |                     | args                   |object        |                |
  |                     | &ensp; \- learningRate |number        | 0.001          |


## loss

An object describing the loss function. This should include two keys :

* `type`: a string indicating which loss function to use. (e.g. `mse`)
* `args`: additional arguments specific to the chosen loss function. These may vary depending on the loss function type, but typically include 'reduction'. 

**Supported Loss Functions:**

  | Supported Loss Function           | Key                   |Data Type | Example Values                     |
  |-----------------------------------|---------------------- |----------|----------------                    |
  | Sparse Categorical Cross Entropy  | type                  |string    | "sparse categorical crossentropy"  |
  |                                   | args                  |object    |                                    |
  |                                   | &ensp;\- fromLogits   |boolean   | true, false                        |
  |                                   | &ensp;\- reduction    |string    | "sum over batch size,"sum"         |
  |                                                                                                           |
  |                                                                                                           |
  | Categorical Cross Entropy         | type                  |string    | "categorical crossentropy"         |
  |                                   | args                  |object    |                                    |
  |                                   | &ensp;\- fromLogits   |boolean   | true, false                        |
  |                                   | &ensp;\- reduction    |string    | "sum over batch size", "sum"       |
  |                                                                                                           |
  | Mean Squared Error                | type                  |string    | "mean squared error"               |
  |                                   | args                  |empty object | {}                              |

## batchSize

A number of examples processeed during each iteration.
