# Library of Recommender System based on PyTorch
## Introduction

A Python package to integrate the pipeline of recommender systems for simple model designing and fast idea verification.

### Installation

```bash
bash install.sh
```

### Requirement
0. `Python` >= 3.8
1. `PyTorch` >= 1.2.0
2. `visdom` == 0.1.8.9 (for visualization)
3. `nvidia_ml_py3` == 7.352.0 (for auto-selection of GPU)
4. Negative sampling recommender systems

### Advantage

1. Separate model design from other dirty things like CLI, logging, GPU selection, etc.
2. Provide two general methods to save and visualize key values in the pipeline, `TrainHook` and `Metric`.
3. Provide widely used loss functions and model design patterns in recommendation, supporting customization easily.

## Submodules

### analysis

The submodule is to automatically analyse the dataset or the model results for paper presentation. Now the submodule only supports:

- `multimodel.py`: organize models' metrics and report them as a csv table

### argument



### data



### utils



### logger.py

The submodule is to easily record the hyperparameters setting, the environment setting (like optimizer) and all related values of each model into specific path.

- `Logger`: the class to implement the function above

### loss.py

The submodule is to contain some commonly used loss functions in recommendation, including `L2loss`, `BPR`, `MSE with mask`, `BCE`.

### metric.py

The submodule is to provide a general way to save metrics in model evaluation and visualize them by `Visdom`. We provide `Precision` `Recall` `NDCG` `MRR` for fully-ranking and leave-one-out mode (in leave-one-out mode, `Precision` `Recall`->`HR`)

### model.py

The submodule is to provide some base class for models.

- `Model`: the abstract class for all models in our framework
- `EmbeddingBasedModel`: the abstract class for embedding based models, which provide embedding generation
- `DotBasedModel`: the abstract class for dot based models, which simplify `forward`, `before_evaluation` and `evaluation` and provide `propagate` interface

### pipeline.py



### test.py

The submodule is to evaluate model in fully-ranking or leave-one-out mode by `fully_ranking_test` or `leave_one_out_test`, respectively.

### train.py

The submodule is to train model.

### trainhook.py

The submodule is to provide a general way to save key values in model `forward` phase and visualize them as metrics. The trainhooks will be found in `model.trainhooks` as dict, whose key is the title of trainhook (`__loss__` is used for recording loss of each epoch).

- `Trainhook`: the base class for trainhook, providing interface `start()` `stop()` `__call__()` `title` `value`

- `ValueMeanHook`: the implementation of trainhook, which calculate the meaning of each value

## Example

### Configuration

`config.json`

```json
{
    "user": "name_for_setproctitle",
    "visdom": {
        "server": "visdom_ip",      # use empty string to disable visdom visualization
        "port": {
            "dataset_name1": 10001,
            "dataset_name2": 10002
        }
    },
    "training": {
        "test_interval": 5,
        "early_stop": 50,
        "overfit": {
            "protected_epoch": 10,
            "threshold": 0.1
        }
    },
    "dataset": {
        "path": "path_to_dataset",
        "seed": 123,
        "use_backup": true
    },
    "logger": {
        "path": "path_to_log",
        "policy": "best"
    },
    "metric": {
        "target": {
            "type": "NDCG",
            "topk": 10
        },
        "metrics": [
            {
                "type": "Recall",
                "topk": 5
            },
            {
                "type": "Recall",
                "topk": 10
            },
            {
                "type": "NDCG",
                "topk": 5
            },
            {
                "type": "NDCG",
                "topk": 10
            }
        ]
    }
}

```

- `user`: One part of processes name when running the codes.

#### Visdom

`visdom` part in `config.json`

- `server`: The IP of visdom server. Set it to `""` to disable visualization.
- `port`: Key-value pairs, which key is dataset name and value is the port of visdom server.

#### Training

`training` part in `config.json`

- `test_interval`: The interval epoch for testing.
- `early_stop`: Stop training when the target metric drops `early_stop` times.
- `overfit`: overfitting detection
    - `protected_epoch`: Disable overfitting detection for the first `protected_epoch` epochs.
    - `threshold`: Stop training when the target metric is less than `threshold`, which means the model is overfitting.

#### Dataset

`dataset` part in `config.json`

- `path`: The root folder path of all datasets.
- `seed`: The seed for negative sampling.
- `use_backup`: Save/load the negative sampling results into/from file or not.

#### Logger

`logger` part in `config.json`

- `path`: The root folder path for log files.
- `policy`: [Do not modify] When to save the model as checkpoint.

#### Metric

`metric` part in `config.json`

Each metric is described by
```json
{
    "type": "MetricName",
    "topk": X
}
```
, which means `MetricName@X` (e.g. NDCG@10, Recall@5, MRR@40).

- `target`: Metric used for early stopping, overfitting detection and so on.
- `metrics`: The list of metrics which should be computed and recorded.

### Pretrain Configuration

`pretrain.json`

```json
{
    "BeiBei": {
        "GBMF": "path-to-GBMF-pretrain-model"
    }
}
```

The key of the first layer is dataset name.

The key of the second layer is pretrain model name and the value is the path.

According to dataset used for training or testing, the second layer will be sended into `model.load_pretrain` as python dict `pretrain_info`.

## Roadmap

- [ ] Automatic analysis
  - [ ] Performance evaluation for data sparsity issue
  - [ ] Dataset profile
- [ ] Pass static check by complete type hints
- [ ] Support more activation functions
- [ ] Add `DGL` module support
- [ ] Convergence judgment based on linear fitting instead of early-stopping
- [ ] Replace `Visdom` to `Tensorboard` for visualization of training
- [ ] Avoid same-id error in metric and speed up by sorting top-k, providing context manager
