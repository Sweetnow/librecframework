# Library of Recommender System based on PyTorch
## Introduction

A Python package to integrate the pipeline of recommender systems for simple model designing and fast idea verification.

### Installation

```bash
pip install . --user
```

### Requirement

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

- `multimodel.py`: organize models' metrics and report them as a csv table.

### argument



### data



### utils

The submodule is to contain utility for the library:

- `convert.py`: convert one kind of objects to  another, like `[name, topk]->Metric` `path->save_model` `scipy.sparse->torch` and so on.
- `gpu_selector.py`: select which GPU to use by the GPU memory requirement of model and the current usage.
- `graph_generation.py`: generate adjacency matrix of both `p` and `q` from p-p adjacency matrix, q-q adjacency matrix and p-q adjacency matrix.
- `training.py`: utility for training, like early-stopping.
- `visshow.py`: a simple wrap for `Visdom`.

### logger.py

The submodule is to easily record the hyperparameters setting, the environment setting (like optimizer) and all related values of each model into specific path.

- `Logger`: the class to implement the function above.

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

```json
{
    "user": "name_for_setproctitle",
    "visdom": {
        "server": "visdom_ip",
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

### MF



## Roadmap

- [ ] Automatic analysis
  - [ ] Performance evaluation for data sparsity issue
  - [ ] Dataset profile
- [x] Pass static check by complete type hints
- [ ] Support more activation functions
- [ ] Convergence judgment based on linear fitting instead of early-stopping
- [ ] Replace `Visdom` to `Tensorboard` for visualization of training
- [ ] Avoid same-id error in metric and speed up by sorting top-k, providing context manager