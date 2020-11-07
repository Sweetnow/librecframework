# Library of Recommender System based on PyTorch
![Python package](https://github.com/Sweetnow/librecframework/workflows/Python%20package/badge.svg)
---

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
2. 



## Submodules

### analysis

The submodule is to automatically analyse the dataset or the model results for paper presentation. Now the submodule only supports:

- `multimodel.py`: Organize models' metrics and report them as a csv table

### argument



### data



### utils



### logger.py



### loss.py



### metric.py



### model.py



### pipeline.py



### test.py

The submodule is to evaluate model in fully-ranking or leave-one-out mode by `fully_ranking_test` or `leave_one_out_test`, respectively.

### train.py

FIXME: `trainhooks` should not be here

### trainhood.py





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
- [ ] Pass static check by complete type hints
- [ ] Support more activation functions
- [ ] Add `DGL` module support
- [ ] Convergence judgment based on linear fitting instead of early-stopping
- [ ] Replace `Visdom` to `Tensorboard` for visualization of training