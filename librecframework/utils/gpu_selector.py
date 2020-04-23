#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List
import logging
import time
from collections import namedtuple
import numpy as np
from pynvml import *

__all__ = ['autoselect']

COUNT = 10
INTERVAL = 0.1

def autoselect(gpu_target: List[int], min_memory: float):
    logging.info(f'GPU search space: {gpu_target}')
    nvmlInit()
    deviceCount = nvmlDeviceGetCount()
    memories = np.zeros((deviceCount, COUNT), dtype=np.float)
    rates = np.zeros((deviceCount, COUNT), dtype=np.float)
    for c in range(COUNT):
        for i in range(deviceCount):
            if i not in gpu_target:
                memories[i, c] = 0
                rates[i, c] = 100
            else:
                handle = nvmlDeviceGetHandleByIndex(i)
                memories[i, c] = nvmlDeviceGetMemoryInfo(handle).free / 1024**3
                rates[i, c] = int(nvmlDeviceGetUtilizationRates(handle).gpu)
        time.sleep(INTERVAL)
    nvmlShutdown()
    memories = memories.mean(1)
    rates = rates.mean(1)
    # enough memory GPU ids
    memory_enough_ids = np.where(memories > min_memory)[0]
    if len(memory_enough_ids) > 0:
        # min util GPU
        gpuid = memory_enough_ids[np.argmin(rates[memory_enough_ids])]
        # if multi GPUs' util are the same, choose one that has the most memory
        gpu_min_ids = np.where(rates[memory_enough_ids] <= rates[gpuid])[0]
        gpu_min_ids = memory_enough_ids[gpu_min_ids]
        gpuid = gpu_min_ids[np.argmin(memories[gpu_min_ids])]
        logging.info(f'Auto select GPU {gpuid}')
    else:
        raise MemoryError(str(memories))
    return int(gpuid)

if __name__ == "__main__":
    print(autoselect([0,1,2,3,4,5,6,7], 2))
