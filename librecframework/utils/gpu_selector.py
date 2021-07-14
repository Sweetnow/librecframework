#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from typing import List
import logging
import time
import numpy as np
from pynvml import nvmlDeviceGetCount, nvmlInit, nvmlDeviceGetHandleByIndex, \
    nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, nvmlShutdown

__all__ = ['autoselect']

COUNT = 10
INTERVAL = 0.1

if sys.platform.startswith('linux'):
    def autoselect(gpu_target: List[int], min_memory: float) -> int:
        logging.info(f'GPU search space: {gpu_target}')
        nvmlInit()
        deviceCount = nvmlDeviceGetCount()
        memories = np.zeros((deviceCount, COUNT), dtype=np.float32)
        rates = np.zeros((deviceCount, COUNT), dtype=np.float32)
        for c in range(COUNT):
            for i in range(deviceCount):
                if i not in gpu_target:
                    memories[i, c] = 0
                    rates[i, c] = 100
                else:
                    handle = nvmlDeviceGetHandleByIndex(i)
                    memories[i, c] = nvmlDeviceGetMemoryInfo(
                        handle).free / 1024**3
                    rates[i, c] = int(
                        nvmlDeviceGetUtilizationRates(handle).gpu)
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
            gpuid = gpu_min_ids[np.argmax(memories[gpu_min_ids])]
            logging.info(f'Auto select GPU {gpuid}')
        else:
            raise MemoryError(str(memories))
        return int(gpuid)
else:
    def autoselect(gpu_target: List[int], min_memory: float) -> int:
        return 0


if __name__ == "__main__":
    print(autoselect([0, 1, 2, 3, 4, 5, 6, 7], 0))
