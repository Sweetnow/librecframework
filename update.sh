#!/bin/bash
PATH=$PATH:~/.local/bin/
pipreqs ./ --pypi-server https://pypi.tuna.tsinghua.edu.cn/simple --force
PYTHONPATH=. pytest
if [ $? -ne 0 ]
then
    exit 1
fi
pip install . --user