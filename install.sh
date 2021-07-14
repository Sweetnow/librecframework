#!/bin/bash

set -e

PATH=$PATH:~/.local/bin/
pip install -r requirements.txt --user
PYTHONPATH=. pytest
pip install . --user
