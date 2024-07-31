#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate SA_min1
python train.py --SR --real
