#!/bin/bash

python3 experiment-shifted-decision-boundary.py > results/sdb.txt &
python3 experiment-random-relabeling.py > results/rr.txt &
python3 experiment-random-massaging.py > results/rm.txt &
python3 experiment-fair-weak-learner.py > results/fwl.txt &

