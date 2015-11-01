#!/bin/bash

python3 -m baselines.adult > results/adult-baseline.txt &
python3 -m baselines.german > results/german-baseline.txt &
python3 -m baselines.singles > results/singles-baseline.txt &

