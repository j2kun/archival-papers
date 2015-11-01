# Code and experiments for "A confidence-based approach for balancing fairness and accuracy"

All experiments used in this paper were implemented in Python 3 with the following
dependencies

    numpy
    matplotlib
    scikitlearn


## One-click rerun of all experiments

To re-run all experiments used in the paper, run the following from the command line

    ./run-all.sh

This will re-run all the experiments and output the data to plaintext
files in the results/ subdirectory.

To generate all plots used in the paper, run the following from the command line:

    python plot-all.py

To run the baselines (learning algorithms with no modifications), run

    ./run-baselines.sh


## Datasets

The datasets are given the following names

    adult
    german 
    singles 

### Loading into Python

For each dataset there is a data loader module and a baseline (see the
Baselines section below). We will use `adult` as the prototype, and unless
otherwise stated all datasets operate the same way with `adult` replaced by the
dataset name. The raw data files are `adult.train` and `adult.test`. If
preprocessing occurred to split a dataset into training and testing subsets,
then the unprocessed data files are in the `preprocessed/` subdirectory along
with python scripts to perform the (randomized) preprocessing. Additional
preprocessing is performed to turn categorical features into (possibly many)
binary features.

To load a dataset, you can run the following commands from the base directory
of the project.

    $ python
    Python 3.3.3 (default, Dec 30 2013, 23:51:18) 
    [GCC 4.2.1 Compatible Apple LLVM 5.0 (clang-500.2.79)] on darwin
    >>> from data import adult
    >>> trainingData, testData = adult.load()
    >>> adult.protectedIndex
    1
    >>> len(trainingData)
    32561
    >>> trainingData[0]
    ((39, 1, 0, 0, 0, 0, 0, 1, 0, 0, 13, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2174, 0, 40, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0), -1)


## Baselines

The experiments for the baseline algorithms for each dataset are in the
`baselines/` subdirectory. They can be run by running the following command from
the project base, where `adult` is replaced with the name of the dataset you
want to run.

    $ python -m baselines.adult

This will by default run the vanilla svm/boosting/logisitic regression
algorithms on the adult dataset and measure all of our metrics on each. To run
a subset of these experiments, modify `baselines/adult.py` and/or
`baselines/baseline.py`.

## Experiments

The experiments are organized by method. So random relabeling is in the
experiment-random-relabeling.py file. Each experiment has a runAll() function
that runs all of the experiments for every dataset and learner (SVM, logistic
regression, and AdaBoost). Note that boosting and SVM take ~5-30 minutes per
run on large datasets, and each experiment averages over 10 runs.

## Plots

The main plots in the paper are produced by the MarginAnalyzer class in
margins.py. See the MarginAnalyzer.plotMarginHistogram and
MarginAnalyzer.plotTradeoff functions for details.
