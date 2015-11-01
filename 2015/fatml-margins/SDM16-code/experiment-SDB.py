#!/usr/bin/env python3

import random
import boosting
import svm
import lr
from data import adult, german, singles
from weaklearners.decisionstump import buildDecisionStump
from errorfunctions import signedStatisticalParity, labelError, individualFairness
from utils import errorBars, arrayErrorBars, sign, variance, experimentCrossValidate
from margin import svmRBFMarginAnalyzer, svmLinearMarginAnalyzer, boostingMarginAnalyzer, lrSKLMarginAnalyzer


def lrLearner(train, protectedIndex, protectedValue):
   marginAnalyzer = lrSKLMarginAnalyzer(train, protectedIndex, protectedValue)
   shift = marginAnalyzer.optimalShift()
   print('best shift is: %r' % (shift,))
   return marginAnalyzer.conditionalShiftClassifier(shift)


def boostingLearner(train, protectedIndex, protectedValue):
   marginAnalyzer = boostingMarginAnalyzer(train, protectedIndex, protectedValue)
   shift = marginAnalyzer.optimalShift()
   print('best shift is: %r' % (shift,))
   return marginAnalyzer.conditionalShiftClassifier(shift)


def svmLearner(train, protectedIndex, protectedValue):
   marginAnalyzer = svmRBFMarginAnalyzer(train, protectedIndex, protectedValue)
   shift = marginAnalyzer.optimalShift()
   print('best shift is: %r' % (shift,))
   return marginAnalyzer.conditionalShiftClassifier(shift)


def svmLinearLearner(train, protectedIndex, protectedValue):
   marginAnalyzer = svmLinearMarginAnalyzer(train, protectedIndex, protectedValue)
   shift = marginAnalyzer.optimalShift()
   print('best shift is: %r' % (shift,))
   return marginAnalyzer.conditionalShiftClassifier(shift)


@arrayErrorBars(2)
def statistics(train, test, protectedIndex, protectedValue, learner):
   h = learner(train, protectedIndex, protectedValue)
   print("Computing error")
   error = labelError(test, h)
   print("Computing bias")
   bias = signedStatisticalParity(test, protectedIndex, protectedValue, h)
   print("Computing UBIF")
   ubif = individualFairness(train, learner, 0.2, passProtected=True)
   return error, bias, ubif


@errorBars(10)
def indFairnessStats(train, learner):
   print("Computing UBIF")
   ubif = individualFairness(train, learner, flipProportion=0.2, passProtected=True)
   print("UBIF:", ubif)
   return ubif

def experiment(dataModule, learner):
   train, test = dataModule.load()
   PI = dataModule.protectedIndex
   PV = dataModule.protectedValue
   output = statistics(train, test, PI, PV, learner)
   print("\tavg, min, max, variance")
   print("error: %r" % (output[0],))
   print("bias: %r" % (output[1],))
   print("ubif: %r" % (output[2],))

def runAll():
   print("Shifted Decision Boundary Relabeling")
   experiments = [
      (('SVM', svmSKL), adult),
      (('SVMlinear', svmLinearSKL), german),
      (('SVM', svmSKL), singles),
      (('AdaBoost', boost), adult),
      (('AdaBoost', boost), german),
      (('AdaBoost', boost), singles),
      (('LR', lrSKL), adult),
      (('LR', lrSKL), german),
      (('LR', lrSKL), singles),
   ]

   for (learnerName, learner), dataset in experiments:
      print("%s %s" % (dataset.name, learnerName), flush=True)
      experimentCrossValidate(dataset, learner)


if __name__ == '__main__':
  runAll()
