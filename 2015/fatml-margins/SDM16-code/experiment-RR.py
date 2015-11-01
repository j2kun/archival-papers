#!/usr/bin/env python3

from data import adult, german, singles
from relabeling import randomOneSideRelabelData
import boosting
import svm
import lr
from errorfunctions import signedStatisticalParity, labelError, individualFairness
from utils import arrayErrorBars, errorBars, experimentCrossValidate

def boostingLearner(data, protectedIndex, protectedValue):
   h = boosting.boost(data)
   return randomOneSideRelabelData(h, data, protectedIndex, protectedValue)


def svmLearner(data, protectedIndex, protectedValue):
   h = svm.svmSKL(data, verbose=True)
   return randomOneSideRelabelData(h, data, protectedIndex, protectedValue)


def lrLearner(data, protectedIndex, protectedValue):
   h = lr.lrSKL(data)
   return randomOneSideRelabelData(h, data, protectedIndex, protectedValue)


@arrayErrorBars(2)
def statistics(train, test, protectedIndex, protectedValue, learner):
   h = learner(train, protectedIndex, protectedValue)
   print("Computing error")
   error = labelError(test, h)
   print("Computing bias")
   bias = signedStatisticalParity(test, protectedIndex, protectedValue, h)
   print("Computing UBIF")
   ubif = individualFairness(train, learner, flipProportion=0.2, passProtected=True)
   return error, bias, ubif

@errorBars(10)
def indFairnessStats(train, learner):
   print("Computing UBIF")
   ubif = individualFairness(train, learner, flipProportion=0.2, passProtected=True)
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
   print("Random Relabeling")
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
      print("%s %s" % (dataset.name, learnerName))
      experimentCrossValidate(dataset, learner, 5, statistics)


if __name__ == '__main__':
  runAll()
