#!/usr/bin/env python3

from data import adult, german, singles
from relabeling import randomOneSideRelabelData
import boosting
import svm
import lr
from errorfunctions import signedStatisticalParity, labelError, individualFairness
from utils import arrayErrorBars

def boostingLearner(data, protectedIndex, protectedValue):
   h = boosting.boost(data)
   return randomOneSideRelabelData(h, data, protectedIndex, protectedValue)


def svmLearner(data, protectedIndex, protectedValue):
   h = svm.svmSKL(data, verbose=True)
   return randomOneSideRelabelData(h, data, protectedIndex, protectedValue)

def svmLinearLearner(data, protectedIndex, protectedValue):
   h = svm.svmLinearSKL(data, verbose=True)
   return randomOneSideRelabelData(h, data, protectedIndex, protectedValue)

def lrLearner(data, protectedIndex, protectedValue):
   h = lr.lrSKL(data)
   return randomOneSideRelabelData(h, data, protectedInde, protectedValuex)


@arrayErrorBars(10)
def statistics(train, test, protectedIndex, protectedValue, learner):
   h = learner(train, protectedIndex, protectedValue)
   print("Computing error")
   error = labelError(test, h)
   print("Computing bias")
   bias = signedStatisticalParity(test, protectedIndex, protectedValue, h)
   print("Computing UBIF")
   ubif = individualFairness(train, learner, flipProportion=0.2, passProtected=True)
   return error, bias, ubif


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
   datasets = [
               adult,
               german,
               singles,
               ]

   learners = [
      ('SVMlinear', svmLinearLearner),
      ('LR', lrLearner), 
      #('SVM', svmLearner), 
      ('AdaBoost', boostingLearner),
   ] 

   for dataset in datasets:
      for learnerName, learner in learners:
         print("%s %s" % (dataset.name, learnerName))
         experiment(dataset, learner)


if __name__ == '__main__':
   runAll()
