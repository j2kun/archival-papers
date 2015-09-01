#!/usr/bin/env python3

from data import adult, german, singles
from utils import errorBars, arrayErrorBars
from boosting import boost
from svm import svmSKL, svmLinearSKL
from lr import lrSKL
from weaklearners.decisionstump import buildDecisionStump
from errorfunctions import signedStatisticalParity, labelError, individualFairness
from massaging import randomOneSideMassageData


@arrayErrorBars(10)
def statistics(massager, trainingData, testData, protectedIndex, protectedValue,
               learner, flipProportion=0.2):
   massagedData = massager(trainingData, protectedIndex, protectedValue)
   h = learner(massagedData)

   error = labelError(testData, h)
   bias = signedStatisticalParity(testData, protectedIndex, protectedValue, h)
   ubif = individualFairness(trainingData, learner, flipProportion)

   return error, bias, ubif


def experiment(dataModule, learner):
   train, test = dataModule.load()
   PI = dataModule.protectedIndex
   PV = dataModule.protectedValue
   output = statistics(randomOneSideMassageData, train, test, PI, PV, learner)
   print("\tavg, min, max, variance")
   print("error: %r" % (output[0],))
   print("bias: %r" % (output[1],))
   print("ubif: %r" % (output[2],))


def runAll():
   print("Random Massaging")
   datasets = [
               german,
               adult,
               singles,
              ]
   learners = [
               ('SVMlinear', svmLinearSKL),
               ('AdaBoost', boost), 
               ('LR', lrSKL),
              ]

   for dataset in datasets:
      for learnerName, learner in learners:
         print("%s %s" % (dataset.name, learnerName))
         experiment(dataset, learner)


if __name__ == '__main__':
   runAll()

