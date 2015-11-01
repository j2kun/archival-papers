'''
run by running the following in the project's root directory

   $ python3 -m baselines.german

General statistics
Size of training data: 666
Size of testing data: 333
Size of protected test data: 135
SP of training data: 0.050912
SP of test data: 0.033951


Boosting baseline
Training error: 0.231231
Test error: 0.264264
SP of the hypothesis: 0.033951
kNN consistency of the hypothesis: 0.879279
UBIF of the hypothesis on training: 0.750000

Logistic regression
Training error: 0.300300
Test error: 0.3005 (2e-5)
SP of the hypothesis: 0.009 (5e-5)
UBIF of the hypothesis on training: 0.926 (0.0008)
kNN consistency of the hypothesis: 1.0
'''

from data import german
from boosting import boost
from svm import svmSKL, svmLinearSKL
from lr import lrSKL
from baselines.baseline import *
from errorfunctions import labelError

#def parameterSweep(trainingData, testData):
   #import numpy
   #lambdaRange = numpy.arange(0.1, 2.0, 0.05)
   #roundRange = numpy.arange(25,1000,25)
   #print("lambda, num rounds, test error")

   #for theLambda in lambdaRange:
      #for roundNum in roundRange:
         #_, h = svmDetailedGradientDescent(trainingData, rounds=roundNum, lambdaParameter=theLambda)
         #print("%G, %G, %G" % (theLambda, roundNum, labelError(testData, h)))


if __name__ == "__main__":
   trainingData, testData = german.load()
   protectedIndex = german.protectedIndex
   protectedValue = german.protectedValue

   # to determine which parameters to use for svm
   # parameterSweep(trainingData, testData)

   print("General statistics")
   print(indFairnessStats(trainingData, boost))


   #generalStatistics(trainingData, testData, protectedIndex, protectedValue)
   #print("\nLogistic regression")
   #runBaseline(trainingData, testData, logisticRegression, protectedIndex)
   #print("\nSVM")
   #runBaseline(trainingData, testData, svmGradientDescent, protectedIndex)
   # print("\nBoosting")
   # runBaseline(trainingData, testData, boost, protectedIndex, protectedValue)

   # because the germans dataset has more variance
   #runBaselineAveraged(trainingData, testData, boost, protectedIndex, protectedValue)
   #runBaselineAveraged(trainingData, testData, svmLinearSKL, protectedIndex, protectedValue)

