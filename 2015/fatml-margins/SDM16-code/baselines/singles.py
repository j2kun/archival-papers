'''
run by running the following in the project's root directory

   $ python -m baselines.singles

'''

from data import singles
from boosting import boost
from svm import svmSKL
from lr import lrSKL
from baselines.baseline import *
from errorfunctions import labelError


if __name__ == "__main__":
   trainingData, testData = singles.load()
   protectedIndex = singles.protectedIndex

   #parameterSweep(trainingData, testData)
   print("General statistics")
   generalStatistics(trainingData, testData, protectedIndex)
   print("\n Logistic regression baseline")
   runBaselineAveraged(trainingData, testData, lrSKL, protectedIndex)
   print("\nSVM baseline")
   runBaselineAveraged(trainingData, testData, svmSKL, protectedIndex)
   print("\nBoosting baseline")
   runBaselineAveraged(trainingData, testData, boost, protectedIndex)
