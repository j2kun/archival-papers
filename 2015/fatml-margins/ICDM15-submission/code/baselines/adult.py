'''
run by running the following in the project's root directory

   $ python -m baselines.adult

'''

from data import adult
from boosting import boost
from svm import svmSKL
from lr import lrSKL
from baselines.baseline import *
from errorfunctions import labelError


if __name__ == "__main__":
   trainingData, testData = adult.load()
   protectedIndex = adult.protectedIndex
   protectedValue = adult.protectedValue

   print("General statistics")
   generalStatistics(trainingData, testData, protectedIndex, protectedValue)
   print("\n Logistic regression baseline")
   runBaselineAveraged(trainingData, testData, lrSKL, protectedIndex, protectedValue)
   print("\nSVM baseline")
   runBaselineAveraged(trainingData, testData, svmSKL, protectedIndex, protectedValue)
   print("\nBoosting baseline")
   runBaselineAveraged(trainingData, testData, boost, protectedIndex, protectedValue)
