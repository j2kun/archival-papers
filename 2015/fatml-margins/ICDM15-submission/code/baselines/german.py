'''
run by running the following in the project's root directory

   $ python -m baselines.singles

'''

from data import german 
from boosting import boost
from svm import svmLinearSKL
from lr import lrSKL
from baselines.baseline import *
from errorfunctions import labelError


if __name__ == "__main__":
   trainingData, testData = german.load()
   protectedIndex = german.protectedIndex
   protectedValue = german.protectedValue

   print("General statistics")
   generalStatistics(trainingData, testData, protectedIndex, protectedValue)
   print("\n Logistic regression baseline")
   runBaselineAveraged(trainingData, testData, lrSKL, protectedIndex, protectedValue)
   print("\nSVM Linear baseline")
   runBaselineAveraged(trainingData, testData, svmLinearSKL, protectedIndex, protectedValue)
   print("\nBoosting baseline", protectedValue)
   runBaselineAveraged(trainingData, testData, boost, protectedIndex, protectedValue)
