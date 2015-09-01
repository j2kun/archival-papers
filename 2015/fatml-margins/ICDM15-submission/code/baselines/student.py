'''
run by running the following in the project's root directory

   $ python -m baselines.student

'''

from data import student
from boosting import boost
from svm import svmGradientDescent
from baselines.baseline import runBaseline, generalStatistics


if __name__ == "__main__":
   trainingData, testData = student.load('mat')
   protectedIndex = student.protectedIndex

   generalStatistics(trainingData, testData, protectedIndex)
   runBaseline(trainingData, testData, svmGradientDescent, protectedIndex)
   runBaseline(trainingData, testData, boost, protectedIndex)

   trainingData, testData = student.load('por')
   protectedIndex = student.protectedIndex

   generalStatistics(trainingData, testData, protectedIndex)
   runBaseline(trainingData, testData, svmGradientDescent, protectedIndex)
   runBaseline(trainingData, testData, boost, protectedIndex)
