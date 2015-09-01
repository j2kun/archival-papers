#!/usr/bin/env python3

import boosting
import numpy
from utils import arrayErrorBars, errorBars
from weaklearners.decisionstump import buildDecisionStump
import errorfunctions as ef
from data import adult, singles, german

'''
Run boosting on a decision stump finder that uses an error function which
is a linear combination of statistical imparity and label error

	w * statisticalParity + (1-w) * labelError

'''
def makeErrorFunction(protectedIndex, protectedValue, spWeight):
   sp = lambda data, h: ef.statisticalParity(data, protectedIndex, protectedValue, h=h)
   le = ef.minLabelErrorOfHypothesisAndNegation
   return ef.makeLinearCombination(sp, le, spWeight)


@arrayErrorBars(10)
def statistics(train, test, protectedIndex, protectedValue, numRounds=20):
   weight = 0.5
   flipProportion = 0.2

   error = makeErrorFunction(protectedIndex, protectedValue, weight)
   weakLearner = lambda draw: buildDecisionStump(draw, errorFunction=error)

   h = boosting.boost(train, weakLearner = weakLearner)

   bias = ef.signedStatisticalParity(test, protectedIndex, protectedValue, h)
   error = ef.labelError(test, h)
   ubif = ef.individualFairness(train, boosting.boost, flipProportion)

   return error, bias, ubif


def experiment(dataModule):
   train, test = dataModule.load()
   PI = dataModule.protectedIndex
   PV = dataModule.protectedValue
   output = statistics(train, test, PI, PV)
   print(dataModule.name)
   print("\tavg, min, max, variance")
   print("error: %r" % (output[0],))
   print("bias: %r" % (output[1],))
   print("ubif: %r" % (output[2],))


def runAll():
   datasets = [
      adult,
      singles,
      german,
   ]

   print("Fair weak learner")
   for dataset in datasets:
      experiment(dataset)


if __name__ == '__main__':
   runAll()
