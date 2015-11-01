from errorfunctions import labelError, statisticalParity, individualFairness, kNNfairnessNormalizedLP
from utils import arrayErrorBars, errorBars

def generalStatistics(trainingData, testData, protectedIndex, protectedValue):
   print('Size of training data:', len(trainingData))
   print('Size of testing data:', len(testData))

   print("SP of training data: %f" % statisticalParity(trainingData, protectedIndex, protectedValue))
   print("SP of test data: %f" % statisticalParity(testData, protectedIndex, protectedValue))


FLIP_PROPORTION = 0.2

def runBaseline(trainingData, testData, learner, protectedIndex, protectedValue):
   h = learner(trainingData)

   print("Training error: %f" % labelError(trainingData, h))
   print("Test error: %f" % labelError(testData, h))
   print("SP of the hypothesis: %f" % statisticalParity(testData, protectedIndex, protectedValue, h))

   UBIF = individualFairness(trainingData, learner, FLIP_PROPORTION)
   print("UBIF of the hypothesis on training: %f" % UBIF)
   #print("kNN consistency of the hypothesis: %f" % kNNfairnessNormalizedLP(testData, h, 10))

   #RUB = resistanceToUniformBias(trainingData, testData, learner, FLIP_PROPORTION)
   #print("RUB of the hypothesis: %f" % RUB)


@arrayErrorBars(20)
def statistics(trainingData, testData, learner, protectedIndex, protectedValue):
   h = learner(trainingData)

   trainingError = labelError(trainingData, h)
   testError = labelError(testData, h)
   sp = statisticalParity(testData, protectedIndex, protectedValue, h)
   UBIF = individualFairness(trainingData, learner, FLIP_PROPORTION)

   return trainingError, testError, sp, UBIF

@errorBars(10)
def indFairnessStats(trainingData, learner):
   UBIF = individualFairness(trainingData, learner, FLIP_PROPORTION)
   return UBIF


def runBaselineAveraged(train, test, learner, protectedIndex, protectedValue):
   output = statistics(train, test, learner, protectedIndex, protectedValue)
   print("\tavg, min, max, variance")
   print("train error: %r" % (output[0],))
   print("test error: %r" % (output[1],))
   print("bias: %r" % (output[2],))
   print("ubif: %r" % (output[3],))
   return output
