from utils import sign, zeroOneSign, normalize01, lpDistance
import random
import heapq


def minLabelErrorOfHypothesisAndNegation(data, h):
   posData, negData = ([(x, y) for (x, y) in data if h(x) == 1],
                       [(x, y) for (x, y) in data if h(x) == -1])

   posError = sum(y == -1 for (x, y) in posData) + sum(y == 1 for (x, y) in negData)
   negError = sum(y == 1 for (x, y) in posData) + sum(y == -1 for (x, y) in negData)
   return min(posError, negError) / len(data)


def makeLinearCombination(error1, error2, error1Weight):
   def customError(data, h):
      e1 = error1(data, h)
      e2 = error2(data, h)
      return error1Weight * e1 + (1 - error1Weight) * e2

   return customError


def precomputedLabelError(data, labels):
   return sum(1 for (x,l) in zip(data, labels) if x[1] != l) / len(data)


def labelError(data, h):
   return len([1 for (x,y) in data if h(x) != y]) / len(data)


# data is a list of unlabeled examples
def precomputedLabelStatisticalParity(data, labels, protectedIndex, protectedValue, weights=None):
   if weights == None:
      weights = [1]*len(data)

   protectedClass = [(x,wt,l) for (x,wt,l) in zip(data, weights, labels)
                        if x[protectedIndex] == protectedValue]
   elseClass  = [(x,wt,l) for (x,wt,l) in zip(data, weights, labels)
                        if x[protectedIndex] != protectedValue]

   if len(protectedClass) == 0:
      print("Nobody in the protected class")
      return sum(w for (x,w,l) in elseClass  if l == 1) / sum(w for (x,w,l) in elseClass)
   elif len(elseClass) == 0:
      print("Nobody in the else class")
      return -sum(w for (x,w,l) in protectedClass if l == 1) / sum(w for (x,w,l) in protectedClass)
   else:
      protectedProb = sum(w for (x,w,l) in protectedClass if l == 1) / sum(w for (x,w,l) in protectedClass)
      elseProb =  sum(w for (x,w,l) in elseClass  if l == 1) / sum(w for (x,w,l) in elseClass)

   return elseProb - protectedProb


# data is a list of labeled examples (this is weird; should be consistent)
def signedStatisticalParity(data, protectedIndex, protectedValue, h=None, weights=None):
   if len(data[0]) == 2: # should do better type checking here...
      pts, labels = zip(*data)
   else:
      pts = data
      labels = None

   if h is not None:
      labels = [h(x) for x in pts]
      #print(len([x for x in labels if x == 1]))
      #print(len([x for x in labels if x == -1]))

   if labels is None:
      raise Exception("Must provide either labels or a hypothesis to signedStatisticalParity")
   return precomputedLabelStatisticalParity(pts, labels, protectedIndex, protectedValue, weights)


def statisticalParity(data, protectedIndex, protectedValue, h=None, weights=None):
   return abs(signedStatisticalParity(data, protectedIndex, protectedValue, h, weights))


# add a new unbiased feature, introduce uniform random discrimination on that feature
# run the learner on the biased data, and then see how many of the flipped labels
# it can recover.
# learner: data -> classifier function
def individualFairness(data, learner, flipProportion=0.2, passProtected=False):
   protectedIndex = 0
   protectedValue = 0
   unbiasedData = [((random.choice([0,1]),) + x[0], x[1]) for x in data]

   indicesOfProtected = [i for i,x in enumerate(unbiasedData)
                           if x[0][protectedIndex] == 0 and x[1] == 1]
   m = len(indicesOfProtected)

   indicesOfFlippedData = set(random.sample(indicesOfProtected, int(flipProportion * m)))
   biasedData = [(x[0], (-1 if i in indicesOfFlippedData else x[1])) for i,x in enumerate(unbiasedData)]

   if passProtected:
      h = learner(biasedData, protectedIndex, protectedValue)
   else:
      h = learner(biasedData)

   flippedPts = [x for i,x in enumerate(biasedData) if i in indicesOfFlippedData]
   error = sum(h(x) != y for (x,y) in flippedPts) / len(flippedPts)
   return error


def kNNfairness(data, d, h=None, k=10):
   if h == None:
      data=list(data)
      label_dict = dict(data)
      h = lambda x:label_dict[x]
   unlabeledData = [d[0] for d in data]
   return 1 - sum(abs(sum(h(y) for y in heapq.nsmallest(k, unlabeledData, key=lambda y: d(x,y)))/k - h(x)) for x in unlabeledData)/len(unlabeledData)

def kNNfairnessNormalizedLP(data, h=None, k=10, p=2):
   return kNNfairness(zip(normalize01([d[0] for d in data]), [d[1] for d in data]), lambda x,y: lpDistance(x, y, p), h, k)
