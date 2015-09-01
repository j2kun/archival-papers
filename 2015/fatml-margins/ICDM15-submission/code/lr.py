import random
import numpy
from decimal import Decimal
from utils import sigmoid, sign, zeroOneSign

DEFAULT_NUM_ROUNDS = 1000
DEFAULT_ETA = 0.01
numpy.seterr(over='ignore')

#Runs SGD for rounds rounds with parameter eta to minimize the risk function E_{z~D}(l(w,z))
#dl takes two arguments, w[t] and z, and returns a subgradient of l(w,z) at w[t]
#d is the dimension of the space
#drawExample draws an example from the distribution D
#This follows chapter 14.5.1 of Shalev-ShwartzBD14
def sgdGenerator(dl, d, drawExample, eta=DEFAULT_ETA, rounds=DEFAULT_NUM_ROUNDS):
   w = [None]*(rounds+1)
   w[0] = numpy.zeros(d)
   for t in range(rounds):
      z = drawExample()
      v = numpy.array(dl(w[t], z))
      assert len(v)==d
      w[t+1] = w[t] - eta*v
      yield (1/(t+1)) * sum(w[:(t+1)])

def lrGenerator(data, eta=DEFAULT_ETA, rounds=DEFAULT_NUM_ROUNDS):
   d=len(data[0][0])
   drawExample = lambda : random.choice(data)

   def dl(w, z):
      x,y = z
      tmp = numpy.exp(-y*numpy.dot(w, x))
      if numpy.isinf(tmp):
         return (-y * numpy.array(x))
      else:
         return tmp/(1+tmp) * (-y*numpy.array(x))

   yield from sgdGenerator(dl, d, drawExample, eta, rounds)


def detailedLR(data, eta=DEFAULT_ETA, rounds=DEFAULT_NUM_ROUNDS):
   for w in lrGenerator(data, eta, rounds):
      pass
   return w, lambda x: sigmoid(numpy.dot(w,x)), lambda x: sign(numpy.dot(w,x))


def logisticRegression(data, eta=DEFAULT_ETA, rounds=DEFAULT_NUM_ROUNDS):
   return detailedLR(data, eta, rounds)[-1]
   
#use scikit-learn
def lrDetailedSKL(data):
   from sklearn import linear_model
   points, labels = zip(*data)
   clf = linear_model.LogisticRegression()
   lrClassifier = clf.fit(points, labels)
   return lambda x: lrClassifier.predict_proba(x)[0][1], lambda x: 1 if lrClassifier.predict_proba(x)[0][1]>=0.5 else -1
   
   
def lrSKL(data):
   return lrDetailedSKL(data)[1]
