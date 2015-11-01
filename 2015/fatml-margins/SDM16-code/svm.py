import math
import numpy
from numpy.linalg import norm
import random
from utils import sign

DEFAULT_NUM_ROUNDS = 200
DEFAULT_LAMBDA = 1.0
DEFAULT_GAMMA = 0.1


def hyperplaneToHypothesis(w):
   return lambda x: sign(numpy.dot(w,x))


# use scikit-learn to do the svm for us
def svmDetailedSKL(data, gamma=DEFAULT_GAMMA, verbose=False, kernel='rbf'):
   if verbose:
      print("Loading scikit-learn")

   from sklearn import svm
   points, labels = zip(*data)
   clf = svm.SVC(kernel=kernel, gamma=gamma)

   if verbose:
      print("Training classifier")

   skClassifier = clf.fit(points, labels)
   hypothesis = lambda x: skClassifier.predict([x])[0]
   bulkHypothesis = lambda data: skClassifier.predict(data)

   alphas = skClassifier.dual_coef_[0]
   supportVectors = skClassifier.support_vectors_
   error = lambda data: 1 - skClassifier.score(*zip(*data))

   intercept = skClassifier.intercept_
   margin = lambda y: skClassifier.decision_function([y])[0]
   bulkMargin = lambda pts: skClassifier.decision_function(pts)

   if verbose:
      print("Done")

   return (hypothesis, bulkHypothesis, skClassifier, error, alphas, intercept,
            gamma, supportVectors, bulkMargin, margin)


def svmSKL(data, gamma=DEFAULT_GAMMA, verbose=False, kernel='rbf'):
   return svmDetailedSKL(data, gamma, verbose, kernel)[0]

def svmLinearSKL(data, verbose=False):
   return svmDetailedSKL(data, 0, verbose, 'linear')[0]


# so we can test rounds/parameters
def svmGradientDescentGenerator(data, lambdaParameter=DEFAULT_LAMBDA, rounds=DEFAULT_NUM_ROUNDS):
   n, m = len(data[0][0]), len(data)
   examples = numpy.array([x[0] for x in data])
   labels = [x[1] for x in data]
   current = numpy.zeros(n)
   hyperplanes = [None] * rounds

   for t in range(rounds):
      hyperplanes[t] = current / (lambdaParameter * (t+1))
      i = random.randint(0, m-1)

      if labels[i] * numpy.dot(hyperplanes[t], examples[i]) < 1:
         current += labels[i] * examples[i]

      yield (1 / (t+1)) * sum(hyperplanes[:(t+1)])


# perform gradient descent to optimize the SVM objective
# return the final hyperplane and the hypothesis
def svmDetailedGradientDescent(data, lambdaParameter=DEFAULT_LAMBDA, rounds=DEFAULT_NUM_ROUNDS):
   n, m = len(data[0][0]), len(data)
   examples = numpy.array([x[0] for x in data])
   labels = [x[1] for x in data]
   current = numpy.zeros(n)
   hyperplanes = [None] * rounds

   for t in range(rounds):
      hyperplanes[t] = current / (lambdaParameter * (t+1))
      i = random.randint(0, m-1)

      if labels[i] * numpy.dot(hyperplanes[t], examples[i]) < 1:
         current += labels[i] * examples[i]

   finalHyperplane = (1 / rounds) * sum(hyperplanes)
   return finalHyperplane, hyperplaneToHypothesis(finalHyperplane)


# so we can test rounds/parameters
def kSVMGradientDescentGenerator(data, kernel=lambda x,y: numpy.dot(x,y), lambdaParameter=DEFAULT_LAMBDA, rounds=DEFAULT_NUM_ROUNDS):
   n, m = len(data[0][0]), len(data)
   examples = numpy.array([x[0] for x in data])
   labels = [x[1] for x in data]
   alphas = [None]*(rounds+1)
   betas = [None]*(rounds+1)
   betas[0] = numpy.zeros(m)

   #kernelMatrix = [[kernel(examples[i], examples[j]) for j in range(m)] for i in range(m)]

   for t in range(rounds):
      alphas[t] = betas[t]/(lambdaParameter*(t+1))
      i = random.randint(0,m-1)
      betas[t+1] = numpy.copy(betas[t])

      if labels[i]*sum(alphas[t][j]*kernel(examples[j],examples[i]) for j in range(m)) < 1:
         betas[t+1][i] = betas[t][i] + labels[i]
      alpha = 1/(t+1) * sum(alphas[:(t+1)])

      #print(sum(alpha[j] != 0 for j in range(m)))

      supp = [(a,ex) for (a,ex) in zip(alpha, examples) if a!= 0]

      #yield lambda x: sum(alpha[j]*kernel(examples[j], x) for j in range(m) if alpha[j]!=0)
      yield lambda x: sum(a*kernel(ex, x) for (a,ex) in supp)


# perform gradient descent to optimize the SVM objective
# return the final hyperplane and the hypothesis
def kSVMDetailedGradientDescent(data, kernel=lambda x,y: numpy.dot(x,y), lambdaParameter=DEFAULT_LAMBDA, rounds=DEFAULT_NUM_ROUNDS):
   for margin in kSVMGradientDescentGenerator(data, kernel, lambdaParameter, rounds):
      pass
   return margin


# just return the hypothesis
def kSVMGradientDescent(data,kernel):
   margin = kSVMDetailedGradientDescent(data, kernel, DEFAULT_LAMBDA, DEFAULT_NUM_ROUNDS)
   return lambda x: sign(margin(x))

# compute the margin of a point
def margin(point, hyperplane):
   return numpy.dot(hyperplane, point)

# compute the absolute value of the margin of a point
def absMargin(point, hyperplane):
   return abs(margin(point, hyperplane))
