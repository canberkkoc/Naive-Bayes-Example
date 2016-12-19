import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# Csvleri panda dataframelere dönüştürme işlemi
Xtest = pd.read_csv("Xtest.txt", header=None)
Xtrain = pd.read_csv("Xtrain.txt", header=None)
Ytest = pd.read_csv("label_test.txt", header=None)
Ytrain = pd.read_csv("label_train.txt", header=None)

class Bayes(object):
    def fit(self, X, y):
        self.gaussians = dict()
        labels = set(y.as_matrix().flatten())
        for c in labels:
            current_x = Xtrain[Ytrain[0] == c]
            self.gaussians[c] = {
                'mu': current_x.mean(),
                'sigma': np.cov(current_x.T),
            }
            

    def predict_one(self, x):
        lls = self.distributions(x)
        return np.argmax(lls)

    def predict(self, X):
        Ypred = X.apply(lambda x: self.predict_one(x), axis=1)
        return Ypred

    def distributions(self, x):
        lls = np.zeros(len(self.gaussians))
        for c,g in self.gaussians.iteritems():
            x_minus_mu = x - g['mu']
            k1 = np.log(2*np.pi)*x.shape + np.log(np.linalg.det(g['sigma']))
            k2 = np.dot( np.dot(x_minus_mu, np.linalg.inv(g['sigma'])), x_minus_mu)
            ll = -0.5*(k1 + k2)
            lls[c] = ll
        return lls


if __name__ == '__main__':
    bayes = Bayes()
    bayes.fit(Xtrain, Ytrain)
    Ypred = bayes.predict(Xtest)
    C = np.zeros((10,10), dtype=np.int)
    # print len(Ypred), len(Ytest)
    for p,t in zip(Ypred.as_matrix().flatten(), Ytest.as_matrix().flatten()):
        C[t,p] += 1
    print "Matris Cizimi:"
    print C
    print "İsabet oranı:", np.trace(C) / 500.0

    if 'reconstruct' in sys.argv:
        # Resimler üzerinde orta noktaları çizdirme
        Q = pd.read_csv("mnist_csv/Q.txt", header=None).as_matrix()
        for c,g in bayes.gaussians.iteritems():
            y = np.dot(Q, g['mu'].as_matrix())
            y = np.reshape(y, (28,28))
            plt.imshow(y)
            plt.title(c)
            plt.show()
