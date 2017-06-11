"""
Author      : Srikanth Mujjiga
Email       : smujjiga@outlook.com
Date        : 5.Jun.2017
Description : Fast scalable implementation of TFKLD from paper http://www.cc.gatech.edu/~jeisenst/papers/ji-emnlp-2013.pdf
This reimplementation is based on https://github.com/jiyfeng/tfkld to speed up weight calculations
"""

import numpy as np
import time
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as ssp
import sys

reload(sys)
sys.setdefaultencoding("latin-1")


class TFKLD(object):
    def __init__(self):
        self._weights = None

    def fit(self, trainX, trainY):
        assert len(trainX) == 2*len(trainY)
        print ("Training Size: {0}".format(len(trainX)))

        print ("Finding vocabulary")
        self.countizer = CountVectorizer(dtype=np.float32, ngram_range=(1,1),
                                         encoding='latin-1', token_pattern=u'(?u)\\b\\w+\\b')
        self.countizer.fit(trainX)
        self.word2id = dict()
        for i, word in enumerate(self.countizer.get_feature_names()):
            self.word2id[word] = i
        self.nTerms = i+1

        print ("Vocabulary Size: {0}".format(self.nTerms))

        print ("Calculating Weights")
        self._calculate_weights(trainX, trainY)

    def _calculate_weights(self, trainX, trainY):
        nSamples = len(trainX)
        count = np.ones((4, self.nTerms))
        self.missing_tokens = dict()

        for n in range(0, nSamples, 2):
            start_time = time.time()
            if n+1 % 10000  == 0:
                print ('Processed {0} rows, Batch processed in {1} secs'.format(n, time.time()-start_time))
                start_time = time.time()

            s1 = dict()
            s2 = dict()

            for word in trainX[n].split():
                if word not in self.word2id:
                    self.missing_tokens.setdefault(word, 0)
                    self.missing_tokens[word] += 1
                else:
                    s1.setdefault(word, 0)
                    s1[word] += 1

            for word in trainX[n+1].split():
                if word not in self.word2id:
                    self.missing_tokens.setdefault(word, 0)
                    self.missing_tokens[word] += 1
                else:
                    s2.setdefault(word, 0)
                    s2[word] += 1

            label = trainY[n // 2]

            for k in s1:
                if not s2.has_key(k):
                    if label == 0:
                        count[0, self.word2id[k]] += 1.0
                    else: #label == 1:
                        count[2, self.word2id[k]] += 1.0

                else: #s2.has_key(k):
                    del s2[k]
                    if label == 0:
                        count[1, self.word2id[k]] += 1.0
                    else: #label == 1:
                        count[3, self.word2id[k]] += 1.0
            del s1

            for k in s2:
                if label == 0:
                    count[0, self.word2id[k]] += 1.0
                else:  # label == 1:
                    count[2, self.word2id[k]] += 1.0
            del s2

            self.weights = self._computeKLD(count)
            assert len(self.weights) == self.nTerms

    def _computeKLD(self, count):
        # Smoothing
        count = count + 0.05
        # Normalize
        pattern = [[1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1]]
        pattern = np.array(pattern)
        prob = count / (pattern.dot(count))
        #
        ratio = np.log((prob[0:2,:] / prob[2:4,:]) + 1e-7)
        weight = (ratio * prob[0:2,:]).sum(axis=0)
        return weight

    def transform(self, trainX):
        start_time = time.time()

        if self._weights is None:
            self._weights = ssp.diags(self.weights, 0)

        print ("Transforming data of size: {0}".format(len(trainX)))
        X = ssp.lil_matrix(self.countizer.transform(trainX))
        print ("Reweighing data")
        X = X*self._weights
        print ("Finished in: ".format(time.time() - start_time))
        return X

