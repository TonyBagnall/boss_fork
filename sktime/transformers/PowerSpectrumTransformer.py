import numpy as np
import pandas as pd
from sktime.transformers.Transformer import Transformer
from sklearn.base import BaseEstimator


class PowerSpectrumTransformer(Transformer):

    def __init__(self, maxLag=100):
        self._maxLag = maxLag

    def transform(self, X):
        n_samps, self._num_atts = X.shape
        transformedX = np.empty(shape=(n_samps,self._lag))
        for i in range(0,n_samps):
            transformedX[i] = self.ps(X[i])
        return transformedX

    def ps(self,x):
        y = np.zeros(self._num_atts)
        fft=np.fft(x)

        for lag in range(1, self._lag+1):
            y[lag - 1] = np.corrcoef(x[lag:], x[:-lag])[0][1]
        return np.array(y)
