import tensorflow as tf
import numpy as np

from mil.models.model import Transformer

class Padding(Transformer): 
    def __init__(self, max_len=None):
        self.max_len = max_len

    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        if self.max_len is None:
            X_trans = tf.keras.preprocessing.sequence.pad_sequences(X, padding="post", dtype=np.float32)
        else:
            X_trans = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=self.max_len, padding="post", dtype=np.float32)
        return X_trans