import numpy as np

from mil.models.model import Transformer
from mil.utils.utils import bags2instances

class NormalizeBagsImage(Transformer):
    def __init__(self, maximum=255):
        self.maximum = maximum
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return [np.array(bag) / self.maximum for bag in X]
    
class StandarizerBagsList(Transformer):
    def fit(self, X, y=None):
        ins = bags2instances(X)
        
        self.mean = np.mean(ins, axis=0)
        self.std = np.std(ins, axis=0)
        return self
        
    def transform(self, X):
        return [(np.array(bag) - self.mean) / self.std for bag in X]