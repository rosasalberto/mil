from sklearn.base import BaseEstimator, TransformerMixin

from mil.errors.custom_exceptions import DimensionError
from mil.utils.utils import recursive_len

class BagRepresentation(BaseEstimator, TransformerMixin):
    """ 
    Base class for bag representation.
    transform method return a unique vector represesenting the entire bag.
    """
    def check_exceptions(self, bags):
        if recursive_len(bags) < 3:
           raise DimensionError 
               
    def fit(self, X, y=None):
        raise NotImplementedError
        
    def transform(self, X):
        raise NotImplementedError