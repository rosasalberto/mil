from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin

from mil.errors.custom_exceptions import DimensionError
from mil.utils.utils import recursive_len

class Model(BaseEstimator):
    def check_exceptions(self, bags):
        if recursive_len(bags) < 3:
           raise DimensionError 
    
    def fit(self, X, y=None):
        raise NotImplementedError
        
class Classifier(Model, ClassifierMixin):
    def fit(self, X, y=None):
        raise NotImplementedError
        
    def predict(self, X):
        raise NotImplementedError
        
class Regressor(Model, RegressorMixin):
    def fit(self, X, y=None):
        raise NotImplementedError
        
    def predict(self, X):
        raise NotImplementedError
        
class Transformer(Model, TransformerMixin):        
    def fit(self, X, y=None):
        raise NotImplementedError
        
    def transform(self, X):
        raise NotImplementedError
        
class EmptyTransformer(Model, TransformerMixin):        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return X