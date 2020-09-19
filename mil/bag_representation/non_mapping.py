import numpy as np
from scipy.stats.mstats import gmean

from mil.bag_representation.bag_representation import BagRepresentation
        
class ArithmeticMeanBagRepresentation(BagRepresentation):
    """ Represent bag with the arithmetic mean value of each feature inside the bag """
    def fit(self, X=None, y=None):
        return self
    
    def transform(self, X):
        self.check_exceptions(X)
        return np.array([np.mean(bag, axis=0) for bag in X])
 
class MedianBagRepresentation(BagRepresentation):
    """ Represent bag with the median value of each feature inside the bag """
    def fit(self, X=None, y=None):
        return self
    
    def transform(self, X):
        self.check_exceptions(X)
        return np.array([np.median(bag, axis=0) for bag in X]) 
            
class GeometricMeanBagRepresentation(BagRepresentation):
    """ Represent bag with the geometric mean value of each feature inside the bag """        
    def fit(self, X=None, y=None):
        return self
    
    def transform(self, X):
        self.check_exceptions(X)
        return np.array([gmean(bag, axis=0) for bag in X])
                
class MinBagRepresentation(BagRepresentation):
    """ Represent bag with the min value of each feature inside the bag """
    def fit(self, X=None, y=None):
        return self
    
    def transform(self, X):
        self.check_exceptions(X)
        return np.array([np.min(bag, axis=0) for bag in X])
                
class MaxBagRepresentation(BagRepresentation):
    """ Represent bag with the max value of each feature inside the bag """
        
    def fit(self, X=None, y=None):
        return self
    
    def transform(self, X):
        self.check_exceptions(X)
        return np.array([np.max(bag, axis=0) for bag in X])
            
class MeanMinMaxBagRepresentation(BagRepresentation):
    """ Represent bag with the mean of the min-max value of each feature inside the bag """
    def __init__(self):
        self.min_bag = MinBagRepresentation()
        self.max_bag = MaxBagRepresentation()
        
    def fit(self, X=None, y=None):
        return self
    
    def transform(self, X):
        self.check_exceptions(X)
        bag_min = self.min_bag.fit_transform(X)
        bag_max = self.max_bag.fit_transform(X)
        return (bag_min + bag_max) / 2