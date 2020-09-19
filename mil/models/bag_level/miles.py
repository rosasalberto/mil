from mil.models import LinearSVC
from mil.models.model import Classifier
from mil.bag_representation.mapping import MILESMapping 

class MILES(Classifier):
    """
    Mapping bags to a instance based feature space, from paper
    MILES: Multiple-Instance Learning via Embedded Instance Selection (Chen et al.)
    http://infolab.stanford.edu/~wangz/project/imsearch/SVM/PAMI06/chen.pdf
    """
    def __init__(self, sigma2=4.5**2, C=0.5):
        """
        Parameters
        ----------
        sigma2 : parameter sigma^2 in line 4 of Algorithm 4.1 in MILES paper.
        C : float, regularizer parameter of linear svm.
        """
        self.sigma2 = sigma2
        self.C = C
    
    def fit(self, X, y, **kwargs):
        """ 
        Parameters
        ----------
        X : array-like containing all the training bags, with shape [bags, instances, features]
        y : array-like containing all the training labels.
        """
        self.check_exceptions(X)
        # mapping bags to the instance based feature space
        self.mapping = MILESMapping(self.sigma2)
        mapped_bags = self.mapping.fit_transform(X)
        
        #train the SVM
        self.model = LinearSVC(penalty="l1", C=self.C, dual=False, 
                               class_weight='balanced', max_iter=10000)
        self.model.fit(mapped_bags, y, **kwargs)
        
        # get parameters from SVM
        self.coef_ = self.model.coef_[0]
        self.intercept_ = self.model.intercept_
        
        return self
        
    def predict(self, X):
        """ 
        Parameters
        ----------
        X : array-like containing all the training bags, with shape [bags, instances, features]
        """
        self.check_exceptions(X)
        # mapping bags to the instance based feature space
        mapped_bags = self.mapping.transform(X)    
        
        #predict classes
        return self.model.predict(mapped_bags)