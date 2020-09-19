import numpy as np

from mil.bag_representation.bag_representation import BagRepresentation
from mil.utils.utils import bags2instances

class MILESBase(BagRepresentation):
    """
    Base class for MILES based methdos.
    """
    def __init__(self, sigma2):
        """
        Parameters
        ----------
        sigma2 : parameter sigma^2 in line 4 of Algorithm 4.1 in MILES paper.
        """
        self.sigma2 = sigma2
    
    def fit(self, X, y=None):
        """ 
        Parameters
        ----------
        X : array-like containing all the training bags, with shape [bags, instances, features]
        y : optional. array-like containing all the training labels.
        """
        raise NotImplementedError
        
    def calculate_iip(self, X, y):
        """ Method for calculating the instance pool to do the bag-instance similarity.

        Parameters
        ----------
        X : array-like containing all the training bags, with shape [bags, instances, features]
        y : optional. array-like containing all the training labels.
        """
        raise NotImplementedError
    
    def transform(self, X):
        """ Get the bag representation calculating the bag-instance similarity.
        
        Parameters
        ----------
        X : array-like containing all the bags to do the embedding, 
            with shape [bags, instances, features]
            
        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self.check_exceptions(X)
        dist = [self.get_bag_instances_distance(bag) for bag in X]
        return self.similarity_measure(dist)
        
    def similarity_measure(self, dist):
        """ Calculates the similarity measure between bags and instances
            used in the MILES paper.

        Parameters
        ----------
        dist : distance measure between bags and instances.

        Returns
        -------
        a matrix containing the distance between each bag and instance in the instance pool.

        """
        return np.exp(-np.array(dist)**2./self.sigma2)
        
        
    def get_instance_instances_distance(self, instance):
        """ Calculates the distance between instance, and the instance pool.

        Parameters
        ----------
        instance : array-like of shape [features]

        Returns
        -------
        an array containing the distance between a instance and the instance pool. 
        The distance is the norm between the instance features and the instance pool.

        """
        axes = tuple([e for e in np.arange(1, len(np.array(instance).shape) + 1)])
        return np.linalg.norm(instance - self.iip_, axis=axes)
    
    def get_bag_instances_distance(self, bag):
        """ Calculates the minimum instances between the instances 
            of bag and the training instances

        Parameters
        ----------
        bag : array-like of shape [instances, features]

        Returns
        -------
        the minimum distance between the instance of a bag and the training pool.

        """
        d = [self.get_instance_instances_distance(ins) for ins in bag]
        return np.min(d, axis=0)

class MILESMapping(MILESBase):
    """
    Mapping bags to a instance based feature space, from paper
    MILES: Multiple-Instance Learning via Embedded Instance Selection (Chen et al.)
    http://infolab.stanford.edu/~wangz/project/imsearch/SVM/PAMI06/chen.pdf
    """
    def __init__(self, sigma2=4.5**2):
        """
        Parameters
        ----------
        sigma2 : parameter sigma^2 in line 4 of Algorithm 4.1 in MILES paper.
        """
        super(MILESMapping, self).__init__(sigma2)
    
    def fit(self, X, y=None):
        """ Training method to be called before predicting.

        Parameters
        ----------
        X : array-like containing all the training bags, with shape [bags, instances, features]
        y : optional. array-like containing the labels of the bags X.

        Returns
        -------
        self

        """

        self.check_exceptions(X)
        self.calculate_iip(X, y)
        
        return self
    
    def calculate_iip(self, X, y):
        """ Assign the instance pool using all the training instances.

        Parameters
        ----------
        X : array-like containing all the training bags, with shape [bags, instances, features]
        y : optional. array-like containing the labels of the bags X.
        """
        self.iip_ = np.array(bags2instances(X))
        
class DiscriminativeMapping(MILESBase):
    """
    Mapping bags to a instance based feature space, from paper
    Multi-instance Learning with Discriminative Bag Mapping (Wu et al.)
    http://www.cse.fau.edu/~xqzhu/papers/TKDE.Wu.2017.Multiinstance.pdf
    """
    def __init__(self, m=2, sigma2=8e5):
        """
        Parameters
        ----------
        m : number of instances to construct the instance pool.
        sigma2 : parameter sigma^2 in line 4 of Algorithm 4.1 in MILES paper.
        """
        super(DiscriminativeMapping, self).__init__(sigma2)
        self.m = m
    
    def fit(self, X, y):
        """
        X : array-like containing all the training bags, 
                             with shape [bags, instances, features]
        """
        self.check_exceptions(X)
        self.calculate_iip(X, np.array(y))
        
        return self
    
    def calculate_iip(self, X, y):
        """ Assign the instance pool using the m most discriminative training instances.

        Parameters
        ----------
        X : array-like containing all the training bags, with shape [bags, instances, features]
        y : optional. array-like containing the labels of the bags X.
        """
        label = np.where(y==0, -1, y)
        
        # get instances-bag similarity
        m = MILESMapping()
        ins_bag = m.fit_transform(X).T
        
        # calculate Q
        Y = np.array([i * j for i in label for j in label]).reshape(len(label), len(label))
        B, A = np.unique(Y, return_counts=True)[1]
        Q = np.where(Y==-1, -1/A, 1/B)
        
        # calculate J
        J = np.sum(ins_bag @ Q, axis=1)
        
        # get m best index of J, get maximum m in J
        self.items_ = np.argpartition(J, -self.m)[-self.m:] 
            
        # select dip
        self.iip_ = np.array(bags2instances(X))[self.items_]

        