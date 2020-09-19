"""
Iterated discrimination APR algorithm. 
Based on:
Thomas G. Dietterich, Richard H. Lathrop, Tomas Lozano-Perez "Solving the multiple instance problem
with axis-parallel rectangles" 1997 
and the Matlab implementation https://github.com/DMJTax/mil
"""
import numpy as np
from sklearn.neighbors import KernelDensity

from mil.models.model import Classifier
from mil.utils.utils import bags2instances

class APR(Classifier):
    """ 
    Approach is to construct an APR by starting with
    a single positive instance and “growing” the APR by expanding it to cover additional
    positive instances. 
    We call it the “iterated discrimination” algorithm, and it has three basic procedures:
    -Grow. An algorithm for growing an APR with “tight” bounds along a specified set
    of features.
    -Discrim. An algorithm for choosing a set of discriminating features by analyzing
    an APR.
    -Expand. An algorithm for expanding the bounds of an APR to improve its generalization ability. 
    """
    def __init__(self, thres=0.5, epsilon=0.1, step=1, verbose=1):
        """
        Parameters
        ----------
        thres : float, optional
            Feature d is discriminant for the instance if the value is more than 'thres' outside the APR along the feature d. The default is 0.5.
        epsilon : float, optional
            Probability value to limit the density function of the chosen features. The default is 0.1.
            For example if 0.1, then we take the 10% percentile value and the 90% value to increase the bounds of the APR.
        step : float, optional
            Each how many steps, backfitting will be computed (to reduce computational cost in training) . The default is 1.
        verbose : int 0 or 1. printing options

        """
        self.thres = thres
        self.epsilon = epsilon
        self.step = step
        self.verbose = verbose
        
    def fit(self, X, y, **kwargs):  
        """
        Parameters
        ----------
        X : array-like with training bags of shape [n_bags, n_instances, n_features]
        y : array-like with labels of the bags
        verbose : printing options
        
        """        
        y = np.array(y)
        if len(np.unique(y)) != 2:
            raise Exception('Training samples must contain 2 classes')
        
        self.__split_bags(X, y)
        
        iteration = 1
        converged = False
        while not converged:
            self.__grow()
            
            if self.verbose:
                apr_size = np.sum(np.diff([self.mn_, self.mx_], axis=0))
                print("Iteration: {}, APR size: {}, Discriminative features: {}".
                      format(iteration, round(apr_size, 2), len(self.rel_features_)))
            
            converged = self.__discriminate()
            self.rel_features_ = np.argwhere(self.mask_features == 1).reshape(-1)
            iteration +=1
        
        # replicate this 2 arrays for better hyperparameter tuning on self.expand with epsilon.
        self.mn__pred = self.mn_.copy()
        self.mx__pred = self.mx_.copy()
        
        if self.verbose:
            print("---Grow and discriminate has converged---\n")
            
        self.expand__()
        
        return self
        
    def predict(self, X):
        """
        Parameters
        ----------
        bags : testing bags of shape [n_bags, n_instances, n_features]
            
        Returns
        -------
        y_pred : array-like with the class prediction of each bag

        """
        mn = self.mn_[self.rel_features_]
        mx = self.mx_[self.rel_features_]
                    
        y_pred = np.zeros(len(X))
        for i in range(len(X)):
            for j in range(len(X[i])):
                if self.__instance_inside_apr(X[i][j], mn, mx):
                    y_pred[i] = 1
                    
        return np.array(y_pred)
    
    def get_positive_instances(self, X):
        """ Get instances inside the APR

        Parameters
        ----------
        X_test : contains the bags to predict the positive instances

        Returns
        -------
        pos_ins : a list containing the indexs of the positive instances in x_test

        """
        mn = self.mn_[self.rel_features_]
        mx = self.mx_[self.rel_features_]    
            
        pos = []
        for i in range(len(X)):
            for j in range(len(X[i])):
                if self.__instance_inside_apr(X[i][j], mn, mx):
                    pos.append([i,j])
                    
        return np.array(pos)
        
    def __split_bags(self, bags, labels):
        """
        Parameters
        ----------
        bags : array-like with shape [bag, instances, features]
            Bags to split.
        labels : numpy array
            Labels of the bags.

        Returns
        -------
        None.

        """
        
        self.positive_bags = [bags[i] for i in range(len(labels)) if labels[i] == 1]
        negative_bags =  [bags[i] for i in range(len(labels)) if labels[i] == 0]
        self.negative_instances = bags2instances(negative_bags)
        
        self.n_neg_instances = len(self.negative_instances)
        self.n_features = len(self.negative_instances[0])
        
        self.rel_features_ = np.arange(self.n_features)
        self.mask_features = np.ones(self.n_features)
        
    def __get_min_max_array(self):
        """ 
        Gives the minimum feature value between the maxs of self.positive_bags,
        and the maximum feature value from the min of the self.positive_bags and 
        the relative features.
        """        
        minimax = np.array([np.min([np.max(np.array(instance)[:, self.rel_features_], axis=0) for instance in self.positive_bags], axis=0),
                            np.max([np.min(np.array(instance)[:, self.rel_features_], axis=0) for instance in self.positive_bags], axis=0)])
        return minimax
    
    def __compute_eucledian_distance(self):
        """
        Returns
        -------
        d : numpy array with the eucledian distance between each instance in the 
        positive bags and the centroid, around the rel_features.

        """        
        euc = [np.sqrt(np.sum(np.square(np.array(instance)[:, self.rel_features_] - self.center), axis=1)) for instance in self.positive_bags]
        return euc
        
    def __get_initial_positive_instance(self):
        """
        Gives the instance in the positive bags with less eucledian distance from the centroid.
        """
        d = self.__compute_eucledian_distance()
        
        mins = [np.min(ins) for ins in d]
        self.start_bag = np.argmin(mins)
        self.start_inst = np.argmin(d[self.start_bag])
        
    def __get_min_max(self, array):
        """
        Parameters
        ----------
        array : numpy array
            Array to calculate min and max along the features axis.

        Returns
        -------
        numpy array
            The min and max values of the array along features axis.

        """
        return np.array([np.min(array, axis=0), np.max(array, axis=0)]).reshape(2,-1)
        
    def __initialize_pointers(self):
        """
        Create pointers to keep track of instances and bags.
        """
        # choose positive instance
        self.chosen = np.zeros([len(self.positive_bags), len(self.rel_features_)])
        self.chosen[0,:] = np.array(self.positive_bags[self.start_bag][self.start_inst])[self.rel_features_]
        
        # control positive self.positive_bags with instance used
        self.usage = np.zeros([len(self.positive_bags)])
        self.usage[self.start_bag] = 1
        
        # pointer of bag
        self.pointer_bags = np.zeros([len(self.positive_bags)])
        self.pointer_bags[0] = self.start_bag
        
        # pointer of instance used in bag
        self.pointer_inst = np.zeros([len(self.positive_bags)])
        self.pointer_inst[0] = self.start_inst
        
    def __choose_new_positive_instance(self, i):
        """ Choosing new positive instance that minimizes the new APR size.
        
        Parameters
        ----------
        i : int
            Value to keep track of the instances already added.

        """
        newSize = np.inf
        for j in range(len(self.positive_bags)):
            #if we don't have an instance from bag j
            if not self.usage[j]: 
                tmpx = np.array(self.positive_bags[j])[:,self.rel_features_].reshape(-1,len(self.rel_features_))
                
                for k in range(len(self.positive_bags[j])):  #we iterate through the instances
                    tmpAPR = self.__get_min_max([self.curAPR[0], self.curAPR[1], tmpx[k]]) #we get the new min max APR, including the new instance.
                    tmpSize = np.sum(np.diff(tmpAPR, axis=0)) #calculating APR size
                    
                    #from all the non used self.positive_bags, we keep the instance that minimizes the APR size.
                    if (tmpSize < newSize):  
                        newSize = tmpSize;
                        self.pointer_bags[i] = j
                        self.pointer_inst[i] = k
                        self.chosen[i] = tmpx[k]
        
    def __backfitting(self, i):
        """
        Iterate through chosen bags, to see if another instance from chosen bags minimizes APR 

        Parameters
        ----------
        i : int
            Value to keep track of the instances already added..

        Returns
        -------
        None.

        """
        if i < 2: # Only works when we have already choosed instances from a minimum of 3 different bags
            return
        
        if (i%self.step != 0) & (i != (len(self.positive_bags)-1)): #compute backfitting only each self.step 
            return
        
        changed = True
        while changed:
            changed = False
            
            #iterate through the chosen self.positive_bags
            for m in range(0,i+1):
                a = np.arange(i+1)         
                #iterate through all the chosen, to see if another instance minimizes APR size.
                b = a[np.arange(i+1)!= m] 
                
                #calculate APR from chosen instances from current self.positive_bags
                tmpAPR = self.__get_min_max(self.chosen[b]) 
                
                curr_inst = self.pointer_inst[m]
                curr_bag = self.pointer_bags[m]
                tmpx = np.array(self.positive_bags[int(curr_bag)])[:, self.rel_features_]
                
                newSize = np.inf
                #we iterate through the instances for the chosen self.positive_bags, to see if another instance reduce more APR.
                for n in range(len(self.positive_bags[int(curr_bag)])): 
                    tmpAPR1 = self.__get_min_max([tmpAPR[0], tmpAPR[1], tmpx[n]])
                    tmpSize1 = np.sum(np.diff(tmpAPR1, axis=0))
                    
                    if (tmpSize1 < newSize):
                        newSize = tmpSize1
                        self.pointer_inst[m] = n
                        self.chosen[m] = tmpx[n]
                        
                if (self.pointer_inst[m] != curr_inst):
                    changed = True
    
    def __grow(self):
        """
        -Grow. An algorithm for growing an APR with “tight” bounds along a specified set
        of features.

        """
        
        # checking there are features to grow.
        if len(self.rel_features_) == 0:
            raise Exception('All features become non-relevant.')
            
        # get minimax array, gives the minimum feature value between the maxs of self.positive_bags, and the maximum feature value from the min of the self.positive_bags.
        minimax = self.__get_min_max_array()
        
        #get mean between minimax values
        self.center = np.mean(minimax, axis=0)
        
        # get the initial positive instance, the one closer with mininum euclidean distance to the center.
        self.__get_initial_positive_instance()
        self.__initialize_pointers()
        
        for i in range(1, len(self.positive_bags)):
            # greedy improve
            self.curAPR = self.__get_min_max(self.chosen[0:i])    #calculating current APR, from all chosen instances.

            # choosing new instance that minimizes APR
            self.__choose_new_positive_instance(i)
            
            self.usage[int(self.pointer_bags[i])] = 1 #we mark the bag as used
            
            # checking the chosen bags, to see if another instance reduces APR
            self.__backfitting(i)

        apr = self.__get_min_max(self.chosen[:])
        
        self.mn_ = np.zeros(self.n_features)
        self.mx_ = np.zeros(self.n_features)
        self.mn_[self.rel_features_] = apr[0]
        self.mx_[self.rel_features_] = apr[1]
        
    
    def __discriminate(self):
        """
        Discriminate. An algorithm for choosing a set of discriminating features by analyzing an APR.
        """
        
        self.result = np.zeros([self.n_features]) # array to store wether a feature is discriminant = 1, or no = 0.
        under_consider = self.mask_features.copy() # features we have in consideration.
        self.discrimed = np.zeros(self.n_neg_instances) # array to store wether a instance has been discriminated.
        count = 0 # counter of discrimated instances.
        
        # lower bounds and upper bounds for considered features.
        lbs = np.where(under_consider != 0, self.mn_, 0)
        ubs = np.where(under_consider != 0, self.mx_, 0)
        
        # outer distance from APR
        outdistance = np.where(self.negative_instances < lbs, np.abs(self.negative_instances - lbs), 0)
        outdistance = np.where(self.negative_instances > ubs, np.abs(self.negative_instances - ubs), outdistance)
        
        # loop until, all the instances are discriminatives or there is no features under consideration.
        while not((count==self.n_neg_instances) or (np.sum(under_consider)==0)):

            # instances that has been discriminated.  
            index_inst = np.arange(self.n_neg_instances)[self.discrimed != 0] 
            # features that has been discrinated.
            index_feat = np.arange(self.n_features)[under_consider == 0] 
            
            # we apply the mask, to only keep outer distance of the non discrimated features and instances.
            self.masked_out = outdistance.copy()
            self.masked_out[index_inst,:] = 0
            self.masked_out[:,index_feat] = 0
            
            instances_mean_0 = np.where(np.mean(self.masked_out, axis=1) == 0)[0]
            instances_inside_apr = np.intersect1d(instances_mean_0, np.arange(self.n_neg_instances)[self.discrimed == 0])
            # if all the features are inside the APR then we discriminate the instance.
            if len(instances_inside_apr) > 0:
                self.discrimed[instances_inside_apr] = 1
                count = count + len(instances_inside_apr)
                
            # we calculate the number of instances which a feature is discriminative.
            self.thres_feat = self.masked_out > self.thres
            counts = np.count_nonzero(self.thres_feat, axis=0)
            
            # in the instances where no features are above thresholds, we add the max feature.         
            max_value = np.max(self.masked_out, axis=1)
            ins = np.where((max_value < self.thres) & (max_value > 0))[0]
            arg_sum = np.argmax(self.masked_out, axis=1)[ins]

            if len(arg_sum) > 0:
                counts[arg_sum] = counts[arg_sum] + 1
                for i in range(len(arg_sum)):
                    self.thres_feat[ins[i],arg_sum[i]] = True
                
            #we get the most discriminant feature.
            ind = np.argmax(counts)
            # we discriminate the instances of the most discriminated feature.
            discrim_ins = self.thres_feat[:,ind].nonzero()[0]

            if len(discrim_ins) > 0:
                self.discrimed[discrim_ins] = 1
                count = count + len(discrim_ins)
                
            # update discrimant features.
            under_consider[ind] = 0
            self.result[ind] = 1  
         
        converged = (self.result == self.mask_features).all()            
        self.mask_features = self.result
        
        return converged
    
    def expand__(self):
        """
        -Expand. An algorithm for expanding the bounds of an APR to improve its generalization ability.
        The objective is to estimate a kernel density for the chosen instances with rel_features.
        After we have our density function, we sample n samples and we get the epsilon percentiles.
        
        Taking the percentiles from a sampled distribution.
        """        
        self.mn_ = self.mn__pred.copy()
        self.mx_ = self.mx__pred.copy()
        
        for i in range(len(self.rel_features_)):
            
            kd = KernelDensity()
            kd.fit(self.chosen[:,i].reshape(1,-1))
            u = kd.sample(100, random_state=0)
            mn, mx = np.percentile(u, [self.epsilon * 100, (1 - self.epsilon) * 100])
            
            # if the bounds are outside the APR, we update the APR bounds.
            if mn < self.mn_[self.rel_features_[i]]:
                self.mn_[self.rel_features_[i]] = mn
                
            if mx > self.mx_[self.rel_features_[i]]:
                self.mx_[self.rel_features_[i]] = mx
        
    def __instance_inside_apr(self, instance, mn, mx):
        """
        Parameters
        ----------
        instance : instance of shape [n_features]

        Returns
        -------
        boolean, wether the instance is inside the APR.

        """
        ins_rel_features = np.array(instance)[self.rel_features_]
        return ((mn <= ins_rel_features) & (ins_rel_features <= mx)).all()
        