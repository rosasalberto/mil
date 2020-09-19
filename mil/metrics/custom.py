import numpy as np
import tensorflow as tf
from tensorflow.keras.backend import epsilon

from mil import metrics

class Metric(tf.keras.metrics.Metric):
    """ Custom base class for implementing a metric,
        each Metric subclass has to implement this methods. """
    def __init__(self, name, **kwargs):
        super(Metric, self).__init__(name=name, **kwargs)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        """ Update the state of the metric 

        Parameters
        ----------
        y_true : array-like containing the ground_truth
        y_pred : array-like containing the network predictions
        sample_weight : optional. weight more some predictions.

        """
        raise NotImplementedError
        
    def result(self):
        """ Get the result of a metric """
        raise NotImplementedError
    
    def reset_states(self):
        """ Reset the state of the metric """
        raise NotImplementedError
        
class Specificity(Metric):
    def __init__(self, name='specificity', **kwargs):
        super(Specificity, self).__init__(name=name, **kwargs)
        self.specificity = self.add_weight(name='specificity', initializer='zeros')
        self.specificity.assign(np.nan)
        self.tn = metrics.TrueNegatives()
        self.fp = metrics.FalsePositives()
        
    def update_state(self, y_true, y_pred, sample_weight=None):             
        self.tn.update_state(y_true, y_pred, sample_weight)
        self.fp.update_state(y_true, y_pred, sample_weight)
        
        tn = self.tn.result()
        fp = self.fp.result()
        
        value = tf.where(tn + fp == 0, np.nan, tn / (tn + fp + epsilon()))
        self.specificity.assign(value)
    
    def result(self):
        return self.specificity
    
    def reset_states(self):
        self.specificity.assign(np.nan)
        self.tn.reset_states()
        self.fp.reset_states()
        
class Sensibility(Metric):
    def __init__(self, name='sensibility', **kwargs):
        super(Sensibility, self).__init__(name=name, **kwargs)
        self.sensibility = self.add_weight(name='sensibility', initializer='zeros')
        self.sensibility.assign(np.nan)
        self.tp = metrics.TruePositives()
        self.fn = metrics.FalseNegatives()
        
    def update_state(self, y_true, y_pred, sample_weight=None):             
        self.tp.update_state(y_true, y_pred, sample_weight)
        self.fn.update_state(y_true, y_pred, sample_weight)
        
        tp = self.tp.result()
        fn = self.fn.result()
        
        value = tf.where(tp + fn == 0, np.nan, tp / (tp + fn + epsilon()))
        self.sensibility.assign(value)
    
    def result(self):
        return self.sensibility
    
    def reset_states(self):
        self.sensibility.assign(np.nan)
        self.tp.reset_states()
        self.fn.reset_states()
               
class PPV(Metric):
    def __init__(self, name='ppv', **kwargs):
        super(PPV, self).__init__(name=name, **kwargs)
        self.ppv = self.add_weight(name='ppv', initializer='zeros')
        self.ppv.assign(np.nan)
        self.tp = metrics.TruePositives()
        self.fp = metrics.FalsePositives()
        
    def update_state(self, y_true, y_pred, sample_weight=None):             
        self.tp.update_state(y_true, y_pred, sample_weight)
        self.fp.update_state(y_true, y_pred, sample_weight)
        
        tp = self.tp.result()
        fp = self.fp.result()
        
        value = tf.where(tp + fp == 0, np.nan, tp / (tp + fp + epsilon()))
        self.ppv.assign(value)
    
    def result(self):
        return self.ppv
    
    def reset_states(self):
        self.ppv.assign(np.nan)
        self.tp.reset_states()
        self.fp.reset_states()
        
class NPV(Metric):
    def __init__(self, name='npv', **kwargs):
        super(NPV, self).__init__(name=name, **kwargs)
        self.npv = self.add_weight(name='npv', initializer='zeros')
        self.npv.assign(np.nan)
        self.tn = metrics.TrueNegatives()
        self.fn = metrics.FalseNegatives()
        
    def update_state(self, y_true, y_pred, sample_weight=None):             
        self.tn.update_state(y_true, y_pred, sample_weight)
        self.fn.update_state(y_true, y_pred, sample_weight)
        
        tn = self.tn.result()
        fn = self.fn.result()
        
        value = tf.where(tn + fn == 0, np.nan, tn / (tn + fn + epsilon()))
        self.npv.assign(value)
    
    def result(self):
        return self.npv
    
    def reset_states(self):
        self.npv.assign(np.nan)
        self.tn.reset_states()
        self.fn.reset_states()