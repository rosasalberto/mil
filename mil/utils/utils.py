import numpy as np
import sklearn

def recursive_len(x, dim=0):
    try:
        dim = recursive_len(x[0], dim+1)
    except:
        pass
    return dim
    
def bags2instances(bags):
    return [instance for bag in bags for instance in bag]
    
def get_samples_weight(y_train):
    return sklearn.utils.class_weight.compute_sample_weight('balanced', y_train).reshape(-1,1)