import numpy as np
from sklearn.pipeline import Pipeline

from mil.metrics.manager import MetricsManager
from mil.errors.custom_exceptions import DimensionError, FitNonCalledError, PrepareNonCalledError
from mil.utils.utils import recursive_len, get_samples_weight
from mil.utils.progress_bar import ProgressBar
from mil.validators import PredefinedSplit

class Trainer:
    """ Base class for training and evaluating a model given input data.
        Supports a pipeline execution """
    def __init__(self):
        self.prepared = False
        self.fitted = False
    
    def prepare(self, model, preprocess_pipeline=[], metrics=[]):
        """" Method to prepare the training, creates the pipeline and
            declares the metrics to be used on the evaluation 
        
        Parameters
        ----------
        model : a Model object or KerasClassifier object which handles the training and predictions.
        preprocess_pipeline : list of mil objects to be called before accesing the model
                              when training the model, the fit_transform method of each 
                              object in the list is called, when evaluating the model, 
                              the transform method is called.
        metrics : a list containing the Metrics callable modules, or the associated string of 
                  the metrics to be evaluated.
        """
        self.model = model
        pre_pip = preprocess_pipeline.copy()
        pre_pip.append(('model', self.model))
        self.pipeline = Pipeline(pre_pip)
        
        self.metrics_train = MetricsManager(metrics)
        self.metrics_val = MetricsManager(metrics)
        self.metrics_test = MetricsManager(metrics)
        
        # progress bar
        metrics = list(self.metrics_train.metrics.keys())
        self.prog = ProgressBar(metrics)
        
        self.prepared = True
        
    def __check_shape(self, bags):
        if recursive_len(bags) < 3:
           raise DimensionError 
        
    def __check_exception_fit(self):
        if not self.prepared:
            raise PrepareNonCalledError
            
    def __check_exception_predict(self):
        if not self.fitted:
            raise FitNonCalledError
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, groups=None, 
            validation_strategy=None, sample_weights=None, verbose=1, **kwargs):
        """ Method to train the model

        Parameters
        ----------
        X_train : array-like containing all the training bags
        y_train : array-like containing all the training labels
        X_val : optional. array-like containing all the validation bags
        y_val : optional. array-like containing all the validation labels
        groups : optional. array-like of the splitting group of PredefinedSplid 
                class in sklearn.
        validation_strategy : optional. mil validator object defining the validation strategy
        sample_weights : optional. 'balanced' or array-like containing all the weights of each sample
        verbose : optional. printing options in the evaluation process.

        Returns
        -------
        history : a dictionary with some useful information of the training process.

        """
        self.sample_weights = sample_weights
        
        # dict to history training info
        history = {'metrics_train':[],
                    'metrics_val':[]}
        
        # check exceptions
        self.__check_exception_fit()
        self.__check_shape(X_train)
        if X_val is not None: self.__check_shape(X_val)
        
        # reset metrics
        self.metrics_train.reset_states()
        self.metrics_val.reset_states()
        
        # check validation splits
        self.validation_strategy = validation_strategy
        X, y = self.__check_validation_split(X_train, y_train, X_val, y_val)
        
        # check splits
        self.tot_splits = 1 if self._val_train is None else self._val_train.get_n_splits(X, y, groups)
            
        # prepare progress bar
        self.prog.prepare(self.tot_splits, verbose=verbose)
        
        # evaluate
        if self._val_train is None:       
            # eval training data
            self.__eval_training_data(X, y, **kwargs)
            
            # update progress bar
            self.prog.update_bar(self.metrics_train.result())
            
        else:
            # split data
            self.tot_splits = self._val_train.get_n_splits(X, y, groups)
            for train_index, val_index in self._val_train.split(X, y, groups):
                # splits
                X_train, X_val, y_train, y_val = self.__split_data(X, y, train_index, val_index)
                
                # eval splits
                self.__eval_training_data(X_train, y_train, **kwargs)
                self.__eval_validation_data(X_val, y_val)
                
                # update progress bar
                results_train = self.metrics_train.result()
                results_val = self.metrics_val.result()
                self.prog.update_bar(results_train, results_val)
                # save results to dict
                history['metrics_train'].append(results_train)
                history['metrics_val'].append(results_val)
        self.fitted = True
        return history
        
    def __define_sample_weights(self, y_train):
        if self.sample_weights == 'balanced':
            sample_weights = get_samples_weight(y_train).reshape(-1)
        elif self.sample_weights is not None:
            sample_weights = self.sample_weights
        else:
            sample_weights = np.ones(shape=(len(y_train)))
        return sample_weights
        
    def __split_data(self, X, y, train_index, val_index):
        X_train = [X[i] for i in train_index]
        y_train = [y[i] for i in train_index]  
        X_val = [X[i] for i in val_index]
        y_val = [y[i] for i in val_index] 
        return X_train, X_val, y_train, y_val
        
    def __eval_training_data(self, X_train, y_train, **kwargs):
        # calculate sample weights
        sample_weights = self.__define_sample_weights(y_train)
        # fitting pipeline
        self.pipeline.fit(X_train, y_train, model__sample_weight=sample_weights, **kwargs)
        y_pred_train = self.pipeline.predict(X_train)
        # update metrics
        self.metrics_train.update_state(y_train, y_pred_train)
        
    def __eval_validation_data(self, X_val, y_val):
        # evaluating pipeline
        y_pred_val = self.pipeline.predict(X_val)
        # update metrics
        self.metrics_val.update_state(y_val, y_pred_val)
                    
    def __check_validation_split(self, X_train, y_train, X_val, y_val):
        # check if X_val is not None, if it is not, then we do the PredefinedSplit
        if X_val is not None:
            test_fold = np.zeros(len(X_train) + len(X_val))
            test_fold[:len(X_train)] = -1
            
            X_train = list(X_train.copy())
            y_train = list(y_train.copy())
            [X_train.append(bag) for bag in X_val]
            [y_train.append(label) for label in y_val]
            
            self._val_train = PredefinedSplit(test_fold)
        else:
            if self.validation_strategy is None:
                self._val_train = None
            else:
                self._val_train = self.validation_strategy
        return X_train, np.array(y_train).reshape(-1)
                
    def predict(self, X_test, **kwargs):
        """ Get the predictions for the test set

        Parameters
        ----------
        X_test : array-like containing all the testing bags

        Returns
        -------
        an array-like containing all the prediction for the test set.
        """
        self.__check_exception_predict()
        self.__check_shape(X_test)
        y_pred = self.pipeline.predict(X_test, **kwargs)
        return y_pred
        
    def predict_metrics(self, X_test, y_test):
        """ Get the metrics result for the test bags

        Parameters
        ----------
        X_test : array-like containing all the testing bags
        y_test : ground_truth labels of the test set.

        Returns
        -------
        a dict containing the evaluation metrics of the test set

        """
        # reset metrics
        self.metrics_test.reset_states()
        # get predictions
        y_pred = self.predict(X_test)
        # update metrics
        self.metrics_test.update_state(y_test, y_pred)
        return self.metrics_test.result()
        
    def get_positive_instances(self, X, **kwargs):
        """ Get instances with greater impact on the bag embedding

        Parameters
        ----------
        X : contains the bags to predict the positive instances

        Returns
        -------
        pos_ins : a list containing the indexs of the positive instances in X

        """
        pos_ins = None
        try:
            if len(self.pipeline) > 1:
                X = self.pipeline[:-1].transform(X)
            pos_ins = self.model.get_positive_instances(X, **kwargs)
        except Exception as e:
            print(e)
            raise Exception('model has not implemented get_positive_instances method')
        return pos_ins