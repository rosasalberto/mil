from mil.metrics import *
from mil.errors.custom_exceptions import ExpectedListError
from mil.errors.custom_warnings import invalid_metric_string

class MetricsManager:
    def __init__(self, metrics=[]):
        self.check_exceptions(metrics)
        self.import_metrics(metrics)
            
    def check_exceptions(self, metrics):
        if type(metrics) != list:
            raise ExpectedListError
            
    def import_metrics(self, metrics_):
        metrics = metrics_.copy()
        self.metrics = {}
        
        # Complete all list of possible imports 
        if 'acc' in metrics:
            metrics.remove('acc')
            self.metrics['accuracy'] = Accuracy()
        if 'spec' in metrics:
            metrics.remove('spec')
            self.metrics['specificity'] = Specificity()
        if 'sens' in metrics:
            metrics.remove('sens')
            self.metrics['sensibility'] = Sensibility()
        if 'ppv' in metrics:
            metrics.remove('ppv')
            self.metrics['ppv'] = PPV()
        if 'npv' in metrics:
            metrics.remove('npv')
            self.metrics['npv'] =  NPV()
            
        # Adding non string metrics
        for m in metrics:
            if type(m) == str: 
                invalid_metric_string()
            elif str(type(m)).split('.')[0].split("'")[-1] == 'mil':
                self.metrics[type(m).__name__.lower()] = m()
            else:
                self.metrics[m.__name__.lower()] = m()
            
    def update_state(self, y_true, y_pred, sample_weight=None):
        for key in self.metrics:
            self.metrics[key].update_state(y_true, y_pred, sample_weight)
            
    def result(self):
        result = {}
        for key in self.metrics:
            result[key] = self.metrics[key].result().numpy()
        return result
            
    def reset_states(self):
        for key in self.metrics:
            self.metrics[key].reset_states()