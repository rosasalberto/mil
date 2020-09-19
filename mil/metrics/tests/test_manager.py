import unittest

from mil.metrics import Specificity, AUC, BinaryAccuracy
from mil.metrics.manager import MetricsManager
from mil.errors.custom_exceptions import ExpectedListError

class TestMetricsManager(unittest.TestCase):    
    def test_check_exceptions_when_metrics_is_not_list_raise_exception(self):
        metrics = 'acc'
        with self.assertRaises(ExpectedListError):
            MetricsManager(metrics=metrics)
            
    def test_result_return_correct_metrics_dict_when_no_metrics(self):
        metrics_manager = MetricsManager()
        d = metrics_manager.result()
        keys = d.keys()
        self.assertEqual(len(keys), 0)
        
    def test_result_when_passing_more_than_1_tf_metrics_returns_same_number_of_metrics(self):
        metrics = [AUC, BinaryAccuracy]
        metrics_manager = MetricsManager(metrics=metrics)
        d = metrics_manager.result()
        keys = d.keys()
        self.assertEqual(len(keys), 2)
            
    def test_result_return_correct_metrics_dict(self):
        metrics = ['acc', Specificity]
        metrics_manager = MetricsManager(metrics=metrics)
        d = metrics_manager.result()
        keys = d.keys()
        check = ('accuracy' in keys) and ('specificity' in keys)
        self.assertTrue(check)
        self.assertEqual(len(keys), 2)
        
    def test_result_return_correct_results(self):
        y_true = [1,1,1,0,0,0]
        y_pred = [1,1,1,0,0,0]
        metrics = ['acc',Specificity,'sens']
        metrics_manager = MetricsManager(metrics=metrics)
        metrics_manager.update_state(y_true, y_pred)
        d = metrics_manager.result()
        self.assertEqual(d['accuracy'], 1)
        self.assertEqual(d['specificity'], 1)
        self.assertEqual(d['sensibility'], 1)
        
    def test_result_return_correct_results_when_reseting_state(self):
        y_true = [1,1,1,0,0,0]
        y_pred = [1,1,1,0,0,0]
        metrics = ['acc',Specificity,'sens']
        metrics_manager = MetricsManager(metrics=metrics)
        metrics_manager.update_state(y_true, y_pred)
        metrics_manager.reset_states()
        
        y_true = [1,1,1,0,0,0]
        y_pred = [0,0,0,1,1,1]
        metrics_manager.update_state(y_true, y_pred)
        d = metrics_manager.result()
        self.assertEqual(d['accuracy'], 0)
        self.assertEqual(d['specificity'], 0)
        self.assertEqual(d['sensibility'], 0)
        
    def test_result_return_correct_results_when_updating_state_twice(self):
        y_true = [1,1,1,0,0,0]
        y_pred = [1,1,1,0,0,0]
        metrics = ['acc',Specificity,'sens']
        metrics_manager = MetricsManager(metrics=metrics)
        metrics_manager.update_state(y_true, y_pred)
        
        y_true = [1,1,1,0,0,0]
        y_pred = [0,0,0,1,1,1]
        metrics_manager.update_state(y_true, y_pred)
        d = metrics_manager.result()
        self.assertEqual(d['accuracy'], 0.5)
        self.assertEqual(d['specificity'], 0.5)
        self.assertEqual(d['sensibility'], 0.5)
        
if __name__ == '__main__':
    unittest.main()