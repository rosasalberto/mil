import unittest

from mil.bag_representation import ArithmeticMeanBagRepresentation
from mil.models import MILES, SVC, AttentionDeepPoolingMil
from mil.trainer.trainer import Trainer
from mil.validators import PredefinedSplit, LeaveOneGroupOut
from mil.errors.custom_exceptions import DimensionError, FitNonCalledError, PrepareNonCalledError
from mil.utils.padding import Padding
from mil.data.datasets import mnist_bags

class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.X_train = [[[1,1], [0,0], [2,2]],
                        [[0,1], [0,2], [1,2]],
                        [[1,2], [1,0], [0,2]],
                        [[1,1], [0,0]]]
        
        self.X_val = [[[0,1], [1,0], [2,2]],
                      [[0,2], [2,0], [1,2]],
                      [[2,1], [1,0]]]
        
        self.y_train = [1, 1, 0, 0]
        self.y_val = [1, 1, 0]
        
        self.trainer = Trainer()
        
    def test_when_calling_fit_before_preparing_raise_error(self):
        with self.assertRaises(PrepareNonCalledError):
            self.trainer.fit(self.X_train, self.y_train)
            
    def test_when_calling_predict_before_fit_raise_error(self):
        with self.assertRaises(FitNonCalledError):
            self.trainer.predict(self.X_val)
            
    def test_when_calling_fit_or_predict_with_wrong_dimension_raise_error(self):
        self.trainer.prepare(model=SVC())
        with self.assertRaises(DimensionError):
            self.trainer.fit([[0,1,2]],[0])
            self.trainer.predict([0,1])
    
    def test_when_fitting_with_none_validation_and_not_validation_strategy_splits_once(self):
        self.trainer.prepare(model=MILES())
        self.trainer.fit(self.X_train, self.y_train, verbose=0)
        self.assertEqual(self.trainer.tot_splits, 1)
        
    def test_when_fitting_with_validation_splits_once(self):
        self.trainer.prepare(model=MILES())
        self.trainer.fit(self.X_train, self.y_train, self.X_val, self.y_val, verbose=0)
        self.assertEqual(self.trainer.tot_splits, 1)
        
    def test_when_fitting_with_no_validation_and_validation_strat_splits_correct(self):
        validation_strat = PredefinedSplit(test_fold=[1,0,1,0])
        self.trainer.prepare(model=MILES())
        self.trainer.fit(self.X_train, self.y_train, 
                         validation_strategy=validation_strat, verbose=0)
        self.assertEqual(self.trainer.tot_splits, 2)
        
    def test_when_fitting_with_validation_and_validation_strat_splits_once(self):
        validation_strat = PredefinedSplit(test_fold=[-1,1,0,0])
        self.trainer.prepare(model=MILES())
        self.trainer.fit(self.X_train, self.y_train, self.X_val, self.y_val, 
                         validation_strategy=validation_strat, verbose=0)
        self.assertEqual(self.trainer.tot_splits, 1)
        
    def test_when_fitting_validation_strat_leave_group_out_splits_correct(self):
        validation_strat = LeaveOneGroupOut()
        groups = [0,1,0,1]
        self.trainer.prepare(model=MILES())
        self.trainer.fit(self.X_train, self.y_train, groups=groups, 
                         validation_strategy=validation_strat, verbose=0)
        self.assertEqual(self.trainer.tot_splits, 2)
        
    def test_when_prepare_with_metrics_then_metrics_are_calculated(self):
        validation_strat = LeaveOneGroupOut()
        groups = [0,1,0,1]
        self.trainer.prepare(model=MILES(), metrics=['acc','spec'])
        self.trainer.fit(self.X_train, self.y_train, groups=groups, 
                         validation_strategy=validation_strat, verbose=0)
        n_keys_train = len(self.trainer.metrics_train.metrics.keys())
        n_keys_val = len(self.trainer.metrics_train.metrics.keys())
        self.assertEqual(n_keys_train, 2)
        self.assertEqual(n_keys_val, 2)
        
    def test_when_prepare_with_bag_representation_and_model_fits_correctly(self):
        validation_strat = LeaveOneGroupOut()
        groups = [0,1,0,1]
        pipeline = [('bag_representation', ArithmeticMeanBagRepresentation())]
        self.trainer.prepare(model=SVC(class_weight='balanced'), 
                              preprocess_pipeline=pipeline,
                              metrics=['acc','spec'])
        self.trainer.fit(self.X_train, self.y_train, groups=groups, 
                         validation_strategy=validation_strat, verbose=0)
        y_pred = self.trainer.predict(self.X_train)
        self.assertEqual(len(y_pred), 4)
        
    def test_when_fitting_keras_model_all_correct_with_padding(self):
        (bags_train, y_train, _), (bags_test, y_test, _) = mnist_bags.load()
        pipeline = [('padding', Padding())]
        self.trainer.prepare(model=AttentionDeepPoolingMil(), 
                              preprocess_pipeline=pipeline,
                              metrics=['acc','spec'])
        self.trainer.fit(bags_train, y_train, verbose=0, model__verbose=0)
        y_pred = self.trainer.predict(bags_train)
        self.assertEqual(len(y_pred), 200)
        
if __name__ == '__main__':
    unittest.main()