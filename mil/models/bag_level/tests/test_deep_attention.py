import unittest
import numpy as np

from mil.models import AttentionDeepPoolingMil
from mil.utils.padding import Padding

class TestAttentionDeepPoolingMil(unittest.TestCase):
    def setUp(self):        
        self.training_bag = np.random.normal(0, 1, (30, 3, 28, 28, 1))
        self.training_label = np.zeros(30)
        self.training_label[15:] = 1
        self.model = AttentionDeepPoolingMil()
    
    def test_when_fitting_keras_model_all_correct_with_padding(self):
        pipeline = [('padding', Padding())]
        self.model.fit(self.training_bag, self.training_label, verbose=0, epochs=10)
        y_pred = self.model.predict(self.training_bag)
        self.assertEqual(len(y_pred), 30)
        
    def test_when_fitting_with_2d_all_correct(self):
        """ when modelling with [bags, instances, features] shape works well """
        training_bag = np.random.normal(0, 1, (30, 10, 28))
        self.model.fit(training_bag, self.training_label, verbose=0, epochs=10)
        y_pred = self.model.predict(training_bag)
        self.assertEqual(len(y_pred), 30)
        
    def test_when_fitting_with_3d_all_correct(self):
        """ when modelling with [bags, instances, features, features] shape works well """
        training_bag = np.random.normal(0, 1, (30, 10, 28, 28))
        self.model.fit(training_bag, self.training_label, verbose=0, epochs=10)
        y_pred = self.model.predict(training_bag)
        self.assertEqual(len(y_pred), 30)
        
    def test_when_fitting_with_4d_all_correct(self):
        """ when modelling with [bags, instances, features, features, features] shape works well """
        training_bag = np.random.normal(0, 1, (30, 10, 28, 28, 1))
        self.model.fit(training_bag, self.training_label, verbose=0, epochs=10)
        y_pred = self.model.predict(training_bag)
        self.assertEqual(len(y_pred), 30)
        
    def test_when_fitting_with_5d_all_correct(self):
        """ when modelling with [bags, instances, features, features, features, features] shape works well """
        training_bag = np.random.normal(0, 1, (30, 10, 28, 28, 20, 1))
        self.model.fit(training_bag, self.training_label, verbose=0, epochs=10)
        y_pred = self.model.predict(training_bag)
        self.assertEqual(len(y_pred), 30)
    
if __name__ == '__main__':
    unittest.main()