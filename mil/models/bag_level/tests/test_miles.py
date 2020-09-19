import unittest
import numpy as np

from mil.errors.custom_exceptions import DimensionError
from mil.models import MILES

class TestMILES(unittest.TestCase):
    def setUp(self):        
        self.training_bag = np.random.normal(0, 1, (30,3,10))
        self.training_label = np.zeros(30)
        self.training_label[15:] = 1
        self.model = MILES()
    
    def test_miles_when_fit_array_less_than_3D_raise_exception(self):
        with self.assertRaises(DimensionError):
            self.model.fit([2,3],[0,0])
            
    def test_miles_when_predict_array_less_than_3D_raise_exception(self):
        with self.assertRaises(DimensionError):
            self.model.predict([2,3])
            
    def test_miles_gives_correct_coefs_shape(self):
        self.model.fit(self.training_bag,self.training_label)
        coefs = self.model.coef_
        # the number of coefs corresponds to the total number of instances.
        self.assertEqual(len(coefs), 90)
    
if __name__ == '__main__':
    unittest.main()