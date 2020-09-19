import unittest

import numpy as np

from mil.metrics import Specificity, Sensibility, PPV, NPV

class TestSpecificity(unittest.TestCase):
    def setUp(self):
        self.specificity = Specificity()
        
    def test_specificity_when_all_real_values_are_positive_return_nan(self):
        y_real = [1,1,1,1,1]
        y_pred = [1,1,0,0,1]
        self.specificity.update_state(y_real, y_pred)
        score = self.specificity.result()
        check = np.isnan(score)
        self.assertTrue(check)
        
    def test_specificity_when_perfect_specificity_gives_correct_value(self):
        y_real = [1,0,0,0,0,0]
        y_pred = [1,0,0,0,0,0]
        self.specificity.update_state(y_real, y_pred)
        score = self.specificity.result()
        self.assertEqual(score, 1)

    def test_specificity_gives_correct_value(self):
        y_real = [1,0,0,0,0,0]
        y_pred = [1,0,0,0,0,1]
        self.specificity.update_state(y_real, y_pred)
        score = self.specificity.result()
        self.assertEqual(score, 0.8)
        
    def test_specificity_gives_correct_value_when_updating_twice(self):
        y_real = [1,0,0,0,0,0]
        y_pred = [1,0,0,0,0,1]
        self.specificity.update_state(y_real, y_pred)
        y_real = [1,0,0,0,0,0]
        y_pred = [1,0,0,0,0,0]
        self.specificity.update_state(y_real, y_pred)
        score = self.specificity.result()
        self.assertEqual(score, 0.9)
        
    def test_specificity_reseting_states_gives_nan(self):
        y_real = [1,0,0,0,0,0]
        y_pred = [1,0,0,0,0,1]
        self.specificity.update_state(y_real, y_pred)
        self.specificity.reset_states()
        score = self.specificity.result()
        check = np.isnan(score)
        self.assertTrue(check)
        
    def test_specificity_reseting_states_and_updating_later_gives_correct_result(self):
        y_real = [1,0,0,0,0,0]
        y_pred = [1,0,0,0,0,1]
        self.specificity.update_state(y_real, y_pred)
        self.specificity.reset_states()
        y_real = [1,0,0,0,0,0]
        y_pred = [1,0,0,0,0,1]
        self.specificity.update_state(y_real, y_pred)
        score = self.specificity.result()
        self.assertEqual(score, 0.8)
        
class TestSensibility(unittest.TestCase):
    def setUp(self):
        self.sensibility = Sensibility()
        
    def test_sensibility_when_all_real_values_are_negative_return_nan(self):
        y_real = [0,0,0,0,0]
        y_pred = [1,1,0,0,1]
        self.sensibility.update_state(y_real, y_pred)
        score = self.sensibility.result()
        check = np.isnan(score)
        self.assertTrue(check)
        
    def test_sensibility_when_perfect_sensibility_gives_correct_value(self):
        y_real = [0,1,1,1,1,1]
        y_pred = [0,1,1,1,1,1]
        self.sensibility.update_state(y_real, y_pred)
        score = self.sensibility.result()
        self.assertEqual(score, 1)

    def test_sensibility_gives_correct_value(self):
        y_real = [1,1,1,1,0,1]
        y_pred = [1,1,1,1,1,0]
        self.sensibility.update_state(y_real, y_pred)
        score = self.sensibility.result()
        self.assertEqual(score, 0.8)
        
    def test_sensibility_gives_correct_value_when_updating_twice(self):
        y_real = [1,1,1,1,0,1]
        y_pred = [1,1,1,1,1,0]
        self.sensibility.update_state(y_real, y_pred)
        y_real = [0,1,1,1,1,1]
        y_pred = [0,1,1,1,1,1]
        self.sensibility.update_state(y_real, y_pred)
        score = self.sensibility.result()
        self.assertEqual(score, 0.9)
        
    def test_sensibility_reseting_states_gives_nan(self):
        y_real = [0,1,1,1,1,1]
        y_pred = [0,1,1,1,1,1]
        self.sensibility.update_state(y_real, y_pred)
        self.sensibility.reset_states()
        score = self.sensibility.result()
        check = np.isnan(score)
        self.assertTrue(check)
        
    def test_sensibility_reseting_states_and_updating_later_gives_correct_result(self):
        y_real = [1,1,1,1,0,1]
        y_pred = [1,1,1,1,1,0]
        self.sensibility.update_state(y_real, y_pred)
        self.sensibility.reset_states()
        y_real = [1,1,1,1,0,1]
        y_pred = [1,1,1,1,1,0]
        self.sensibility.update_state(y_real, y_pred)
        score = self.sensibility.result()
        self.assertEqual(score, 0.8)
        
class TestPPV(unittest.TestCase):
    def setUp(self):
        self.ppv = PPV()
        
    def test_ppv_when_all_prediction_values_are_negative_return_nan(self):
        y_real = [1,1,0,0,1]
        y_pred = [0,0,0,0,0]
        self.ppv.update_state(y_real, y_pred)
        score = self.ppv.result()
        check = np.isnan(score)
        self.assertTrue(check)

    def test_ppv_gives_correct_value(self):
        y_real = [1,1,1,1,1,0]
        y_pred = [1,1,1,1,0,1]
        self.ppv.update_state(y_real, y_pred)
        score = self.ppv.result()
        self.assertEqual(score, 0.8)
        
    def test_ppv_gives_correct_value_when_updating_twice(self):
        y_real = [1,1,1,1,1,0]
        y_pred = [1,1,1,1,0,1]
        self.ppv.update_state(y_real, y_pred)
        y_real = [1,1,1,1,1,1]
        y_pred = [1,1,1,1,0,1]
        self.ppv.update_state(y_real, y_pred)
        score = self.ppv.result()
        self.assertEqual(score, 0.9)
        
    def test_ppv_reseting_states_gives_nan(self):
        y_real = [1,1,1,1,1,0]
        y_pred = [1,1,1,1,0,1]
        self.ppv.update_state(y_real, y_pred)
        self.ppv.reset_states()
        score = self.ppv.result()
        check = np.isnan(score)
        self.assertTrue(check)
        
    def test_ppv_reseting_states_and_updating_later_gives_correct_result(self):
        y_real = [1,1,1,1,0,1]
        y_pred = [1,1,1,1,1,0]
        self.ppv.update_state(y_real, y_pred)
        self.ppv.reset_states()
        y_real = [1,1,1,1,1]
        y_pred = [1,1,1,1,1]
        self.ppv.update_state(y_real, y_pred)
        score = self.ppv.result()
        self.assertEqual(score, 1)
        
class TestNPV(unittest.TestCase):
    def setUp(self):
        self.npv = NPV()
        
    def test_npv_when_all_prediction_values_are_positive_return_nan(self):
        y_real = [1,1,0,0,1]
        y_pred = [1,1,1,1,1]
        self.npv.update_state(y_real, y_pred)
        score = self.npv.result()
        check = np.isnan(score)
        self.assertTrue(check)

    def test_npv_gives_correct_value(self):
        y_real = [0,0,0,0,0,1]
        y_pred = [0,0,0,0,1,0]
        self.npv.update_state(y_real, y_pred)
        score = self.npv.result()
        self.assertEqual(score, 0.8)
        
    def test_npv_gives_correct_value_when_updating_twice(self):
        y_real = [1,1,0,0,1]
        y_pred = [1,1,1,1,1]
        self.npv.update_state(y_real, y_pred)
        y_real = [0,0,0,0,0,1]
        y_pred = [0,0,0,0,1,0]
        self.npv.update_state(y_real, y_pred)
        score = self.npv.result()
        self.assertEqual(score, 0.8)
        
    def test_npv_reseting_states_gives_nan(self):
        y_real = [1,1,1,1,1,0]
        y_pred = [1,1,1,1,0,1]
        self.npv.update_state(y_real, y_pred)
        self.npv.reset_states()
        score = self.npv.result()
        check = np.isnan(score)
        self.assertTrue(check)
        
    def test_npv_reseting_states_and_updating_later_gives_correct_result(self):
        y_real = [0,0,0,0,0,1]
        y_pred = [0,0,0,0,1,0]
        self.npv.update_state(y_real, y_pred)
        self.npv.reset_states()
        y_real = [0,1,0,0,0,0]
        y_pred = [0,1,0,0,0,0]
        self.npv.update_state(y_real, y_pred)
        score = self.npv.result()
        self.assertEqual(score, 1)

if __name__ == '__main__':
    unittest.main()