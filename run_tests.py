import unittest

from mil.bag_representation.tests.test_mapping import TestBagMapping
from mil.bag_representation.tests.test_non_mapping import TestNonBagMapping
from mil.metrics.tests.test_custom import TestSpecificity, TestSensibility, TestPPV, TestNPV
from mil.metrics.tests.test_manager import TestMetricsManager
from mil.trainer.tests.test_trainer import TestTrainer
from mil.models.bag_level.tests.test_miles import TestMILES
from mil.models.bag_level.tests.test_deep_attention import TestAttentionDeepPoolingMil
from mil.data.datasets.tests.test_load_datasets import TestLoadDatasets

def suite_bag_representation():
    suite = unittest.TestSuite()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBagMapping)
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestNonBagMapping))
    return suite

def suite_metrics():
    suite = unittest.TestSuite()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSpecificity)
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestSensibility))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestPPV))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestNPV))
    return suite

def suite_metrics_manager():
    suite = unittest.TestSuite()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMetricsManager)
    return suite

def suite_trainer():
    suite = unittest.TestSuite()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTrainer)
    return suite

def suite_models():
    suite = unittest.TestSuite()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMILES)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAttentionDeepPoolingMil)
    return suite
    
def suite_data():
    suite = unittest.TestSuite()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLoadDatasets)
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite_bag_representation())
    runner.run(suite_metrics())
    runner.run(suite_metrics_manager())
    runner.run(suite_trainer())
    runner.run(suite_models())
    runner.run(suite_data())