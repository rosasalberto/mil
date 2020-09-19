import unittest

from mil.data.datasets import musk1, musk2, protein, elephant, corel_dogs, \
                              ucsb_breast_cancer, web_recommendation_1, birds_brown_creeper, \
                              mnist_bags

class TestLoadDatasets(unittest.TestCase):    
    def test_load_musk1(self):
        (x_train, y_train), (x_test, y_test) = musk1.load()
        self.assertEqual(len(x_train[0][0]), 166)
        self.assertEqual(len(y_train), 73)
        
    def test_load_musk2(self):
        (x_train, y_train), (x_test, y_test) = musk2.load()
        self.assertEqual(len(x_train[0][0]), 166)
        self.assertEqual(len(y_train), 81)

    def test_load_protein(self):
        (x_train, y_train), (x_test, y_test) = protein.load()
        self.assertEqual(len(x_train[0][0]), 9)
        self.assertEqual(len(y_train), 154)

    def test_load_elephant(self):
        (x_train, y_train), (x_test, y_test) = elephant.load()
        self.assertEqual(len(x_train[0][0]), 230)
        self.assertEqual(len(y_train), 160)

    def test_load_corel_dogs(self):
        (x_train, y_train), (x_test, y_test) = corel_dogs.load()
        self.assertEqual(len(x_train[0][0]), 9)
        self.assertEqual(len(y_train), 1600)

    def test_load_ucsb_breast_cancer(self):
        (x_train, y_train), (x_test, y_test) = ucsb_breast_cancer.load()
        self.assertEqual(len(x_train[0][0]), 708)
        self.assertEqual(len(y_train), 46)

    def test_load_web_recommendation_1(self):
        (x_train, y_train), (x_test, y_test) = web_recommendation_1.load()
        self.assertEqual(len(x_train[0][0]), 5863)
        self.assertEqual(len(y_train), 60)

    def test_load_birds_brown_creeper(self):
        (x_train, y_train), (x_test, y_test) = birds_brown_creeper.load()
        self.assertEqual(len(x_train[0][0]), 38)
        self.assertEqual(len(y_train), 438)

    def test_load_mnist_bags(self):
        (x_train, y_train, _), (x_test, y_test, _)  = mnist_bags.load()
        (x_train, y_train, _), (x_test, y_test, _)  = mnist_bags.load_42()
        (x_train, y_train, _), (x_test, y_test, _)  = mnist_bags.load_2_and_3()
        self.assertEqual(len(x_train), 200)
        self.assertEqual(len(x_test), 100)
        
if __name__ == '__main__':
    unittest.main()