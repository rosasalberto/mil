import unittest
import numpy as np

from mil.errors.custom_exceptions import DimensionError
from mil.bag_representation import *

class TestNonBagMapping(unittest.TestCase):
    def setUp(self):
        self.non_squared_bags = [[[10,2,4],[2,4,2]],
                                [[1,2,2]]]
        
        self.squared_bags = [[[10,2,4],[2,4,2]],
                            [[1,2,2],[3,4,4]]]
        
        self.other_non_squared_bags = [[[10,2,4],[2,2,2],[1,1,1]],
                                       [[1,2,2],[3,4,4]]]
        
        self.other_squared_bags = [[[10,2,4],[2,2,2],[1,1,1]],
                                   [[1,2,2],[3,4,4],[3,4,4]]]
        
        self.other_squared_bags_4d = [[[[10,2,4],[2,2,2],[1,1,1]],
                                    [[1,2,2],[3,4,4],[3,4,4]]],
                                    [[[10,2,4],[2,2,2],[1,1,1]],
                                    [[1,2,2],[3,4,4],[3,4,4]]]]
    
    def test_arithmetic_mean_when_passing_array_less_than_3D_raise_exception(self):
        bag_repr = ArithmeticMeanBagRepresentation()
        with self.assertRaises(DimensionError):
            bag_repr.fit_transform([2,3])
    
    def test_arithmetic_mean_when_passing_list_non_square_returns_correct_values(self):
        bag_repr = ArithmeticMeanBagRepresentation()
        bag_non_map = bag_repr.fit_transform(self.non_squared_bags)
        check = (bag_non_map == [[6,3,3],[1,2,2]]).all()
        self.assertTrue(check)
    
    def test_arithmetic_mean_when_passing_list_square_returns_correct_values(self):
        bag_repr = ArithmeticMeanBagRepresentation()
        bag_non_map = bag_repr.fit_transform(self.squared_bags)
        check = (bag_non_map == [[6,3,3],[2,3,3]]).all()
        self.assertTrue(check)
        
    def test_arithmetic_mean_when_passing_array_returns_correct_values(self):
        bag_repr = ArithmeticMeanBagRepresentation()
        bag_non_map = bag_repr.fit_transform(np.array(self.squared_bags))
        check = (bag_non_map == [[6,3,3],[2,3,3]]).all()
        self.assertTrue(check)
        
    def test_arithmetic_mean_when_passing_array_4d_returns_correct_values(self):
        bag_repr = ArithmeticMeanBagRepresentation()
        bag_non_map = bag_repr.fit_transform(np.array(self.other_squared_bags_4d))
        expected = [[[5.5, 2.,  3. ],
                     [2.5, 3.,  3. ],
                     [2.,  2.5, 2.5]],
                    [[5.5, 2.,  3. ],
                     [2.5, 3.,  3. ],
                     [2.,  2.5, 2.5]]],
        check = (bag_non_map == expected).all()
        self.assertTrue(check)
        
    def test_geometric_mean_when_passing_list_non_square_returns_correct_values(self):
        bag_repr = GeometricMeanBagRepresentation()
        expected = np.array([[np.sqrt(20),np.sqrt(8),np.sqrt(8)],[1,2,2]])
        bag_non_map = bag_repr.fit_transform(self.non_squared_bags)
        check = np.isclose(bag_non_map, expected).all()
        self.assertTrue(check)
    
    def test_geometric_mean_when_passing_list_square_returns_correct_values(self):
        bag_repr = GeometricMeanBagRepresentation()
        expected = np.array([[np.sqrt(20),np.sqrt(8),np.sqrt(8)],[np.sqrt(3),np.sqrt(8),np.sqrt(8)]])
        bag_non_map = bag_repr.fit_transform(self.squared_bags)
        check = np.isclose(bag_non_map, expected).all()
        self.assertTrue(check)
        
    def test_geometric_mean_when_passing_array_returns_correct_values(self):
        bag_repr = GeometricMeanBagRepresentation()
        expected = np.array([[np.sqrt(20),np.sqrt(8),np.sqrt(8)],[np.sqrt(3),np.sqrt(8),np.sqrt(8)]])
        bag_non_map = bag_repr.fit_transform(self.squared_bags)
        check = np.isclose(bag_non_map, expected).all()
        self.assertTrue(check)
        
    def test_min_when_passing_list_non_square_returns_correct_values(self):
        bag_repr = MinBagRepresentation()
        expected = [[2,2,2],[1,2,2]]
        bag_non_map = bag_repr.fit_transform(self.non_squared_bags)
        check = np.isclose(bag_non_map, expected).all()
        self.assertTrue(check)
    
    def test_min_when_passing_list_square_returns_correct_values(self):
        bag_repr = MinBagRepresentation()
        expected = [[2,2,2],[1,2,2]]
        bag_non_map = bag_repr.fit_transform(self.squared_bags)
        check = np.isclose(bag_non_map, expected).all()
        self.assertTrue(check)
        
    def test_min_when_passing_array_returns_correct_values(self):
        bag_repr = MinBagRepresentation()
        expected = [[2,2,2],[1,2,2]]
        bag_non_map = bag_repr.fit_transform(self.squared_bags)
        check = np.isclose(bag_non_map, expected).all()
        self.assertTrue(check)
        
    def test_max_when_passing_list_non_square_returns_correct_values(self):
        bag_repr = MaxBagRepresentation()
        expected = [[10,4,4],[1,2,2]]
        bag_non_map = bag_repr.fit_transform(self.non_squared_bags)
        check = np.isclose(bag_non_map, expected).all()
        self.assertTrue(check)
    
    def test_max_when_passing_list_square_returns_correct_values(self):
        bag_repr = MaxBagRepresentation()
        expected = [[10,4,4],[3,4,4]]
        bag_non_map = bag_repr.fit_transform(self.squared_bags)
        check = np.isclose(bag_non_map, expected).all()
        self.assertTrue(check)
        
    def test_max_when_passing_array_returns_correct_values(self):
        bag_repr = MaxBagRepresentation()
        expected = [[10,4,4],[3,4,4]]
        bag_non_map = bag_repr.fit_transform(self.squared_bags)
        check = np.isclose(bag_non_map, expected).all()
        self.assertTrue(check)
        
    def test_meanminmax_when_passing_list_non_square_returns_correct_values(self):
        bag_repr = MeanMinMaxBagRepresentation()
        expected = [[5.5,1.5,2.5],[2,3,3]]
        bag_non_map = bag_repr.fit_transform(self.other_non_squared_bags)
        check = np.isclose(bag_non_map, expected).all()
        self.assertTrue(check)
    
    def test_meanminmax_when_passing_list_square_returns_correct_values(self):
        bag_repr = MeanMinMaxBagRepresentation()
        expected = [[5.5,1.5,2.5],[2,3,3]]
        bag_non_map = bag_repr.fit_transform(self.other_squared_bags)
        check = np.isclose(bag_non_map, expected).all()
        self.assertTrue(check)
        
    def test_meanminmax_when_passing_array_returns_correct_values(self):
        bag_repr = MeanMinMaxBagRepresentation()
        expected = [[5.5,1.5,2.5],[2,3,3]]
        bag_mapped = bag_repr.fit_transform(self.other_squared_bags)
        check = np.isclose(bag_mapped, expected).all()
        self.assertTrue(check)
        
if __name__ == '__main__':
    unittest.main()