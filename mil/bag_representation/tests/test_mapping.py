import unittest
import numpy as np

from mil.errors.custom_exceptions import DimensionError
from mil.bag_representation import MILESMapping

class TestBagMapping(unittest.TestCase):
    def setUp(self):        
        self.other_squared_bags = [[[10,2,4],[2,2,2],[1,1,1]],
                                   [[1,2,2],[3,4,4],[3,4,4]]]
    
    def test_miles_mapping_when_passing_array_different_than_3D_raise_exception(self):
        bag_repr = MILESMapping()
        with self.assertRaises(DimensionError):
            bag_repr.fit_transform([2,3])
            
    def test_miles_mapping_gives_correct_mapping(self):
        expected = [[1., 1., 1., 0.95181678, 0.64118039,0.64118039],
                    [0.07300087, 0.95181678, 0.90595519, 1., 1.,1.]]
        bag_repr = MILESMapping()
        bag_map = bag_repr.fit_transform(self.other_squared_bags)
        check = np.isclose(bag_map, expected).all()
        self.assertTrue(check)
    
if __name__ == '__main__':
    unittest.main()