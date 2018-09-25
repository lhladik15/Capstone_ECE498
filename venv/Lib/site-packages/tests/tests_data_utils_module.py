__author__ = "Adam Jarzebak"
__copyright__ = "Copyright 2018, Adam Jarzebak"
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Adam Jarzebak"
__email__ = "adam@jarzebak.eu"
"""
This file contains tests for data_utils module.
"""
import unittest
from data_lib.data_utils import *


class TestGridReferenceToNEConversion(unittest.TestCase):

    def setUp(self):
        self.grid_reference_1 = 'TQ1026048120'
        self.grid_reference_2 = 'SO9059703314'

    def testConversion_1(self):
        n, e = grid_reference_to_northing_easting(self.grid_reference_1)
        self.assertLessEqual((510260, 148120.0), (n, e))

    def testConversion_2(self):
        n, e = grid_reference_to_northing_easting(self.grid_reference_2)
        self.assertLessEqual((390597, 203314), (n, e))

    def tearDown(self):
        pass


class TestTemplate(unittest.TestCase):

    def setUp(self):
        pass

    def testTitle(self):
        self.assertIn('', '')

    def tearDown(self):
        pass
