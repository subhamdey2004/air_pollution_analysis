import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # Add project root to path

import unittest
import pandas as pd
from src import data_preprocessing as dp

class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'City': ['A', 'B'],
            'date': ['2025-10-01', '2025-10-02'],
            'PM2.5': [50, 60],
            'PM10': [80, 90]
        })

    def test_clean_air_quality(self):
        cleaned = dp.clean_air_quality(self.df)
        self.assertIn('City', cleaned.columns)
        self.assertIn('date', cleaned.columns)
        self.assertIn('PM2.5', cleaned.columns)
        self.assertIn('PM10', cleaned.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned['date']))

if __name__ == '__main__':
    unittest.main()
