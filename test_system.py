"""
test_system.py
End-to-end automated verification script for the EV Battery Life & Health Prediction System.
"""

import os
import json
import unittest
from app import app
from ml_pipeline.model_engine import BatteryHealthEngine

class TestEVBatterySystem(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.engine = BatteryHealthEngine()

    def test_01_model_engine_ready(self):
        self.assertTrue(self.engine.is_ready, "ML Engine should have loaded SoH and RUL models.")

    def test_02_prediction_ranges(self):
        sample = {
            'voltage': 3.75,
            'current': 1.5,
            'temperature': 28.0,
            'capacity': 1.82,
            'cycle': 120,
            'time': 432000,
            'resistance': 70.0,
            'impedance': 0.052,
            'soc': 80.0,
            'age': 90.0
        }
        res = self.engine.predict_single(sample)
        self.assertIn('predictions', res)
        preds = res['predictions']
        self.assertGreaterEqual(preds['soh_predicted'], 50.0)
        self.assertLessEqual(preds['soh_predicted'], 105.0)
        self.assertGreaterEqual(preds['rul_cycles'], 0)
        self.assertIn(preds['health_status'].split()[0], ['EXCELLENT', 'GOOD', 'DEGRADED', 'CRITICAL'])
        self.assertGreater(len(res['advisories']), 0)
        self.assertGreater(len(res['forecast']['cycles']), 0)

    def test_03_web_routes(self):
        # 1. Main diagnostic page
        res1 = self.app.get('/')
        self.assertEqual(res1.status_code, 200)
        self.assertIn(b'Battery Telemetry', res1.data)

        # 2. Batch upload page
        res2 = self.app.get('/batch')
        self.assertEqual(res2.status_code, 200)
        self.assertIn(b'NASA & CALCE', res2.data)

        # 3. Analytics page
        res3 = self.app.get('/analytics')
        self.assertEqual(res3.status_code, 200)
        self.assertIn(b'Exploratory Data Analysis', res3.data)

    def test_04_api_predict_endpoint(self):
        payload = {
            'voltage': 3.65,
            'current': 1.8,
            'temperature': 32.0,
            'capacity': 1.70,
            'cycle': 160,
            'time': 576000,
            'resistance': 82.0,
            'impedance': 0.065,
            'soc': 70.0,
            'age': 120.0
        }
        res = self.app.post('/api/predict', json=payload)
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data['success'])
        self.assertIn('predictions', data['data'])

    def test_05_api_analytics_data(self):
        res = self.app.get('/api/analytics-data')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data['success'])
        self.assertIn('trajectory', data)
        self.assertIn('correlations', data)

    def test_06_api_retrain(self):
        res = self.app.post('/api/retrain', json={'filename': 'sample_nasa_b0005_test.csv'})
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data['success'])
        self.assertIn('metrics', data)
        self.assertIn('soh', data['metrics'])

if __name__ == '__main__':
    unittest.main()
