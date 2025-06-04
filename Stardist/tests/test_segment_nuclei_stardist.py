import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import numpy as np
from pathlib import Path
import sys

# Temporarily adjust sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# We need to ensure that utils_stardist can be imported.
# It's assumed that csbdeep.utils.normalize is available.
# TensorFlow dependent parts in utils_stardist are expected to be commented out for this test.
from Stardist.utils_stardist import segment_nuclei, get_stardist_model

# Mock stardist.models directly here for more control if utils_stardist.py itself tries to import them
# However, get_stardist_model handles its own imports of StarDist2D/3D
# So, we mainly need to mock the instances returned by get_stardist_model

class MockStarDistModel:
    def __init__(self, name="mock_model"):
        self.name = name
        # Mock predict_instances and predict_instances_big
        self.predict_instances = MagicMock(return_value=(np.array([[1,1],[1,1]]), {}))
        self.predict_instances_big = MagicMock(return_value=(np.array([[2,2],[2,2]]), {}))

    # If the model instance itself is checked for properties like _is_tensorflow
    # (though stardist models don't usually expose this directly for users)
    # or if other methods are called, they might need mocking.
    # For this test, we primarily care about the two prediction methods.

class TestSegmentNucleiStardist(unittest.TestCase):

    def setUp(self):
        self.dummy_image_2d = np.random.rand(100, 100).astype(np.float32)
        self.dummy_image_3d = np.random.rand(10, 100, 100).astype(np.float32)
        self.mock_model_instance = MockStarDistModel()

    @patch('Stardist.utils_stardist.normalize') # Mock normalize as it's from csbdeep
    @patch('Stardist.utils_stardist.get_stardist_model') # We'll control the model returned
    def test_segment_nuclei_predict_instances_succeeds_2d(self, mock_get_model, mock_normalize):
        mock_normalize.return_value = self.dummy_image_2d # Normalize returns the same image for simplicity
        mock_get_model.return_value = self.mock_model_instance

        expected_labels = np.array([[1,1],[1,1]])
        self.mock_model_instance.predict_instances.return_value = (expected_labels, {})

        labels = segment_nuclei(self.dummy_image_2d, segmentation_type="2D", model=self.mock_model_instance, n_tiles=(4,4))

        self.mock_model_instance.predict_instances.assert_called_once()
        # Check n_tiles passed to predict_instances (should be 2D version)
        args, kwargs = self.mock_model_instance.predict_instances.call_args
        self.assertEqual(kwargs.get('n_tiles'), (4,4))
        self.mock_model_instance.predict_instances_big.assert_not_called()
        self.assertTrue(np.array_equal(labels, expected_labels))

    @patch('Stardist.utils_stardist.normalize')
    @patch('Stardist.utils_stardist.get_stardist_model')
    def test_segment_nuclei_predict_instances_succeeds_3d(self, mock_get_model, mock_normalize):
        mock_normalize.return_value = self.dummy_image_3d
        mock_get_model.return_value = self.mock_model_instance

        expected_labels = np.array([[[1,1],[1,1]]]) # 3D labels
        self.mock_model_instance.predict_instances.return_value = (expected_labels, {})

        labels = segment_nuclei(self.dummy_image_3d, segmentation_type="3D", model=self.mock_model_instance, n_tiles=(2,4,4))

        self.mock_model_instance.predict_instances.assert_called_once()
        args, kwargs = self.mock_model_instance.predict_instances.call_args
        self.assertEqual(kwargs.get('n_tiles'), (2,4,4))
        self.mock_model_instance.predict_instances_big.assert_not_called()
        self.assertTrue(np.array_equal(labels, expected_labels))

    @patch('Stardist.utils_stardist.normalize')
    @patch('Stardist.utils_stardist.get_stardist_model')
    def test_segment_nuclei_fallback_to_predict_instances_big_2d(self, mock_get_model, mock_normalize):
        mock_normalize.return_value = self.dummy_image_2d
        mock_get_model.return_value = self.mock_model_instance

        # predict_instances fails
        self.mock_model_instance.predict_instances.side_effect = Exception("Memory error sim")
        # predict_instances_big returns different labels
        expected_labels_big = np.array([[2,2],[2,2]])
        self.mock_model_instance.predict_instances_big.return_value = (expected_labels_big, {})

        labels = segment_nuclei(self.dummy_image_2d, segmentation_type="2D", model=self.mock_model_instance, n_tiles=(4,4))

        self.mock_model_instance.predict_instances.assert_called_once()
        self.mock_model_instance.predict_instances_big.assert_called_once()
        # Check n_tiles and axes for predict_instances_big
        args_big, kwargs_big = self.mock_model_instance.predict_instances_big.call_args
        self.assertEqual(kwargs_big.get('n_tiles'), (4,4))
        self.assertEqual(kwargs_big.get('axes'), 'YX')
        self.assertTrue(np.array_equal(labels, expected_labels_big))

    @patch('Stardist.utils_stardist.normalize')
    @patch('Stardist.utils_stardist.get_stardist_model')
    def test_segment_nuclei_fallback_to_predict_instances_big_3d(self, mock_get_model, mock_normalize):
        mock_normalize.return_value = self.dummy_image_3d
        mock_get_model.return_value = self.mock_model_instance

        self.mock_model_instance.predict_instances.side_effect = Exception("Memory error sim")
        expected_labels_big = np.array([[[2,2],[2,2]]]) # 3D labels
        self.mock_model_instance.predict_instances_big.return_value = (expected_labels_big, {})

        labels = segment_nuclei(self.dummy_image_3d, segmentation_type="3D", model=self.mock_model_instance, n_tiles=(2,4,4))

        self.mock_model_instance.predict_instances.assert_called_once()
        self.mock_model_instance.predict_instances_big.assert_called_once()
        args_big, kwargs_big = self.mock_model_instance.predict_instances_big.call_args
        self.assertEqual(kwargs_big.get('n_tiles'), (2,4,4))
        self.assertEqual(kwargs_big.get('axes'), 'ZYX')
        self.assertTrue(np.array_equal(labels, expected_labels_big))

    @patch('Stardist.utils_stardist.normalize')
    @patch('Stardist.utils_stardist.get_stardist_model')
    def test_n_tiles_2d_input_for_2d_segmentation(self, mock_get_model, mock_normalize):
        mock_normalize.return_value = self.dummy_image_2d
        mock_get_model.return_value = self.mock_model_instance

        segment_nuclei(self.dummy_image_2d, segmentation_type="2D", model=self.mock_model_instance, n_tiles=(4,4))
        args, kwargs = self.mock_model_instance.predict_instances.call_args
        self.assertEqual(kwargs.get('n_tiles'), (4,4))

    @patch('Stardist.utils_stardist.normalize')
    @patch('Stardist.utils_stardist.get_stardist_model')
    def test_n_tiles_3d_input_for_2d_segmentation(self, mock_get_model, mock_normalize):
        # When a 3-tuple n_tiles is passed for 2D, it should take the last two.
        mock_normalize.return_value = self.dummy_image_2d
        mock_get_model.return_value = self.mock_model_instance

        segment_nuclei(self.dummy_image_2d, segmentation_type="2D", model=self.mock_model_instance, n_tiles=(2,4,4))
        args, kwargs = self.mock_model_instance.predict_instances.call_args
        self.assertEqual(kwargs.get('n_tiles'), (4,4))

if __name__ == '__main__':
    unittest.main()
