import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from pathlib import Path

# Temporarily adjust sys.path if Stardist is not installed in a way that 'from Stardist...' works directly
# This is often needed if running tests directly from a repo structure.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import the function to be tested
# Note: utils_stardist.py has been modified in previous steps to comment out TF, cle, etc. for importability
from Stardist.utils_stardist import read_image

class TestReadImageBioio(unittest.TestCase):

    def setUp(self):
        # Default image data for mocks
        self.mock_img_data_czi_style = np.random.rand(2, 3, 10, 20, 30).astype(np.float32) # TCZYX example
        self.mock_img_data_squeezed = np.random.rand(3, 20, 30).astype(np.float32) # ZYX example after squeeze
        self.mock_img_data_4d_squeezed = np.random.rand(2, 3, 20, 30).astype(np.float32) # CZYX example after squeeze

    @patch('Stardist.utils_stardist.tifffile.imread')
    @patch('Stardist.utils_stardist.BioImage')
    def test_read_image_bioio_success_no_dimension_order(self, mock_bioimage_constructor, mock_tifffile_read):
        # --- Test successful reading with bioio, no dimension_order ---
        mock_bio_instance = MagicMock()
        mock_bio_instance.to_numpy.return_value = self.mock_img_data_czi_style
        mock_bioimage_constructor.return_value = mock_bio_instance

        img_array, filename = read_image('dummy/path/test_image.czi', slicing_factor_xy=1, slicing_factor_z=1)

        mock_bioimage_constructor.assert_called_once_with(Path('dummy/path/test_image.czi'))
        mock_bio_instance.to_numpy.assert_called_once()
        self.assertEqual(filename, 'test_image')
        # After squeeze, assuming TCZYX -> CZYX if T=1 or ZYX if T=1,C=1
        # The mock_img_data_czi_style is 5D, squeeze will remove leading singletons if any.
        # Let's assume it squeezes to 4D CZYX for this test based on how read_image implements squeeze
        # or 3D ZYX if C is also 1.
        # For this mock, let's say it squeezes to the 4D shape for slicing.
        mock_bio_instance.to_numpy.return_value = np.random.rand(1, 2, 3, 20, 30) # T=1, C=2, Z=3, Y=20, X=30

        img_array_squeezed = np.random.rand(2, 3, 20, 30) # Expected after squeeze
        mock_bio_instance.to_numpy.return_value = img_array_squeezed # Make mock return squeezed shape for simplicity here

        img_array, filename = read_image('dummy/path/test_image.czi', slicing_factor_xy=1, slicing_factor_z=1)
        self.assertTrue(np.array_equal(img_array, img_array_squeezed))


    @patch('Stardist.utils_stardist.tifffile.imread')
    @patch('Stardist.utils_stardist.BioImage')
    def test_read_image_bioio_with_dimension_order(self, mock_bioimage_constructor, mock_tifffile_read):
        # --- Test successful reading with bioio, with dimension_order ---
        mock_bio_instance = MagicMock()
        mock_bio_instance.to_numpy.return_value = self.mock_img_data_4d_squeezed # CZYX
        mock_bioimage_constructor.return_value = mock_bio_instance

        img_array, filename = read_image('dummy/path/test_image.ome.tif',
                                         slicing_factor_xy=1, slicing_factor_z=1,
                                         dimension_order='CZYX')

        mock_bioimage_constructor.assert_called_once_with(Path('dummy/path/test_image.ome.tif'), C=0, Z=1, Y=2, X=3)
        mock_bio_instance.to_numpy.assert_called_once()
        self.assertEqual(filename, 'test_image.ome')
        self.assertTrue(np.array_equal(img_array, self.mock_img_data_4d_squeezed))

    @patch('Stardist.utils_stardist.tifffile.imread')
    @patch('Stardist.utils_stardist.BioImage')
    def test_read_image_bioio_slicing(self, mock_bioimage_constructor, mock_tifffile_read):
        # --- Test slicing ---
        original_shape = (2, 4, 20, 30) # C, Z, Y, X
        mock_data = np.arange(np.prod(original_shape)).reshape(original_shape)

        mock_bio_instance = MagicMock()
        mock_bio_instance.to_numpy.return_value = mock_data
        mock_bioimage_constructor.return_value = mock_bio_instance

        img_array, _ = read_image('dummy/path/test_slicing.tif',
                                  slicing_factor_xy=2, slicing_factor_z=2,
                                  dimension_order='CZYX') # Provide order to ensure slicing logic

        # Expected shape: C=2, Z=4/2=2, Y=20/2=10, X=30/2=15
        self.assertEqual(img_array.shape, (2, 2, 10, 15))
        # Check a few values assuming CZYX order for slicing in read_image for 4D
        # img = img[:, ::slicing_factor_z, ::slicing_factor_xy, ::slicing_factor_xy]
        expected_slice = mock_data[:, ::2, ::2, ::2]
        self.assertTrue(np.array_equal(img_array, expected_slice))

    @patch('Stardist.utils_stardist.tifffile.imread')
    @patch('Stardist.utils_stardist.BioImage', side_effect=Exception("BioImage generic error"))
    def test_read_image_bioio_fails_fallback_to_tifffile_success(self, mock_bioimage_constructor, mock_tifffile_read):
        # --- Test bioio fails, fallback to tifffile successfully for .tif ---
        mock_tifffile_read.return_value = self.mock_img_data_squeezed # ZYX

        img_array, filename = read_image('dummy/path/test_image.tif', slicing_factor_xy=1, slicing_factor_z=1)

        mock_bioimage_constructor.assert_called_once()
        mock_tifffile_read.assert_called_once_with(Path('dummy/path/test_image.tif'))
        self.assertEqual(filename, 'test_image')
        self.assertTrue(np.array_equal(img_array, self.mock_img_data_squeezed))

    @patch('Stardist.utils_stardist.BioImage', side_effect=Exception("BioImage generic error"))
    def test_read_image_bioio_fails_not_tif_raises_error(self, mock_bioimage_constructor):
        # --- Test bioio fails for non-TIFF, should raise original error ---
        with self.assertRaisesRegex(Exception, "BioImage generic error"):
            read_image('dummy/path/test_image.czi', slicing_factor_xy=1, slicing_factor_z=1)
        mock_bioimage_constructor.assert_called_once()


    @patch('Stardist.utils_stardist.tifffile.imread', side_effect=Exception("Tifffile error"))
    @patch('Stardist.utils_stardist.BioImage', side_effect=Exception("BioImage generic error"))
    def test_read_image_bioio_and_tifffile_fail(self, mock_bioimage_constructor, mock_tifffile_read):
        # --- Test both bioio and tifffile fallback fail for .tif ---
        with self.assertRaisesRegex(Exception, "Tifffile error"):
            read_image('dummy/path/test_image.tif', slicing_factor_xy=1, slicing_factor_z=1)
        mock_bioimage_constructor.assert_called_once()
        mock_tifffile_read.assert_called_once()

if __name__ == '__main__':
    unittest.main()
