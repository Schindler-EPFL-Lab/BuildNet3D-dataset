import unittest
from pathlib import Path
from unittest.mock import patch

from buildnet3d_dataset.blender_scripts.dataset import Dataset


class TestDataset(unittest.TestCase):
    @patch("buildnet3d_dataset.material_factory.MaterialFactory")
    def test_read_segmentation(self, mock_material_factory) -> None:
        truth_return_mapping = {
            "background": {"ID": 0, "RGB": (0.0, 0.0, 0.0)},
            "wall": {"ID": 1, "RGB": (0.0, 1.0, 1.0)},
            "window": {"ID": 2, "RGB": (50 / 255, 10 / 255, 75 / 255)},
            "roof": {"ID": 3, "RGB": (125 / 255, 200 / 255, 20 / 255)},
            "other_class": {"ID": 4, "RGB": (1.0, 0.0, 0.0)},
        }
        mock_segmentation_mapping_path = Path(
            "tests/data/building_generation/mock_segmentation_mapping.json"
        )
        dataset_return_mapping = Dataset._read_segmentation_info(
            mock_segmentation_mapping_path
        )
        self.assertTrue(dataset_return_mapping == truth_return_mapping)
