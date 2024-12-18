import unittest
from unittest.mock import patch

from buildnet3d_dataset.modules_manager.parametric_window import (
    ParametricWindow,
)


class TestParametericWindow(unittest.TestCase):
    @patch(
        "buildnet3d_dataset.modules_manager."
        + "parametric_window.ParametricWindow.apply"
    )
    def test_window_area(self, mock_apply) -> None:
        window = ParametricWindow(
            id=1,
            segmentation_color=(0.1, 0.1, 0.1),
            material=["glass"],
            path_to_materials="./",
            width=2,
            length=0.01,
            height=5,
            name="window",
        )
        self.assertEqual(window.area, 10)
