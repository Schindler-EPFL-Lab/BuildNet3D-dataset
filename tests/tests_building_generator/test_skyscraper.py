import unittest
from unittest.mock import patch

from buildnet3d_dataset.buildings.skyscraper import Skyscraper
from buildnet3d_dataset.modules_manager.parametric_window import (
    ParametricWindow,
)


class TestSkyscraperModules(unittest.TestCase):
    @patch(
        "buildnet3d_dataset.modules_manager."
        + "parametric_window.ParametricWindow.apply"
    )
    def setUp(self, mock_apply) -> None:
        self.building = Skyscraper(
            semantic_id=1,
            min_width=12,
            min_length=12,
            min_height=20,
            max_width=12,
            max_length=12,
            max_height=20,
            segmentation_color=(0.5, 0.5, 0.5),
        )

        self.building.add_window_elements(
            module=ParametricWindow(
                id=2,
                material=["glass"],
                path_to_materials="./",
                segmentation_color=(1.0, 0.0, 0.0),
                width=1.5,
                height=1.5,
            ),
            prob_windows=1,
            x_step=3,
        )

    def test_number_of_windows_Skyscraper(self) -> None:
        self.assertEqual(self.building._windows_applied, 108)

    def test_windows_area_Skyscraper(self) -> None:
        self.assertEqual(self.building._windows_area, 108 * 2.25)

    def test_WWR_Skyscraper(self) -> None:
        self.assertAlmostEqual(self.building.wwr, 0.220, 3)


class TestSkyscraper(unittest.TestCase):
    def test_skyscraper_no_windows(self) -> None:
        min_width = 10
        min_length = 20
        min_height = 30
        max_width = 20
        max_length = 30
        max_height = 40
        building = Skyscraper(
            semantic_id=1,
            min_width=min_width,
            min_length=min_length,
            min_height=min_height,
            max_width=max_width,
            max_length=max_length,
            max_height=max_height,
            segmentation_color=(0.5, 0.5, 0.5),
        )
        self.assertEqual(building._windows_applied, 0)
        self.assertEqual(building._windows_area, 0)
        self.assertEqual(building.wwr, 0)

    @patch("buildnet3d_dataset.buildings.volume.Volume._nest")
    def test_skyscraper_building_dimensions(self, mock_nest) -> None:
        min_width = 10
        min_length = 20
        min_height = 30
        max_width = 20
        max_length = 30
        max_height = 40
        building = Skyscraper(
            semantic_id=1,
            min_width=min_width,
            min_length=min_length,
            min_height=min_height,
            max_width=max_width,
            max_length=max_length,
            max_height=max_height,
            segmentation_color=(0.5, 0.5, 0.5),
        )

        self.assertEqual(len(building.volumes), 1)

        dimensions = building.volumes[0].mesh.dimensions
        self.assertGreaterEqual(dimensions[0], min_width)
        self.assertLessEqual(dimensions[0], max_width)

        self.assertGreaterEqual(dimensions[1], min_length)
        self.assertLessEqual(dimensions[1], max_length)

        self.assertGreaterEqual(dimensions[2], min_height)
        self.assertLessEqual(dimensions[2], max_height)

    @patch("buildnet3d_dataset.buildings.volume.Volume._nest")
    def test_skyscraper_building_dimensions_if_dimensions_are_given(
        self, mock_nest
    ) -> None:
        """
        User should be able to specify the building dimensions by setting
        min_dimension=max_dimension
        """
        min_width = 10
        min_length = 20
        min_height = 30
        building = Skyscraper(
            semantic_id=1,
            min_width=min_width,
            min_length=min_length,
            min_height=min_height,
            max_width=min_width,
            max_length=min_length,
            max_height=min_height,
            segmentation_color=(0.5, 0.5, 0.5),
        )

        dimensions = building.volumes[0].mesh.dimensions
        self.assertEqual(dimensions[0], min_width)
        self.assertEqual(dimensions[1], min_length)
        self.assertEqual(dimensions[2], min_height)

    def test_skyscraper_clone_no_change(self) -> None:
        min_width = 10
        min_length = 20
        min_height = 30
        max_width = 20
        max_length = 30
        max_height = 40
        segmentation_color = (0.5, 0.5, 0.5)
        semantic_id = 1
        building = Skyscraper(
            semantic_id=semantic_id,
            min_width=min_width,
            min_length=min_length,
            min_height=min_height,
            max_width=max_width,
            max_length=max_length,
            max_height=max_height,
            segmentation_color=segmentation_color,
            template=True,
        )
        new_building = building.clone()
        self.assertEqual(new_building.min_width, min_width)  # type: ignore
        self.assertEqual(new_building.min_length, min_length)  # type: ignore
        self.assertEqual(new_building.min_height, min_height)  # type: ignore
        self.assertEqual(new_building.max_width, max_width)  # type: ignore
        self.assertEqual(new_building.max_length, max_length)  # type: ignore
        self.assertEqual(new_building.max_height, max_height)  # type: ignore
        self.assertTrue(new_building.segmentation_color == segmentation_color)
        self.assertTrue(new_building.semantic_id == semantic_id)

    def test_skyscraper_clone_change(self) -> None:
        min_width = 10
        min_length = 20
        min_height = 30
        max_width = 20
        max_length = 30
        max_height = 40
        segmentation_color = (0.5, 0.5, 0.5)
        semantic_id = 1
        building = Skyscraper(
            semantic_id=semantic_id,
            min_width=min_width,
            min_length=min_length,
            min_height=min_height,
            max_width=max_width,
            max_length=max_length,
            max_height=max_height,
            segmentation_color=segmentation_color,
            template=True,
        )
        new_min_width = 5
        new_max_length = 100
        new_building = building.clone(min_width=5, max_length=100)
        self.assertEqual(new_building.min_width, new_min_width)  # type: ignore
        self.assertEqual(new_building.min_length, min_length)  # type: ignore
        self.assertEqual(new_building.min_height, min_height)  # type: ignore
        self.assertEqual(new_building.max_width, max_width)  # type: ignore
        self.assertEqual(new_building.max_length, new_max_length)  # type: ignore
        self.assertEqual(new_building.max_height, max_height)  # type: ignore
        self.assertTrue(new_building.segmentation_color == segmentation_color)
        self.assertTrue(new_building.semantic_id == semantic_id)
