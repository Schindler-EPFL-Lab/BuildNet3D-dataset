import shutil
import unittest
from pathlib import Path

from tests.tests_building_generator.test_helpers.test_composed_building import (
    MockComposedBuilding,
)


class TestComposedBuilding(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.building = MockComposedBuilding()
        cls.building.add_roof(0, (1.0, 0, 0), overhang=0.25, thickness=1)

        cls.output_path = Path("./tests/building_generator_test_outputs/")
        cls.output_path.mkdir(exist_ok=True)

    def test_save_obj(self) -> None:
        self.building.save_obj(0, str(self.output_path))

        # save_obj should save a .obj and .mtl
        path = Path(self.output_path, "model/0.obj")
        path_mtl = Path(self.output_path, "model/0.mtl")
        self.assertTrue(path.is_file())
        self.assertTrue(path_mtl.is_file())

    def test_save_segmented_mesh(self) -> None:
        self.building.save_segmented_mesh(filename=0, data_folder=str(self.output_path))
        path = Path(self.output_path, "model/0_segmented.ply")
        self.assertTrue(path.is_file())

    def test_save_ply(self) -> None:
        self.building.save_ply(0, str(self.output_path), 0)

        path = Path(self.output_path, "pointCloud/0.ply")
        self.assertTrue(path.is_file())

    def test_save_dae(self) -> None:
        self.building.save_dae(0, str(self.output_path))

        path = Path(self.output_path, "model/0.dae")
        self.assertTrue(path.is_file())

    def test_save_stl(self) -> None:
        self.building.save_stl(0, str(self.output_path))

        path = Path(self.output_path, "pointCloud/0.stl")
        self.assertTrue(path.is_file())

    def test_add_roof(self) -> None:
        self.assertEqual(len(self.building.roofs), 1)

    def test_roof_dimensions(self) -> None:
        dimensions = self.building.roofs[0].mesh.dimensions
        self.assertEqual(dimensions[0], 15.5)
        self.assertEqual(dimensions[1], 5.5)
        self.assertEqual(dimensions[2], 1)

    def test_roof_location(self) -> None:
        location = self.building.roofs[0].mesh.location
        self.assertEqual(location[0], 0)
        self.assertEqual(location[1], 0)
        self.assertEqual(location[2], 7.5)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.output_path)
