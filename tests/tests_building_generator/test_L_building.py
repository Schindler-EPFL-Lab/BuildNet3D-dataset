import unittest
from unittest.mock import patch

import bpy
import numpy as np

from buildnet3d_dataset.blender_utils import dissolve_faces
from buildnet3d_dataset.buildings.L_building import LBuilding
from buildnet3d_dataset.modules_manager.parametric_window import (
    ParametricWindow,
)

import bmesh  # isort: skip # type: ignore


class TestLBuildingModules(unittest.TestCase):
    @patch(
        "buildnet3d_dataset.modules_manager."
        + "parametric_window.ParametricWindow.apply"
    )
    def setUp(self, mock_apply) -> None:
        self.building = LBuilding(
            semantic_id=1,
            min_width=15,
            min_length=20,
            min_height=7,
            min_cutout_width_percent=0.4,
            min_cutout_length_percent=0.4,
            max_width=15,
            max_length=20,
            max_height=7,
            max_cutout_width_percent=0.4,
            max_cutout_length_percent=0.4,
            segmentation_color=(0.0, 1.0, 0.0),
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
            x_step=2,
        )

    def test_number_of_windows_LBuilding(self) -> None:
        """
        The number of windows for an L building is predictable, but it is also 'wrong'
        as it currently does not produce results for windowing the faces of an L-
        Building, but the faces of two volumes that are then placed together with an
        overlap and the windows that overlap a volume are removed. This needs to be
        fixed in the future.
        """
        self.assertEqual(self.building._windows_applied, 50)

    def test_windows_area_L_Building(self) -> None:
        self.assertEqual(self.building._windows_area, 50 * 2.25)

    def test_WWR_L_building(self) -> None:
        self.assertAlmostEqual(self.building.wwr, 0.152, 3)


class TestLBuilding(unittest.TestCase):
    @patch("buildnet3d_dataset.buildings.volume.Volume._nest")
    def test_L_building_no_windows(self, mock_nest) -> None:
        """
        This specific setting was found to not allow for connection, due to some
        floating point precision in blender.
        """
        min_width = 18
        min_length = 18
        min_height = 18
        min_cutout_width_perc = 0.6445235922439247
        min_cutout_length_perc = 0.10
        building = LBuilding(
            semantic_id=0,
            min_width=min_width,
            min_length=min_length,
            min_height=min_height,
            min_cutout_width_percent=min_cutout_width_perc,
            min_cutout_length_percent=min_cutout_length_perc,
            max_width=min_width,
            max_length=min_length,
            max_height=min_height,
            max_cutout_width_percent=min_cutout_width_perc,
            max_cutout_length_percent=min_cutout_length_perc,
            segmentation_color=(0.9, 0.9, 0.9),
        )
        self.assertEqual(building._windows_applied, 0)
        self.assertEqual(building._windows_area, 0)
        self.assertEqual(building.wwr, 0)

    @patch("buildnet3d_dataset.buildings.volume.Volume._nest")
    def test_L_building_volume_joined(self, mock_nest) -> None:
        """
        This specific setting was found to not allow for connection, due to some
        floating point precision in blender.
        """
        min_width = 18
        min_length = 18
        min_height = 18
        min_cutout_width_perc = 0.6445235922439247
        min_cutout_length_perc = 0.10
        building = LBuilding(
            semantic_id=0,
            min_width=min_width,
            min_length=min_length,
            min_height=min_height,
            min_cutout_width_percent=min_cutout_width_perc,
            min_cutout_length_percent=min_cutout_length_perc,
            max_width=min_width,
            max_length=min_length,
            max_height=min_height,
            max_cutout_width_percent=min_cutout_width_perc,
            max_cutout_length_percent=min_cutout_length_perc,
            segmentation_color=(0.9, 0.9, 0.9),
        )
        building.remove_overlap_faces()
        dissolve_faces(bpy.data.objects[building.volumes[0].name])

        bm = bmesh.new()
        vol = building.volumes[0]
        bm.from_mesh(vol.mesh.data)
        bm.faces.ensure_lookup_table()
        self.assertEqual(len(bm.faces), 8)

    @patch("buildnet3d_dataset.buildings.volume.Volume._nest")
    def test_L_building_dimensions_if_dimensions_are_given(self, mock_nest) -> None:
        min_width = 12
        min_length = 20
        min_height = 5
        min_cutout_width = 3
        min_cutout_length = 10

        building = LBuilding(
            semantic_id=1,
            min_width=12,
            min_length=20,
            min_height=5,
            min_cutout_width_percent=0.25,
            min_cutout_length_percent=0.5,
            max_width=12,
            max_length=20,
            max_height=5,
            max_cutout_width_percent=0.25,
            max_cutout_length_percent=0.5,
            segmentation_color=(0.5, 0.5, 0.5),
        )

        volume_1_bbox = bpy.data.objects[building.volumes[0].mesh.name].bound_box
        volume_1_mat = bpy.data.objects[building.volumes[0].mesh.name].matrix_basis
        volume_2_bbox = bpy.data.objects[building.volumes[1].mesh.name].bound_box
        volume_2_mat = bpy.data.objects[building.volumes[1].mesh.name].matrix_basis

        volume_1_bbox_arr = np.zeros((8, 3))
        volume_1_mat_arr = np.zeros((3, 4))
        volume_2_bbox_arr = np.zeros((8, 3))
        volume_2_mat_arr = np.zeros((3, 4))

        for i, row_v1, row_v2 in zip(range(3), volume_1_mat, volume_2_mat):
            for j, val_v1, val_v2 in zip(range(4), row_v1, row_v2):
                volume_1_mat_arr[i, j] = val_v1
                volume_2_mat_arr[i, j] = val_v2

        for i, vertex_v1, vertex_v2 in zip(range(8), volume_1_bbox, volume_2_bbox):
            for j, coord_v1, coord_v2 in zip(range(3), vertex_v1, vertex_v2):
                volume_1_bbox_arr[i, j] = coord_v1
                volume_2_bbox_arr[i, j] = coord_v2
            volume_1_bbox_arr[i, :] = (
                volume_1_mat_arr[:, :3] @ volume_1_bbox_arr[i, :]
            ) + volume_1_mat_arr[:, 3]
            volume_2_bbox_arr[i, :] = (
                volume_2_mat_arr[:, :3] @ volume_2_bbox_arr[i, :]
            ) + volume_2_mat_arr[:, 3]

        all_verticies = np.vstack([volume_1_bbox_arr, volume_2_bbox_arr])

        ret_width = np.max(all_verticies[:, 0]) - np.min(all_verticies[:, 0])
        ret_length = np.max(all_verticies[:, 1]) - np.min(all_verticies[:, 1])
        ret_cutout_width = np.abs(
            np.min(volume_1_bbox_arr[:, 0] - np.min(volume_2_bbox_arr[:, 0]))
        )
        ret_cutout_length = np.abs(
            np.min(volume_1_bbox_arr[:, 1] - np.min(volume_2_bbox_arr[:, 1]))
        )
        height_v1 = np.max(volume_1_bbox_arr[:, 2]) - np.min(volume_1_bbox_arr[:, 2])
        height_v2 = np.max(volume_2_bbox_arr[:, 2]) - np.min(volume_2_bbox_arr[:, 2])

        self.assertEqual(ret_width, min_width)
        self.assertEqual(ret_length, min_length)
        self.assertEqual(ret_cutout_width, min_cutout_width)
        self.assertEqual(ret_cutout_length, min_cutout_length)
        self.assertEqual(height_v1, min_height)
        self.assertEqual(height_v2, min_height)

    def test_L_building_clone_no_change(self):
        min_width = 18
        min_length = 10
        min_height = 11
        min_cutout_width_perc = 0.6445235922439247
        min_cutout_length_perc = 0.10
        max_width = 20
        max_length = 12
        max_height = 19
        max_cutout_width_perc = 0.9
        max_cutout_length_perc = 0.20
        building = LBuilding(
            semantic_id=0,
            min_width=min_width,
            min_length=min_length,
            min_height=min_height,
            min_cutout_width_percent=min_cutout_width_perc,
            min_cutout_length_percent=min_cutout_length_perc,
            max_width=max_width,
            max_length=max_length,
            max_height=max_height,
            max_cutout_width_percent=max_cutout_width_perc,
            max_cutout_length_percent=max_cutout_length_perc,
            segmentation_color=(0.9, 0.9, 0.9),
        )
        cloned_buliding = building.clone()
        assert isinstance(cloned_buliding, LBuilding)
        self.assertEqual(cloned_buliding._min_width, building._min_width)
        self.assertEqual(cloned_buliding._min_length, building._min_length)
        self.assertEqual(cloned_buliding._min_height, building._min_height)
        self.assertEqual(
            cloned_buliding._min_cutout_width_percent,
            building._min_cutout_width_percent,
        )
        self.assertEqual(
            cloned_buliding._min_cutout_length_percent,
            building._min_cutout_length_percent,
        )
        self.assertEqual(cloned_buliding._max_width, building._max_width)
        self.assertEqual(cloned_buliding._max_length, building._max_length)
        self.assertEqual(cloned_buliding._max_height, building._max_height)
        self.assertEqual(
            cloned_buliding._max_cutout_width_percent,
            building._max_cutout_width_percent,
        )
        self.assertEqual(
            cloned_buliding._max_cutout_length_percent,
            building._max_cutout_length_percent,
        )
        self.assertEqual(
            cloned_buliding.segmentation_color, building.segmentation_color
        )
        self.assertEqual(cloned_buliding.semantic_id, building.semantic_id)

    def test_L_building_clone_changed(self):
        min_width = 18
        min_length = 18
        min_height = 18
        min_cutout_width_perc = 0.6445235922439247
        min_cutout_length_perc = 0.10
        building = LBuilding(
            semantic_id=0,
            min_width=min_width,
            min_length=min_length,
            min_height=min_height,
            min_cutout_width_percent=min_cutout_width_perc,
            min_cutout_length_percent=min_cutout_length_perc,
            max_width=min_width,
            max_length=min_length,
            max_height=min_height,
            max_cutout_width_percent=min_cutout_width_perc,
            max_cutout_length_percent=min_cutout_length_perc,
            segmentation_color=(0.9, 0.9, 0.9),
        )
        new_min_width = 10
        new_max_width = 30
        new_max_cutout_length_percent = 0.5
        cloned_buliding = building.clone(
            min_width=new_min_width,
            max_width=new_max_width,
            max_cutout_length_percent=new_max_cutout_length_percent,
        )
        assert isinstance(cloned_buliding, LBuilding)
        self.assertEqual(cloned_buliding._min_width, new_min_width)
        self.assertEqual(cloned_buliding._min_length, building._min_length)
        self.assertEqual(cloned_buliding._min_height, building._min_height)
        self.assertEqual(
            cloned_buliding._min_cutout_width_percent,
            building._min_cutout_width_percent,
        )
        self.assertEqual(
            cloned_buliding._min_cutout_length_percent,
            building._min_cutout_length_percent,
        )
        self.assertEqual(cloned_buliding._max_width, new_max_width)
        self.assertEqual(cloned_buliding._max_length, building._max_length)
        self.assertEqual(cloned_buliding._max_height, building._max_height)
        self.assertEqual(
            cloned_buliding._max_cutout_width_percent,
            building._max_cutout_width_percent,
        )
        self.assertEqual(
            cloned_buliding._max_cutout_length_percent,
            new_max_cutout_length_percent,
        )
        self.assertEqual(
            cloned_buliding.segmentation_color, building.segmentation_color
        )
        self.assertEqual(cloned_buliding.semantic_id, building.semantic_id)
