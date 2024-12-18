import unittest
from unittest.mock import patch

import bpy
import numpy as np

from buildnet3d_dataset.blender_utils import (
    deselect_all,
    dissolve_faces,
    gancio,
    intersection_check,
    mesh_boolean_union,
    select_all,
)
from buildnet3d_dataset.buildings.volume import Volume

import bmesh  # isort: skip # type: ignore


class TestBlenderUtils(unittest.TestCase):
    def test_intersection_check_touching(self) -> None:
        bpy.ops.mesh.primitive_cube_add(size=1.0)  # type: ignore
        mesh1 = bpy.context.selected_objects[0]  # type: ignore
        mesh1.location = (0, 0, 0)
        mesh1.select_set(False)

        bpy.ops.mesh.primitive_cube_add(size=1.0)  # type: ignore
        mesh2 = bpy.context.selected_objects[0]  # type: ignore
        mesh2.location = (1, 0, 0)
        mesh2.select_set(False)
        self.assertTrue(intersection_check(mesh1, mesh2))

    def test_intersection_check_overlapped(self) -> None:
        bpy.ops.mesh.primitive_cube_add(size=1.0)  # type: ignore
        mesh1 = bpy.context.selected_objects[0]  # type: ignore
        mesh1.location = (3, 2, 0)
        mesh1.select_set(False)

        bpy.ops.mesh.primitive_cube_add(size=1.0)  # type: ignore
        mesh2 = bpy.context.selected_objects[0]  # type: ignore
        mesh2.location = (3, 2.5, 0)
        mesh2.select_set(False)
        self.assertTrue(intersection_check(mesh1, mesh2))

    def test_intersection_check_far_away(self) -> None:
        bpy.ops.mesh.primitive_cube_add(size=1.0)  # type: ignore
        mesh1 = bpy.context.selected_objects[0]  # type: ignore
        mesh1.location = (0, 0, 0)
        mesh1.select_set(False)

        bpy.ops.mesh.primitive_cube_add(size=1.0)  # type: ignore
        mesh2 = bpy.context.selected_objects[0]  # type: ignore
        mesh2.location = (10, 0, 0)
        mesh2.select_set(False)
        self.assertFalse(intersection_check(mesh1, mesh2))

    def test_mesh_boolean_union_touching(self) -> None:
        bpy.ops.mesh.primitive_cube_add(size=1.0)  # type: ignore
        mesh1 = bpy.context.selected_objects[0]  # type: ignore
        mesh1.location = (0, 0, 0)
        mesh1.name = "base_mesh"
        mesh1.select_set(False)

        bpy.ops.mesh.primitive_cube_add(size=1.0)  # type: ignore
        mesh2 = bpy.context.selected_objects[0]  # type: ignore
        mesh2.location = (1, 0, 0)
        mesh2.name = "union_mesh"
        mesh2.select_set(False)

        mesh_boolean_union(mesh1, mesh2)

        for ret, truth, coord in zip(mesh1.dimensions, (2, 1, 1), ["x", "y", "z"]):
            self.assertEqual(ret, truth, f"{coord} dimension incorrect for base mesh")

    def test_mesh_boolean_union_overlapping(self) -> None:
        bpy.ops.mesh.primitive_cube_add(size=1.0)  # type: ignore
        mesh1 = bpy.context.selected_objects[0]  # type: ignore
        mesh1.location = (2, 2, 2)
        mesh1.name = "base_mesh"
        mesh1.select_set(False)

        bpy.ops.mesh.primitive_cube_add(size=1.0)  # type: ignore
        mesh2 = bpy.context.selected_objects[0]  # type: ignore
        mesh2.location = (2, 2.5, 2)
        mesh2.name = "union_mesh"
        mesh2.select_set(False)

        mesh_boolean_union(mesh1, mesh2)

        for ret, truth, coord in zip(mesh1.dimensions, (1, 1.5, 1), ["x", "y", "z"]):
            self.assertEqual(ret, truth, f"{coord} dimension incorrect for base mesh")

    def test_mesh_boolean_union_del_union_mesh(self) -> None:
        bpy.ops.mesh.primitive_cube_add(size=1.0)  # type: ignore
        mesh1 = bpy.context.selected_objects[0]  # type: ignore
        mesh1.location = (2, 2, 2)
        mesh1.name = "base_mesh"
        mesh1.select_set(False)

        bpy.ops.mesh.primitive_cube_add(size=1.0)  # type: ignore
        mesh2 = bpy.context.selected_objects[0]  # type: ignore
        mesh2.location = (2, 2.5, 2)
        mesh2.name = "union_mesh"
        mesh2.select_set(False)

        mesh_boolean_union(mesh1, mesh2)

        self.assertFalse(
            "union_mesh" in [o.name for o in bpy.data.objects]  # type: ignore
        )

    def test_mesh_boolean_union_not_touching(self) -> None:
        bpy.ops.mesh.primitive_cube_add(size=1.0)  # type: ignore
        mesh1 = bpy.context.selected_objects[0]  # type: ignore
        mesh1.location = (0, 0, 0)
        mesh1.name = "base_mesh"
        mesh1.select_set(False)

        bpy.ops.mesh.primitive_cube_add(size=1.0)  # type: ignore
        mesh2 = bpy.context.selected_objects[0]  # type: ignore
        mesh2.location = (2, 2, 2)
        mesh2.name = "union_mesh"
        mesh2.select_set(False)

        mesh_boolean_union(mesh1, mesh2)

        for ret, truth, coord in zip(mesh1.dimensions, (1, 1, 1), ["x", "y", "z"]):
            self.assertEqual(ret, truth, f"{coord} dimension incorrect for base mesh")

        for ret, truth, coord in zip(mesh2.dimensions, (1, 1, 1), ["x", "y", "z"]):
            self.assertEqual(ret, truth, f"{coord} dimension incorrect for union mesh")

    def test_select_all(self) -> None:
        bpy.ops.mesh.primitive_cube_add(size=1.0)  # type: ignore
        mesh1 = bpy.context.selected_objects[0]  # type: ignore
        mesh1.select_set(False)

        bpy.ops.mesh.primitive_cube_add(size=1.0)  # type: ignore
        mesh2 = bpy.context.selected_objects[0]  # type: ignore
        mesh2.select_set(False)

        bpy.ops.mesh.primitive_cube_add(size=1.0)  # type: ignore
        mesh3 = bpy.context.selected_objects[0]  # type: ignore
        mesh3.select_set(True)

        select_all()
        obj_selected: list[bool] = []
        for obj in bpy.data.objects:  # type: ignore
            obj_selected.append(obj.select_get())

        self.assertTrue(np.all(obj_selected))

    def test_deselect_all(self) -> None:
        bpy.ops.mesh.primitive_cube_add(size=1.0)  # type: ignore
        mesh1 = bpy.context.selected_objects[0]  # type: ignore
        mesh1.select_set(True)

        bpy.ops.mesh.primitive_cube_add(size=1.0)  # type: ignore
        mesh2 = bpy.context.selected_objects[0]  # type: ignore
        mesh2.select_set(True)

        bpy.ops.mesh.primitive_cube_add(size=1.0)  # type: ignore
        mesh3 = bpy.context.selected_objects[0]  # type: ignore
        mesh3.select_set(False)

        deselect_all()
        obj_selected: list[bool] = []
        for obj in bpy.data.objects:  # type: ignore
            obj_selected.append(obj.select_get())
        obj_selected_arr = np.array(obj_selected)
        self.assertTrue(np.all(~obj_selected_arr))

    @patch("buildnet3d_dataset.buildings.volume.Volume._nest")
    def test_gancio_yaxis_min_xaxis_min(self, mock_nest) -> None:
        vol1 = Volume(0, "vol1", width=1, length=3, height=3, location=(-2, 3, 0))
        vol2 = Volume(0, "vol2", width=4, length=2, height=2, location=(2, 2, 0))

        gancio(v1=vol1, v2=vol2, axis=1, border1=0, border2=0, overlap=0)

        for ret, truth, coord in zip(vol1.mesh.location, (-2, 3, 1.5), ["x", "y", "z"]):
            self.assertEqual(ret, truth, f"{coord} dimension incorrect for volume 1")

        for ret, truth, coord in zip(
            vol2.mesh.location, (-0.5, 0.5, 1), ["x", "y", "z"]
        ):
            self.assertEqual(ret, truth, f"{coord} dimension incorrect for volume 2")

    @patch("buildnet3d_dataset.buildings.volume.Volume._nest")
    def test_gancio_yaxis_min_xaxis_max(self, mock_nest) -> None:
        vol1 = Volume(0, "vol1", width=1, length=3, height=3, location=(-2, 3, 0))
        vol2 = Volume(0, "vol2", width=4, length=2, height=2, location=(2, 2, 0))

        gancio(v1=vol1, v2=vol2, axis=1, border1=0, border2=1, overlap=0)

        for ret, truth, coord in zip(vol1.mesh.location, (-2, 3, 1.5), ["x", "y", "z"]):
            self.assertEqual(ret, truth, f"{coord} dimension incorrect for volume 1")

        for ret, truth, coord in zip(
            vol2.mesh.location, (-3.5, 0.5, 1), ["x", "y", "z"]
        ):
            self.assertEqual(ret, truth, f"{coord} dimension incorrect for volume 2")

    @patch("buildnet3d_dataset.buildings.volume.Volume._nest")
    def test_gancio_yaxis_max_xaxis_min(self, mock_nest) -> None:
        vol1 = Volume(0, "vol1", width=1, length=3, height=3, location=(-2, 3, 0))
        vol2 = Volume(0, "vol2", width=4, length=2, height=2, location=(2, 2, 0))

        gancio(v1=vol1, v2=vol2, axis=1, border1=1, border2=0, overlap=0)

        for ret, truth, coord in zip(vol1.mesh.location, (-2, 3, 1.5), ["x", "y", "z"]):
            self.assertEqual(ret, truth, f"{coord} dimension incorrect for volume 1")

        for ret, truth, coord in zip(
            vol2.mesh.location, (-0.5, 5.5, 1), ["x", "y", "z"]
        ):
            self.assertEqual(ret, truth, f"{coord} dimension incorrect for volume 2")

    @patch("buildnet3d_dataset.buildings.volume.Volume._nest")
    def test_gancio_yaxis_max_xaxis_max(self, mock_nest) -> None:
        vol1 = Volume(0, "vol1", width=1, length=3, height=3, location=(-2, 3, 0))
        vol2 = Volume(0, "vol2", width=4, length=2, height=2, location=(2, 2, 0))

        gancio(v1=vol1, v2=vol2, axis=1, border1=1, border2=1, overlap=0)

        for ret, truth, coord in zip(vol1.mesh.location, (-2, 3, 1.5), ["x", "y", "z"]):
            self.assertEqual(ret, truth, f"{coord} dimension incorrect for volume 1")

        for ret, truth, coord in zip(
            vol2.mesh.location, (-3.5, 5.5, 1), ["x", "y", "z"]
        ):
            self.assertEqual(ret, truth, f"{coord} dimension incorrect for volume 2")

    @patch("buildnet3d_dataset.buildings.volume.Volume._nest")
    def test_gancio_xaxis_min_yaxis_min(self, mock_nest) -> None:
        vol1 = Volume(0, "vol1", width=1, length=3, height=3, location=(-2, 3, 0))
        vol2 = Volume(0, "vol2", width=4, length=2, height=2, location=(2, 2, 0))

        gancio(v1=vol1, v2=vol2, axis=0, border1=0, border2=0, overlap=0)

        for ret, truth, coord in zip(vol1.mesh.location, (-2, 3, 1.5), ["x", "y", "z"]):
            self.assertEqual(ret, truth, f"{coord} dimension incorrect for volume 1")

        for ret, truth, coord in zip(
            vol2.mesh.location, (-4.5, 2.5, 1), ["x", "y", "z"]
        ):
            self.assertEqual(ret, truth, f"{coord} dimension incorrect for volume 2")

    @patch("buildnet3d_dataset.buildings.volume.Volume._nest")
    def test_gancio_xaxis_min_yaxis_max(self, mock_nest) -> None:
        vol1 = Volume(0, "vol1", width=1, length=3, height=3, location=(-2, 3, 0))
        vol2 = Volume(0, "vol2", width=4, length=2, height=2, location=(2, 2, 0))

        gancio(v1=vol1, v2=vol2, axis=0, border1=0, border2=1, overlap=0)

        for ret, truth, coord in zip(vol1.mesh.location, (-2, 3, 1.5), ["x", "y", "z"]):
            self.assertEqual(ret, truth, f"{coord} dimension incorrect for volume 1")

        for ret, truth, coord in zip(
            vol2.mesh.location, (-4.5, 3.5, 1), ["x", "y", "z"]
        ):
            self.assertEqual(ret, truth, f"{coord} dimension incorrect for volume 2")

    @patch("buildnet3d_dataset.buildings.volume.Volume._nest")
    def test_gancio_xaxis_max_yaxis_min(self, mock_nest) -> None:
        vol1 = Volume(0, "vol1", width=1, length=3, height=3, location=(-2, 3, 0))
        vol2 = Volume(0, "vol2", width=4, length=2, height=2, location=(2, 2, 0))

        gancio(v1=vol1, v2=vol2, axis=0, border1=1, border2=0, overlap=0)

        for ret, truth, coord in zip(vol1.mesh.location, (-2, 3, 1.5), ["x", "y", "z"]):
            self.assertEqual(ret, truth, f"{coord} dimension incorrect for volume 1")

        for ret, truth, coord in zip(
            vol2.mesh.location, (0.5, 2.5, 1), ["x", "y", "z"]
        ):
            self.assertEqual(ret, truth, f"{coord} dimension incorrect for volume 2")

    @patch("buildnet3d_dataset.buildings.volume.Volume._nest")
    def test_gancio_xaxis_max_yaxis_max(self, mock_nest) -> None:
        vol1 = Volume(0, "vol1", width=1, length=3, height=3, location=(-2, 3, 0))
        vol2 = Volume(0, "vol2", width=4, length=2, height=2, location=(2, 2, 0))

        gancio(v1=vol1, v2=vol2, axis=0, border1=1, border2=1, overlap=0)

        for ret, truth, coord in zip(vol1.mesh.location, (-2, 3, 1.5), ["x", "y", "z"]):
            self.assertEqual(ret, truth, f"{coord} dimension incorrect for volume 1")

        for ret, truth, coord in zip(
            vol2.mesh.location, (0.5, 3.5, 1), ["x", "y", "z"]
        ):
            self.assertEqual(ret, truth, f"{coord} dimension incorrect for volume 2")

    def test_dissolve_faces_from_join(self) -> None:
        bpy.ops.mesh.primitive_cube_add(size=1.0)  # type: ignore
        mesh1 = bpy.context.selected_objects[0]  # type: ignore
        mesh1.location = (2, 2, 2)
        mesh1.name = "base_mesh"
        mesh1.select_set(False)

        bpy.ops.mesh.primitive_cube_add(size=1.0)  # type: ignore
        mesh2 = bpy.context.selected_objects[0]  # type: ignore
        mesh2.location = (2, 2.5, 2)
        mesh2.name = "union_mesh"
        mesh2.select_set(False)

        mesh_boolean_union(mesh1, mesh2)
        bm1 = bmesh.new()
        bm1.from_mesh(mesh1.data)
        bm1.faces.ensure_lookup_table()
        old_nb_faces = len(bm1.faces)

        assert old_nb_faces > 6, "Original test case not set up correctly"

        dissolve_faces(mesh1)
        bm2 = bmesh.new()
        bm2.from_mesh(mesh1.data)
        bm2.faces.ensure_lookup_table()
        new_nb_faces = len(bm2.faces)

        self.assertEqual(new_nb_faces, 6)

    def test_dissolve_faces_from_triangulate(self) -> None:
        bpy.ops.mesh.primitive_cube_add(size=1.0)  # type: ignore
        mesh1 = bpy.context.selected_objects[0]  # type: ignore
        mesh1.name = "base_mesh"
        mesh1.select_set(False)
        bm1 = bmesh.new()
        bm1.from_mesh(mesh1.data)

        bmesh.ops.triangulate(bm1, faces=bm1.faces[:])
        bm1.to_mesh(mesh1.data)
        bm1.faces.ensure_lookup_table()
        old_nb_faces = len(bm1.faces)
        bm1.free()

        assert old_nb_faces > 6, "Original test case not set up correctly"

        dissolve_faces(mesh1)
        bm2 = bmesh.new()
        bm2.from_mesh(mesh1.data)
        bm2.faces.ensure_lookup_table()
        new_nb_faces = len(bm2.faces)

        self.assertEqual(new_nb_faces, 6)

    def test_dissolve_faces_null(self) -> None:
        bpy.ops.mesh.primitive_cube_add(size=1.0)  # type: ignore
        mesh1 = bpy.context.selected_objects[0]  # type: ignore
        mesh1.name = "base_mesh"
        mesh1.select_set(False)

        dissolve_faces(mesh1)
        bm2 = bmesh.new()
        bm2.from_mesh(mesh1.data)
        bm2.faces.ensure_lookup_table()
        new_nb_faces = len(bm2.faces)

        self.assertEqual(new_nb_faces, 6)
