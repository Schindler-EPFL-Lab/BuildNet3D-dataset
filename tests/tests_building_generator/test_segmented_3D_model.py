import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import networkx as nx
import numpy as np
import pandas as pd

from buildnet3d_dataset.mesh_tools.segmented_3D_model import Segmented3DModel


class TestSegmented3DModel(unittest.TestCase):
    @patch("pyntcloud.PyntCloud.from_file")
    @patch(
        "buildnet3d_dataset.mesh_tools.segmented_3D_model."
        + "Segmented3DModel._get_segmented_mesh_info"
    )
    @patch(
        "buildnet3d_dataset.mesh_tools.segmented_3D_model."
        + "Segmented3DModel._add_triangle_face_normals"
    )
    @patch(
        "buildnet3d_dataset.mesh_tools.segmented_3D_model."
        + "Segmented3DModel._get_connected_meshes_list"
    )
    def test_fix_vertex_colors(
        self,
        mock_get_connected_meshes_list,
        mock_add_triangle_face_normals,
        mock_get_segmented_mesh_info,
        mock_from_file,
    ) -> None:
        test_colours = pd.read_csv("tests/data/building_generation/vertex_test.csv")
        truth_colours = pd.read_csv("tests/data/building_generation/vertex_truth.csv")

        mock_from_file_return = Mock()
        mock_from_file_return.points = test_colours
        mock_from_file_return.mesh = MagicMock()
        mock_from_file.return_value = mock_from_file_return

        model_path = Path("mock/file/path.ply")
        segmentation_mapping_path = Path(
            "tests/data/building_generation/mock_segmentation_mapping.json"
        )

        segpntcloud = Segmented3DModel(
            model_path=model_path, segmentation_mapping_path=segmentation_mapping_path
        )
        self.assertTrue(
            np.all(
                segpntcloud._points[["red", "green", "blue"]]
                == truth_colours[["red", "green", "blue"]]
            )
        )

    @patch("pyntcloud.PyntCloud.from_file")
    @patch(
        "buildnet3d_dataset.mesh_tools.segmented_3D_model."
        + "Segmented3DModel._get_segmented_mesh_info"
    )
    @patch(
        "buildnet3d_dataset.mesh_tools.segmented_3D_model."
        + "Segmented3DModel._add_triangle_face_normals"
    )
    @patch(
        "buildnet3d_dataset.mesh_tools.segmented_3D_model."
        + "Segmented3DModel._get_connected_meshes_list"
    )
    def test_assign_classes(
        self,
        mock_get_connected_meshes_list,
        mock_add_triangle_face_normals,
        mock_get_segmented_mesh_info,
        mock_from_file,
    ) -> None:
        test_colours = pd.read_csv("tests/data/building_generation/vertex_test.csv")
        truth_colours = pd.read_csv("tests/data/building_generation/vertex_truth.csv")

        mock_from_file_return = Mock()
        mock_from_file_return.points = test_colours
        mock_from_file_return.mesh = MagicMock()
        mock_from_file.return_value = mock_from_file_return

        model_path = Path("tests/data/building_generation/mock_model_cubes.ply")
        segmentation_mapping_path = Path(
            "tests/data/building_generation/mock_segmentation_mapping.json"
        )

        segpntcloud = Segmented3DModel(
            model_path=model_path, segmentation_mapping_path=segmentation_mapping_path
        )
        self.assertTrue(
            np.all(segpntcloud._points.seg_class == truth_colours.seg_class)
        )

    @patch(
        "buildnet3d_dataset.mesh_tools.segmented_3D_model."
        + "Segmented3DModel._get_building_walls"
    )
    @patch(
        "buildnet3d_dataset.mesh_tools.segmented_3D_model."
        + "Segmented3DModel._set_wwr_per_wall"
    )
    @patch(
        "buildnet3d_dataset.mesh_tools.segmented_3D_model."
        + "Segmented3DModel._add_triangle_face_normals"
    )
    def test_get_segmented_mesh_info(
        self,
        mock_add_triangle_face_normals,
        mock_set_wwr_per_wall,
        mock_get_building_walls,
    ) -> None:
        model_path = Path("tests/data/building_generation/mock_model_cubes.ply")
        segmentation_mapping_path = Path(
            "tests/data/building_generation/mock_segmentation_mapping.json"
        )
        segmodel_cubes = Segmented3DModel(
            model_path=model_path, segmentation_mapping_path=segmentation_mapping_path
        )
        truth_mesh_info = pd.read_csv(
            "tests/data/building_generation/mock_model_cubes_mesh_info_truth.csv",
            dtype={
                "id_v1": "uint32",
                "id_v2": "uint32",
                "id_v3": "uint32",
                "x_v1": "float32",
                "y_v1": "float32",
                "z_v1": "float32",
                "nx_v1": "float32",
                "ny_v1": "float32",
                "nz_v1": "float32",
                "red_v1": "uint8",
                "green_v1": "uint8",
                "blue_v1": "uint8",
                "x_v2": "float32",
                "y_v2": "float32",
                "z_v2": "float32",
                "nx_v2": "float32",
                "ny_v2": "float32",
                "nz_v2": "float32",
                "red_v2": "uint8",
                "green_v2": "uint8",
                "blue_v2": "uint8",
                "x_v3": "float32",
                "y_v3": "float32",
                "z_v3": "float32",
                "nx_v3": "float32",
                "ny_v3": "float32",
                "nz_v3": "float32",
                "red_v3": "uint8",
                "green_v3": "uint8",
                "blue_v3": "uint8",
                "seg_class": "object",
            },
        )

        mesh_info = segmodel_cubes._get_segmented_mesh_info()

        self.assertTrue((set(mesh_info.columns) == set(truth_mesh_info.columns)))
        for col in mesh_info.columns:
            self.assertTrue(
                mesh_info[col].equals(truth_mesh_info[col]),
                f"Column '{col}' does not match truth data",
            )

    def test_get_triangle_face_normals(self) -> None:
        model_path = Path("tests/data/building_generation/mock_model_cubes.ply")
        segmentation_mapping_path = Path(
            "tests/data/building_generation/mock_segmentation_mapping.json"
        )
        segmodel_cubes = Segmented3DModel(
            model_path=model_path, segmentation_mapping_path=segmentation_mapping_path
        )
        truth_mesh_info = pd.read_csv(
            "tests/data/building_generation/"
            + "mock_model_cubes_mesh_normal_info_truth.csv",
            dtype={
                "face_normal_x": "float32",
                "face_normal_y": "float32",
                "face_normal_z": "float32",
                "face_normal_magnitude": "float32",
            },
        )

        self.assertTrue(
            set(truth_mesh_info.columns).issubset(set(segmodel_cubes._mesh_df.columns))
        )
        for col in truth_mesh_info.columns:
            self.assertTrue(
                segmodel_cubes._mesh_df[col].equals(truth_mesh_info[col]),
                f"Column '{col}' does not match truth data",
            )

    @patch(
        "buildnet3d_dataset.mesh_tools.segmented_3D_model" + ".triangle_face_normals"
    )
    def test_get_triangle_face_normals_inputs(self, mock_triangle_face_normals) -> None:
        mock_triangle_face_normals.return_value = np.arange(96 * 3).reshape(96, 3)
        model_path = Path("tests/data/building_generation/mock_model_cubes.ply")
        segmentation_mapping_path = Path(
            "tests/data/building_generation/mock_segmentation_mapping.json"
        )
        segmodel_cubes = Segmented3DModel(
            model_path=model_path, segmentation_mapping_path=segmentation_mapping_path
        )

        CB = (
            segmodel_cubes._mesh_df[["x_v2", "y_v2", "z_v2"]].to_numpy()
            - segmodel_cubes._mesh_df[["x_v3", "y_v3", "z_v3"]].to_numpy()
        )
        _ = segmodel_cubes._add_triangle_face_normals()
        _, kwargs = mock_triangle_face_normals.call_args
        AB = kwargs["AB"]
        AC = kwargs["AC"]
        self.assertTrue(np.all(AB == (AC + CB)))

    def test_get_connected_meshes_list_number_of_components(self):
        model_path = Path("tests/data/building_generation/mock_model_house.ply")
        segmentation_mapping_path = Path(
            "tests/data/building_generation/mock_segmentation_mapping.json"
        )
        segmodel_house = Segmented3DModel(
            model_path=model_path, segmentation_mapping_path=segmentation_mapping_path
        )
        wall_df = segmodel_house._mesh_df[segmodel_house._mesh_df.seg_class == "wall"]
        window_df = segmodel_house._mesh_df[
            segmodel_house._mesh_df.seg_class == "window"
        ]
        roof_df = segmodel_house._mesh_df[segmodel_house._mesh_df.seg_class == "roof"]

        walls = segmodel_house._get_connected_meshes_list(wall_df)
        windows = segmodel_house._get_connected_meshes_list(window_df)
        roofs = segmodel_house._get_connected_meshes_list(roof_df)

        self.assertEqual(len(walls), 1)
        self.assertEqual(len(windows), 4)
        self.assertEqual(len(roofs), 1)

    def test_get_all_wall_area(self):
        model_path = Path("tests/data/building_generation/mock_model_house.ply")
        segmentation_mapping_path = Path(
            "tests/data/building_generation/mock_segmentation_mapping.json"
        )
        segmodel_house = Segmented3DModel(
            model_path=model_path, segmentation_mapping_path=segmentation_mapping_path
        )

        self.assertEqual(segmodel_house._get_all_wall_area(), 240.0)

    def test_get_all_window_area(self):
        model_path = Path("tests/data/building_generation/mock_model_house.ply")
        segmentation_mapping_path = Path(
            "tests/data/building_generation/mock_segmentation_mapping.json"
        )
        segmodel_house = Segmented3DModel(
            model_path=model_path, segmentation_mapping_path=segmentation_mapping_path
        )

        self.assertEqual(segmodel_house._get_all_window_area(), 88.0)

    def test_get_window_area_from_component_graph(self):
        model_path = Path("tests/data/building_generation/mock_model_house.ply")
        segmentation_mapping_path = Path(
            "tests/data/building_generation/mock_segmentation_mapping.json"
        )
        segmodel_house = Segmented3DModel(
            model_path=model_path, segmentation_mapping_path=segmentation_mapping_path
        )

        # isolate one window and make a graph, long window of size 0.2 x 16 x 2
        window_df = segmodel_house._mesh_df.loc[
            (segmodel_house._mesh_df.seg_class == "window")
            & (segmodel_house._mesh_df.x_v1 >= 5)
            & (segmodel_house._mesh_df.x_v2 >= 5)
            & (segmodel_house._mesh_df.x_v3 >= 5),
            ["id_v1", "id_v2", "id_v3"],
        ]
        window_graph_df = pd.concat(
            [
                window_df.loc[
                    :,
                    ["id_v1", "id_v2"],
                ].rename(columns={"id_v1": "source", "id_v2": "target"}),
                window_df.loc[
                    :,
                    ["id_v2", "id_v3"],
                ].rename(columns={"id_v2": "source", "id_v3": "target"}),
                window_df.loc[
                    :,
                    ["id_v3", "id_v1"],
                ].rename(columns={"id_v3": "source", "id_v1": "target"}),
            ],
            axis=0,
        ).reset_index(drop=True)
        G = nx.from_pandas_edgelist(
            window_graph_df,
            source="source",
            target="target",
        )

        self.assertEqual(segmodel_house._get_window_area_from_component_graph(G), 32)

    def test_get_all_roof_area(self):
        model_path = Path("tests/data/building_generation/mock_model_house.ply")
        segmentation_mapping_path = Path(
            "tests/data/building_generation/mock_segmentation_mapping.json"
        )
        segmodel_house = Segmented3DModel(
            model_path=model_path, segmentation_mapping_path=segmentation_mapping_path
        )

        self.assertEqual(segmodel_house._get_all_roof_area(), 332.0)

    def test_overall_wwr(self):
        model_path = Path("tests/data/building_generation/mock_model_house.ply")
        segmentation_mapping_path = Path(
            "tests/data/building_generation/mock_segmentation_mapping.json"
        )
        segmodel_house = Segmented3DModel(
            model_path=model_path, segmentation_mapping_path=segmentation_mapping_path
        )

        self.assertAlmostEqual(segmodel_house.wwr, 88.0 / 240.0, places=4)

    def test_get_building_walls_cube_house(self):
        """
        The test will test the attributes of the walls found by the function
        Segmented3DModel._get_building_walls
        """
        model_path = Path("tests/data/building_generation/mock_model_house.ply")
        segmentation_mapping_path = Path(
            "tests/data/building_generation/mock_segmentation_mapping.json"
        )
        segmodel_house = Segmented3DModel(
            model_path=model_path, segmentation_mapping_path=segmentation_mapping_path
        )
        walls = segmodel_house._get_building_walls()
        self.assertEqual(len(walls), 4)

        found_x_min_wall = False
        found_x_max_wall = False
        found_y_min_wall = False
        found_y_max_wall = False

        for wall in walls:
            if (
                wall.center[0] == -5.0
                and wall.center[1] == 0.0
                and wall.center[2] == 2.0
            ):
                found_x_min_wall = True
                self.assertEqual(wall.area, 80.0)
                self.assertTrue(np.all(np.array([-1, 0, 0] == wall.direction)))
                self.assertTrue(
                    np.all(
                        np.array([[-5.0, -5.0], [-10.0, 10.0], [0.0, 4.0]])
                        == wall.domain
                    )
                )
                continue
            if (
                wall.center[0] == 5.0
                and wall.center[1] == 0.0
                and wall.center[2] == 2.0
            ):
                found_x_max_wall = True
                self.assertEqual(wall.area, 80.0)
                self.assertTrue(np.all(np.array([1, 0, 0] == wall.direction)))
                self.assertTrue(
                    np.all(
                        np.array([[5.0, 5.0], [-10.0, 10.0], [0.0, 4.0]]) == wall.domain
                    )
                )
                continue
            if (
                wall.center[0] == 0.0
                and wall.center[1] == -10.0
                and wall.center[2] == 2.0
            ):
                found_y_min_wall = True
                self.assertEqual(wall.area, 40.0)
                self.assertTrue(np.all(np.array([0, -1, 0] == wall.direction)))
                self.assertTrue(
                    np.all(
                        np.array([[-5.0, 5.0], [-10.0, -10.0], [0.0, 4.0]])
                        == wall.domain
                    )
                )
                continue
            if (
                wall.center[0] == 0.0
                and wall.center[1] == 10.0
                and wall.center[2] == 2.0
            ):
                found_y_max_wall = True
                self.assertEqual(wall.area, 40.0)
                self.assertTrue(np.all(np.array([0, 1, 0] == wall.direction)))
                self.assertTrue(
                    np.all(
                        np.array([[-5.0, 5.0], [10.0, 10.0], [0.0, 4.0]]) == wall.domain
                    )
                )
                continue
        self.assertTrue(found_x_min_wall, "Did not find x min wall")
        self.assertTrue(found_x_max_wall, "Did not find x max wall")
        self.assertTrue(found_y_min_wall, "Did not find y min wall")
        self.assertTrue(found_y_max_wall, "Did not find y max wall")
