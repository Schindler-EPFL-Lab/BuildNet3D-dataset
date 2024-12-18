import json
from functools import cached_property
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from pyntcloud import PyntCloud

from buildnet3d_dataset.mesh_tools.mesh_utils import (
    find_if_line_and_plane_intersect,
    triangle_face_normals,
)
from buildnet3d_dataset.post_process_segmentation import color_quantization


class BuildingWall:
    """
    Class to contain information related to each building wall.
    """

    wwr: float
    domain: np.ndarray
    center: np.ndarray
    direction: np.ndarray
    area: float

    def __init__(self, wall_mesh_df: DataFrame) -> None:
        """
        `wall_mesh_df` should be a copy of a slice of a `Segmented3DModel._mesh_df`
        which contains meshes only for a single buliding wall.
        """
        self._mesh_df = wall_mesh_df

        if np.any(
            self._mesh_df[["face_normal_x", "face_normal_y", "face_normal_z"]]
            .nunique()
            .to_numpy()
            != 1
        ):
            raise RuntimeError(
                "`mesh_df` passe to BuildingWall.__inti__ did "
                + "not contain meshes facing only one direction"
            )

        gathered_vertices = pd.concat(
            [
                wall_mesh_df[["id_v1", "x_v1", "y_v1", "z_v1"]].rename(
                    columns={"id_v1": "id", "x_v1": "x", "y_v1": "y", "z_v1": "z"}
                ),
                wall_mesh_df[["id_v2", "x_v2", "y_v2", "z_v2"]].rename(
                    columns={"id_v2": "id", "x_v2": "x", "y_v2": "y", "z_v2": "z"}
                ),
                wall_mesh_df[["id_v3", "x_v3", "y_v3", "z_v3"]].rename(
                    columns={"id_v3": "id", "x_v3": "x", "y_v3": "y", "z_v3": "z"}
                ),
            ],
            axis=0,
        ).drop_duplicates()

        self._points = gathered_vertices.set_index("id")

        self.domain = np.stack(
            [
                np.min(self._points[["x", "y", "z"]].to_numpy(), axis=0),
                np.max(self._points[["x", "y", "z"]].to_numpy(), axis=0),
            ],
            axis=1,
        )
        self.center = np.mean(self.domain, axis=1)
        self.direction = self._mesh_df.loc[
            self._mesh_df.index[0], ["face_normal_x", "face_normal_y", "face_normal_z"]
        ].to_numpy(dtype=np.float32)

        self.area = self._mesh_df.face_normal_magnitude.sum() / 2

    def set_wwr(self, windows_area: float) -> None:
        """
        Calculates the wwr of the wall. `windows_area` is the total area of the windows
        on the wall.
        """
        self.wwr = windows_area / self.area


class Segmented3DModel:
    """
    Class to facilitate information handling of a point cloud mesh model with
    semantic information. Only works for buildings which have their faces perfectly
    along the world axes.
    """

    def __init__(self, model_path: Path, segmentation_mapping_path: Path) -> None:
        """
        `model_path` is where the `.ply` model information is stored.
        `segmentation_mapping_path` is where the semantic segementation mapping json
        file is stored. semantic segmentation information should be stored as
        `{"class_1": {"ID":int, "RGB":(int, int, int)}`.
        """
        with open(segmentation_mapping_path, "r") as f:
            seg_map = json.load(f)
        seg_map.pop("background", None)
        self._segmentation_map = seg_map
        self._pointcloud = PyntCloud.from_file(str(model_path))
        self._points = self._pointcloud.points
        self._meshes = self._pointcloud.mesh
        if self._meshes is None:
            raise RuntimeError(f"Meshs were not exported in .ply file at {model_path}")
        self._meshes.rename(
            columns={"v1": "id_v1", "v2": "id_v2", "v3": "id_v3"}, inplace=True
        )
        self._fix_vertex_colors()
        self._assign_classes()
        self._mesh_df = self._get_segmented_mesh_info()
        self._add_triangle_face_normals()
        windows_df = self._mesh_df[self._mesh_df.seg_class == "window"]
        self._window_graphs = self._get_connected_meshes_list(windows_df)
        self._building_walls = self._get_building_walls()
        self._set_wwr_per_wall()

    def _fix_vertex_colors(self) -> None:
        """
        Goes through `self._points` and changes colors so that they are all one color
        from `self._segmentation_map`.
        """
        rgb_arr = self._points[["red", "green", "blue"]].to_numpy()
        rgb_fixed = color_quantization(
            rgb_arr, [val["RGB"] for val in self._segmentation_map.values()]
        )
        self._points.red = rgb_fixed[:, 0]
        self._points.green = rgb_fixed[:, 1]
        self._points.blue = rgb_fixed[:, 2]

    def _assign_classes(self) -> None:
        """
        Assigns class labels to points in `self._points` from `self._segmentation_map`
        according to the colors provided in `self._segmentation_map`. Colors without an
        assigned class are given `NaN` as value.
        """
        self._points["seg_class"] = ""
        for seg_class, info in self._segmentation_map.items():
            red = info["RGB"][0]
            green = info["RGB"][1]
            blue = info["RGB"][2]
            self._points.loc[
                (self._points.red == red)
                & (self._points.green == green)
                & (self._points.blue == blue),
                "seg_class",
            ] = seg_class

    def _get_segmented_mesh_info(self) -> DataFrame:
        """
        This function combines the data `self._points` with `self._meshes`. so that
        all of the point data is now orgainzied by triangle mesh.

        Note that `self._fix_vertex_colors` and `self._assign_classes` methods need to
        be run prior to this method.

        :return: mesh DataFrame with segmentation information added.
        """
        assert self._meshes is not None
        cols_to_keep = [
            "x",
            "y",
            "z",
            "nx",
            "ny",
            "nz",
            "red",
            "green",
            "blue",
            "seg_class",
        ]
        mesh_df = self._meshes.copy(deep=True)
        for vertex in [1, 2, 3]:
            mesh_df = pd.merge(
                left=mesh_df,
                right=self._points[cols_to_keep].rename(
                    columns={col: col + f"_v{vertex}" for col in cols_to_keep}
                ),
                how="left",
                left_on=f"id_v{vertex}",
                right_index=True,
            )
        if mesh_df.apply(
            lambda x: (x.seg_class_v1 == x.seg_class_v2)
            & (x.seg_class_v2 == x.seg_class_v3),
            axis=1,
        ).all():
            mesh_df["seg_class"] = mesh_df.seg_class_v1
            mesh_df.drop(
                columns=["seg_class_v1", "seg_class_v2", "seg_class_v3"], inplace=True
            )
            return mesh_df
        raise NotImplementedError(
            "Way to handle mesh faces with vertices of different semantic class "
            + "has not been implemented, not possible to caluclate WWR"
        )

    @cached_property
    def wwr(self) -> float:
        """
        Calculate the Winows to Wall Ratio (WWR) for the 3D model.

        WWR is defined as the area of windows divded by area of the vertical walls of
        the building. The area of the windows is defined as the area of bounding box
        of the window facing away from the wall.
        """
        wall_area = self._get_all_wall_area()
        windows_area = self._get_all_window_area()
        return windows_area / wall_area

    def _get_all_wall_area(self) -> float:
        """
        Gets area of the building walls excluding the top and bottom face.

        The magnitude of the norm is divided by 2 as the norm magnitude represents the
        area of a rectangle formed by AB and AC from triangle ABC.
        """
        normals = self._mesh_df.loc[
            self._mesh_df.seg_class == "wall",
            [
                "face_normal_x",
                "face_normal_y",
                "face_normal_z",
                "face_normal_magnitude",
            ],
        ].to_numpy()
        cond = (normals[:, 0] != 0) | (normals[:, 1] != 0)
        side_walls = normals[cond, :]
        area = np.sum(side_walls[:, 3]) / 2
        return area

    def _get_all_roof_area(self) -> float:
        """
        Gets area of the building roof excluding all areas facing downwards.

        The magnitude of the norm is divided by 2 as the norm magnitude represents the
        area of a rectangle formed by AB and AC from triangle ABC.
        """
        normals = self._mesh_df.loc[
            self._mesh_df.seg_class == "roof",
            [
                "face_normal_x",
                "face_normal_y",
                "face_normal_z",
                "face_normal_magnitude",
            ],
        ].to_numpy()
        cond = normals[:, 2] >= 0
        roof_top = normals[cond, :]
        area = np.sum(roof_top[:, 3]) / 2
        return area

    def _get_all_window_area(self) -> float:
        """
        Calculates the area of all of the windows on the building.

        The area of the window is defined as the area of the bounding box of the window
        parallel to the building wall facing outwards.
        """
        window_areas = []
        for window_graph in self._window_graphs:
            window_areas.append(
                self._get_window_area_from_component_graph(window_graph)
            )

        return np.sum(window_areas)

    def _get_window_area_from_component_graph(self, graph: nx.MultiGraph) -> float:
        """
        `graph` is a fully connected graph that represents one window, whose node names
        represent points within the `self._points` DataFrame.

        :return: the area of the largest face of the bounding box of the window.
        """
        nodes = list(graph.nodes)
        node_locations = self._points.loc[nodes, ["x", "y", "z"]].to_numpy()
        bounding_box = np.stack(
            [np.min(node_locations, axis=0), np.max(node_locations, axis=0)], axis=1
        )
        measurements = np.diff(bounding_box, axis=1)
        return np.max([x * y for x, y in combinations(measurements, 2)])

    def _add_triangle_face_normals(self) -> None:
        """
        Adds columns which add the x, y, and z direction of the triangular face's unit
        normals as well as the magnitude of the face normal for all trianglar faces in
        `self.mesh_df`.

        Can only be run after `self._get_segmented_mesh_info`.
        """
        if self._mesh_df is None:
            raise RuntimeError("self.mesh_df does not exist.")
        AB = (
            self._mesh_df[["x_v2", "y_v2", "z_v2"]].to_numpy()
            - self._mesh_df[["x_v1", "y_v1", "z_v1"]].to_numpy()
        )
        AC = (
            self._mesh_df[["x_v3", "y_v3", "z_v3"]].to_numpy()
            - self._mesh_df[["x_v1", "y_v1", "z_v1"]].to_numpy()
        )
        A_normal = self._mesh_df[["nx_v1", "ny_v1", "nz_v1"]].to_numpy()
        B_normal = self._mesh_df[["nx_v2", "ny_v2", "nz_v2"]].to_numpy()
        C_normal = self._mesh_df[["nx_v3", "ny_v3", "nz_v3"]].to_numpy()

        face_normals = triangle_face_normals(
            AB=AB, AC=AC, A_normal=A_normal, B_normal=B_normal, C_normal=C_normal
        )
        face_normals_magnitude = np.linalg.norm(face_normals, axis=1).reshape(
            face_normals.shape[0], 1
        )
        face_unit_normals = face_normals / face_normals_magnitude

        self._mesh_df["face_normal_x"] = face_unit_normals[:, 0]
        self._mesh_df["face_normal_y"] = face_unit_normals[:, 1]
        self._mesh_df["face_normal_z"] = face_unit_normals[:, 2]
        self._mesh_df["face_normal_magnitude"] = face_normals_magnitude

    def _get_connected_meshes_list(
        self, mesh_df_slice: DataFrame
    ) -> list[nx.MultiGraph]:
        """
        `mesh_df_slice` should be a slice of self._mesh_df which contains faces related
        to meshes of interest. Creates a list of connected compoenent subgraphs made
        from the graph reprsentation of `mesh_df_slice`.
        """

        graph_df = pd.concat(
            [
                mesh_df_slice.loc[
                    :,
                    ["id_v1", "id_v2"],
                ].rename(columns={"id_v1": "source", "id_v2": "target"}),
                mesh_df_slice.loc[
                    :,
                    ["id_v2", "id_v3"],
                ].rename(columns={"id_v2": "source", "id_v3": "target"}),
                mesh_df_slice.loc[
                    :,
                    ["id_v3", "id_v1"],
                ].rename(columns={"id_v3": "source", "id_v1": "target"}),
            ],
            axis=0,
        ).reset_index(drop=True)
        G = nx.from_pandas_edgelist(graph_df, source="source", target="target")
        component_graphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]

        return component_graphs

    def _set_wwr_per_wall(self) -> None:
        """
        Calculates the wwr of each wall in `self._building_walls`
        """
        for wall in self._building_walls:
            windows_subgraph_list = self._get_windows_on_wall(wall)
            windows_area = []
            for window in windows_subgraph_list:
                windows_area.append(self._get_window_area_from_component_graph(window))
            wall.set_wwr(np.sum(windows_area))

    def _get_building_walls(self) -> list[BuildingWall]:
        """
        Finds the vertical building walls

        :return: a list containing the vetical facing walls that make up the
        Segmented3DModel.
        """
        building_walls = []
        wall_df = self._mesh_df[
            (self._mesh_df.seg_class == "wall")
            & (self._mesh_df.face_normal_z != 1)
            & (self._mesh_df.face_normal_z != -1)
        ]

        for _, group in wall_df.groupby(
            ["face_normal_x", "face_normal_y", "face_normal_z"]
        ):
            connected_components = self._get_connected_meshes_list(group)
            if len(connected_components) == 1:
                building_walls.append(BuildingWall(group))
                continue

            for component in connected_components:
                nodes = set(component.nodes)

                vertex_1_in_nodes = group.id_v1.isin(nodes)
                vertex_2_in_nodes = group.id_v2.isin(nodes)
                vertex_3_in_nodes = group.id_v3.isin(nodes)

                mask = np.all(
                    np.stack(
                        [
                            vertex_1_in_nodes.to_numpy(),
                            vertex_2_in_nodes.to_numpy(),
                            vertex_3_in_nodes.to_numpy(),
                        ],
                        axis=1,
                    ),
                    axis=1,
                )
                component_mesh_df = group.loc[mask, :]  # type: ignore
                building_walls.append(BuildingWall(component_mesh_df))

        return building_walls

    def _get_windows_on_wall(self, wall: BuildingWall) -> list[nx.MultiGraph]:
        """
        Finds all windows that are placed on a building wall
        """
        windows_mesh_df = self._mesh_df.loc[self._mesh_df.seg_class == "window"]
        windows_mesh_edges = pd.concat(
            [
                windows_mesh_df.loc[
                    :,
                    ["id_v1", "id_v2", "x_v1", "y_v1", "z_v1", "x_v2", "y_v2", "z_v2"],
                ],
                windows_mesh_df.loc[
                    :,
                    ["id_v2", "id_v3", "x_v2", "y_v2", "z_v2", "x_v3", "y_v3", "z_v3"],
                ].rename(
                    columns={
                        "id_v2": "id_v1",
                        "x_v2": "x_v1",
                        "y_v2": "y_v1",
                        "z_v2": "z_v1",
                        "id_v3": "id_v2",
                        "x_v3": "x_v2",
                        "y_v3": "y_v2",
                        "z_v3": "z_v2",
                    }
                ),
                windows_mesh_df.loc[
                    :,
                    ["id_v3", "id_v1", "x_v3", "y_v3", "z_v3", "x_v1", "y_v1", "z_v1"],
                ].rename(
                    columns={
                        "id_v3": "id_v1",
                        "x_v3": "x_v1",
                        "y_v3": "y_v1",
                        "z_v3": "z_v1",
                        "id_v1": "id_v2",
                        "x_v1": "x_v2",
                        "y_v1": "y_v2",
                        "z_v1": "z_v2",
                    }
                ),
            ]
        )

        # Center of the wall to the top most edge should generate vector that is on
        # the wall plane
        plane_vector_1 = wall.domain[:, 1] - wall.center

        # Use plane normal to find one more vector
        plane_vector_2 = np.cross(plane_vector_1, wall.direction)

        plane_point_1 = wall.domain[:, 1]
        plane_point_2 = wall.center
        plane_point_3 = wall.center + plane_vector_2

        intersecting_edges = find_if_line_and_plane_intersect(
            vertices_1=windows_mesh_edges[["x_v1", "y_v1", "z_v1"]].to_numpy(),
            vertices_2=windows_mesh_edges[["x_v2", "y_v2", "z_v2"]].to_numpy(),
            plane_points=np.stack(
                [plane_point_1, plane_point_2, plane_point_3], axis=0
            ),
            domain=wall.domain,
        )

        wall_nodes = set(
            np.concatenate(
                [
                    windows_mesh_edges.loc[intersecting_edges, "id_v1"].to_numpy(),
                    windows_mesh_edges.loc[intersecting_edges, "id_v2"].to_numpy(),
                ]
            )
        )

        windows_on_wall = []

        for window in self._window_graphs:
            if len(wall_nodes.intersection(window.nodes)) > 0:
                windows_on_wall.append(window)

        return windows_on_wall

    def write_out_building_wall_wwr(self, output_dir: Path, filename: str) -> None:
        json_dict = {}
        json_dict["overall"] = {"wwr": self.wwr}
        for i, wall in enumerate(self._building_walls):
            json_dict[f"{i}"] = {
                "center": wall.center.tolist(),
                "direction": wall.direction.tolist(),
                "domain": wall.domain.tolist(),
                "wwr": wall.wwr,
                "area": wall.area,
            }

        output_dir.mkdir(exist_ok=True)
        filename = filename.replace(".json", "")
        with open(Path(output_dir, filename + ".json"), "w") as j:
            json.dump(json_dict, j)
