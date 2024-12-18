import unittest

import numpy as np

from buildnet3d_dataset.mesh_tools.mesh_utils import (
    check_angles,
    find_if_line_and_plane_intersect,
    npsumdot,
    triangle_face_normals,
)


class TestMeshUtils(unittest.TestCase):
    def test_npsumdot(self) -> None:
        test_array1 = np.array([[1, 0, 1], [2, 1, 1], [-1, 10, -5], [0, 0, 0]])

        test_array2 = np.array([[1, 1, 1], [5, 0, -1], [10, 2, 1], [3, 3, 2]])

        truth_array = np.array([2, 9, 5, 0])

        ret_array = npsumdot(test_array1, test_array2)

        self.assertTrue(np.all(ret_array == truth_array))

    def test_check_angles(self) -> None:
        test_face_normals = np.array(
            [[1, 0, 0], [-1, 0, 0], [0, 0, 1], [1, 2, 0], [0, -5, 9], [-2, -3, -4]]
        )

        test_vertex_normals = np.array(
            [
                [[1, 2, 0], [-1, 2, 0], [1, 2, 10]],
                [[0, 10, 10], [-1, -1, -1], [2, 0, 0]],
                [[2, 2, -2], [0, 0, 1], [0, 1, 1]],
                [[-1, 1, 0], [-1, 1, 10], [-1, 0, 2]],
                [[-1, 2, -1], [1, 4, 5], [25, -4, -4]],
                [[0, -1, 0], [2, -1, -5], [1, 2, -2]],
            ]
        ).transpose((0, 2, 1))

        truth_angle_check = np.array(
            [
                [False, True, False],
                [False, False, True],
                [True, False, False],
                [False, False, True],
                [True, False, True],
                [False, False, False],
            ]
        )

        ret_angle_check = check_angles(test_face_normals, test_vertex_normals)
        self.assertTrue(np.all(ret_angle_check == truth_angle_check))

    def test_triangle_face_normals(self) -> None:
        AB = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 1, 1],
                [0, 1, 20],
                [30, -5, 1],
                [-1, -3, -2],
                [4, -1, 1],
            ]
        )

        AC = np.array(
            [
                [0, 1, 0],
                [1, 0, 0],
                [0, 1, 1],
                [1, 1, 0],
                [30, -5, 1],
                [0, 1, 20],
                [4, -1, 1],
                [-1, -3, -2],
            ]
        )

        A_norm = np.array(
            [
                [0, 0, 1],
                [0, 0, 1],
                [2, -2, -2],
                [2, -2, -2],
                [-7, -22, 3],
                [-7, -22, 3],
                [1, -1, 0],
                [1, -1, 0],
            ]
        )

        B_norm = np.array(
            [
                [0, 0, 1],
                [0, 0, 1],
                [0, -2, 3],
                [0, -2, 3],
                [33, -200, 10],
                [33, -200, 10],
                [0, 0, 1],
                [0, 0, 1],
            ]
        )

        C_norm = np.array(
            [
                [0, 0, 1],
                [0, 0, 1],
                [2, 0, 1],
                [2, 0, 1],
                [10, -4, 10],
                [10, -4, 10],
                [0, -5, 1],
                [0, -5, 1],
            ]
        )

        truth_triangle_normals = np.array(
            [
                [0, 0, 1],
                [0, 0, 1],
                [1, -1, 1],
                [1, -1, 1],
                [-101, -600, 30],
                [-101, -600, 30],
                [-5, -7, 13],
                [-5, -7, 13],
            ]
        )

        ret_triangle_normals = triangle_face_normals(AB, AC, A_norm, B_norm, C_norm)

        self.assertTrue(np.all(ret_triangle_normals == truth_triangle_normals))

    def test_find_if_line_and_plane_intersect_on_plane(self):
        plane_vertex_1 = np.array([3, 0, 0])
        plane_vertex_2 = np.array([0, 2, 0])
        plane_vertex_3 = np.array([0, 0, -1])
        domain = np.array([[2, 5], [-2, 2], [-5, 0.4]])
        vertices_1 = np.array(
            [
                [2, -1, -0.833333333],
                [4, 0, 0.333333333],
                [3, 1, 0.5],
                [4, 0, 0.333333333],
            ]
        )
        vertices_2 = np.array(
            [
                [3, -2, -1],
                [6, 1.5, 1.75],
                [0, 0, -1],
                [2.5, 0.5, 0.083333333333],
            ]
        )
        truth_arr = np.array([True, False, False, True])

        ret_arr = find_if_line_and_plane_intersect(
            vertices_1=vertices_1,
            vertices_2=vertices_2,
            plane_points=np.stack(
                [plane_vertex_1, plane_vertex_2, plane_vertex_3], axis=0
            ),
            domain=domain,
        )

        self.assertTrue(np.all(ret_arr == truth_arr))

    def test_find_if_line_and_plane_intersect_one_point_on_plane(self):
        plane_vertex_1 = np.array([3, 0, 0])
        plane_vertex_2 = np.array([0, 2, 0])
        plane_vertex_3 = np.array([0, 0, -1])
        domain = np.array([[2, 5], [-2, 2], [-5, 0.4]])
        vertices_1 = np.array(
            [
                [2, -1, -0.833333333],
                [4, 0, 0.333333333],
                [4, -10, 1.5],
                [4, -2, 0.2],
            ]
        )
        vertices_2 = np.array(
            [
                [1, 1, -1],
                [4, -1.5, 1.75],
                [0, 0, -1],
                [2.5, 0.5, 0.083333333333],
            ]
        )
        truth_arr = np.array([True, True, True, True])

        ret_arr = find_if_line_and_plane_intersect(
            vertices_1=vertices_1,
            vertices_2=vertices_2,
            plane_points=np.stack(
                [plane_vertex_1, plane_vertex_2, plane_vertex_3], axis=0
            ),
            domain=domain,
        )

        self.assertTrue(np.all(ret_arr == truth_arr))

    def test_find_if_line_and_plane_intersect_cross_plane(self):
        plane_vertex_1 = np.array([3, 0, 0])
        plane_vertex_2 = np.array([0, 2, 0])
        plane_vertex_3 = np.array([0, 0, -1])
        domain = np.array([[2, 5], [-2, 2], [-5, 0.4]])
        vertices_1 = np.array(
            [
                [1, -2, 0],
                [5, 6, -3],
                [4, 1.1, 0.6],
                [3, -2, 1],
            ]
        )
        vertices_2 = np.array(
            [
                [3.1, -1.15, -3],
                [5, 1.5, 1.75],
                [6, 3, -2],
                [2.2, 0.4, 0.1],
            ]
        )
        truth_arr = np.array([True, True, False, False])

        ret_arr = find_if_line_and_plane_intersect(
            vertices_1=vertices_1,
            vertices_2=vertices_2,
            plane_points=np.stack(
                [plane_vertex_1, plane_vertex_2, plane_vertex_3], axis=0
            ),
            domain=domain,
        )

        self.assertTrue(np.all(ret_arr == truth_arr))

    def test_find_if_line_and_plane_intersect_mixed(self):
        plane_vertex_1 = np.array([3, 0, 0])
        plane_vertex_2 = np.array([0, 2, 0])
        plane_vertex_3 = np.array([0, 0, -1])
        domain = np.array([[2, 5], [-2, 2], [-5, 0.4]])
        vertices_1 = np.array(
            [
                [5, -2, -0.3333333333],
                [3.4, -1.5, -1.5],
                [4, -3, -1.16666666666],
                [4, 2, 0.1],
                [3, -2, -0.3333333333],
                [3, -2, -1],
                [5, -2, -2],
                [3, 1, 0.5],
            ]
        )
        vertices_2 = np.array(
            [
                [2, 0.1, -0.28333333333],
                [2.1, 0.1, -0.28333333333],
                [5, -2, -0.3333333333],
                [-1, -1, -1],
                [2, 0, -0.28333333333],
                [2, 0.1, -0.28333333333],
                [1, -5, -0.3333333333],
                [0, 0, -1],
            ]
        )
        truth_arr = np.array([True, False, False, True, False, True, True, False])

        ret_arr = find_if_line_and_plane_intersect(
            vertices_1=vertices_1,
            vertices_2=vertices_2,
            plane_points=np.stack(
                [plane_vertex_1, plane_vertex_2, plane_vertex_3], axis=0
            ),
            domain=domain,
        )

        self.assertTrue(np.all(ret_arr == truth_arr))
