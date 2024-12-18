import unittest
from unittest.mock import patch

import bpy
import numpy as np

from buildnet3d_dataset.buildings.volume import Volume
from buildnet3d_dataset.modules_manager.parametric_window import (
    ParametricWindow,
)
from buildnet3d_dataset.modules_manager.roof import Roof
from buildnet3d_dataset.post_process_segmentation import (
    color_quantization,
    color_quantization_img,
)


class TestSegmentation(unittest.TestCase):
    def test_color_quantization_if_1_class(self) -> None:
        rgb_arr = np.array(
            [
                [90, 0, 10],
                [100, 100, 10],
                [0, 0, 0],
                [100, 100, 10],
                [100, 100, 10],
                [100, 100, 10],
                [0, 0, 10],
                [50, 110, 6],
                [10, 0, 0],
            ]
        )
        quantized_arr = color_quantization(rgb_arr, [(100, 100, 10)])
        self.assertTrue(
            (
                quantized_arr
                == np.array(
                    [
                        [100, 100, 10],
                        [100, 100, 10],
                        [100, 100, 10],
                        [100, 100, 10],
                        [100, 100, 10],
                        [100, 100, 10],
                        [100, 100, 10],
                        [100, 100, 10],
                        [100, 100, 10],
                    ]
                )
            ).all()
        )

    def test_color_quantization_if_4_class(self) -> None:
        rgb_arr = np.array(
            [
                [200, 0, 10],
                [0, 225, 240],
                [75, 10, 100],
                [125, 200, 10],
                [0, 200, 220],
                [100, 100, 10],
                [126, 220, 10],
                [50, 90, 30],
                [225, 30, 10],
            ]
        )
        quantized_arr = color_quantization(
            rgb_arr,
            [(255, 0, 0), (0, 255, 255), (50, 10, 75), (125, 200, 20)],
        )
        self.assertTrue(
            (
                quantized_arr
                == np.array(
                    [
                        [255, 0, 0],
                        [0, 255, 255],
                        [50, 10, 75],
                        [125, 200, 20],
                        [0, 255, 255],
                        [125, 200, 20],
                        [125, 200, 20],
                        [50, 10, 75],
                        [255, 0, 0],
                    ]
                )
            ).all()
        )

    def test_color_quantization_if_11_class(self) -> None:
        rgb_arr = np.array(
            [
                [90, 0, 10],
                [100, 100, 10],
                [0, 0, 0],
                [100, 100, 10],
                [100, 100, 10],
                [100, 100, 10],
                [0, 0, 10],
                [50, 110, 6],
                [10, 0, 0],
            ]
        )
        quantized_arr = color_quantization(
            rgb_arr,
            [
                (90, 0, 10),
                (100, 100, 10),
                (0, 0, 0),
                (100, 100, 10),
                (100, 100, 10),
                (100, 100, 10),
                (0, 0, 10),
                (50, 110, 6),
                (10, 0, 0),
                (20, 30, 600),
                (110, 50, 100),
            ],
        )
        self.assertTrue(
            (
                quantized_arr
                == np.array(
                    [
                        [90, 0, 10],
                        [100, 100, 10],
                        [0, 0, 0],
                        [100, 100, 10],
                        [100, 100, 10],
                        [100, 100, 10],
                        [0, 0, 10],
                        [50, 110, 6],
                        [10, 0, 0],
                    ]
                )
            ).all()
        )

    def test_color_quantization_img_int(self) -> None:
        rgb_image = np.array(
            [
                [[0, 0, 0], [100, 100, 10], [0, 0, 0]],
                [[100, 100, 10], [110, 99, 2], [100, 100, 10]],
                [[0, 0, 10], [50, 110, 6], [10, 0, 0]],
            ]
        )
        post_processed_image = color_quantization_img(
            rgb_image, [(0, 0, 0), (100, 100, 10)]
        )
        self.assertTrue(
            (
                post_processed_image
                == np.array(
                    [
                        [[0, 0, 0], [100, 100, 10], [0, 0, 0]],
                        [[100, 100, 10], [100, 100, 10], [100, 100, 10]],
                        [[0, 0, 0], [100, 100, 10], [0, 0, 0]],
                    ]
                )
            ).all()
        )

    def test_color_quantization_img_float(self) -> None:
        rgb_image = np.array(
            [
                [
                    [0 / 255, 0 / 255, 0 / 255],
                    [100 / 255, 100 / 255, 10 / 255],
                    [0 / 255, 0 / 255, 0 / 255],
                ],
                [
                    [100 / 255, 100 / 255, 10 / 255],
                    [110 / 255, 99 / 255, 2 / 255],
                    [100 / 255, 100 / 255, 10 / 255],
                ],
                [
                    [0 / 255, 0 / 255, 10 / 255],
                    [50 / 255, 110 / 255, 6 / 255],
                    [10 / 255, 0 / 255, 0 / 255],
                ],
            ]
        )
        post_processed_image = color_quantization_img(
            rgb_image, [(0 / 255, 0 / 255, 0 / 255), (100 / 255, 100 / 255, 10 / 255)]
        )
        self.assertTrue(
            (
                post_processed_image
                == np.array(
                    [
                        [
                            [0.0, 0.0, 0.0],
                            [100 / 255, 100 / 255, 10 / 255],
                            [0.0, 0.0, 0.0],
                        ],
                        [
                            [100 / 255, 100 / 255, 10 / 255],
                            [100 / 255, 100 / 255, 10 / 255],
                            [100 / 255, 100 / 255, 10 / 255],
                        ],
                        [
                            [0.0, 0.0, 0.0],
                            [100 / 255, 100 / 255, 10 / 255],
                            [0.0, 0.0, 0.0],
                        ],
                    ]
                )
            ).all()
        )

    @patch("buildnet3d_dataset.buildings.volume.Volume._nest")
    def test_volume_color_attribute(self, mock_nest) -> None:
        segmentation_color = (1.0, 0.25, 0.6)
        volume = Volume(0, "volume", 1, 1, 1, segmentation_color=segmentation_color)
        color_data = (
            bpy.data.objects[volume.mesh.name].data.attributes["Color"].data[0].color
        )
        color_attribute = [c for c in color_data]
        for ret, truth, val in zip(
            color_attribute, [1.0, 0.25, 0.6, 1.0], ["red", "green", "blue", "alpha"]
        ):
            self.assertAlmostEqual(ret, truth, msg=f"{val} value not matching")

    @patch(
        "buildnet3d_dataset.modules_manager."
        + "parametric_window.ParametricWindow.apply"
    )
    def test_parametric_window_color_attribute(self, mock_apply) -> None:
        materials = ["glass"]
        segmentation_color = (0.5, 0.34, 0.9)
        p_window = ParametricWindow(
            id=0,
            segmentation_color=segmentation_color,
            material=materials,
            path_to_materials="./",
        )
        color_data = (
            bpy.data.objects[p_window.mesh.name].data.attributes["Color"].data[0].color
        )
        color_attribute = [c for c in color_data]
        for ret, truth, val in zip(
            color_attribute, [0.5, 0.34, 0.9, 1.0], ["red", "green", "blue", "alpha"]
        ):
            self.assertAlmostEqual(ret, truth, msg=f"{val} value not matching")

    @patch("buildnet3d_dataset.buildings.volume.Volume._nest")
    def test_roof_color_attribute(self, mock_nest) -> None:
        segmentation_color = (0.7, 0.2, 0.87)
        roof = Roof(
            semantic_id=0,
            segmentation_color=segmentation_color,
            width=1,
            length=1,
            thickness=1,
            location=(0, 0, 0),
        )
        color_data = (
            bpy.data.objects[roof.mesh.name].data.attributes["Color"].data[0].color
        )
        color_attribute = [c for c in color_data]
        for ret, truth, val in zip(
            color_attribute, [0.7, 0.2, 0.87, 1.0], ["red", "green", "blue", "alpha"]
        ):
            self.assertAlmostEqual(ret, truth, msg=f"{val} value not matching")
