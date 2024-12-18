from pathlib import Path

import bpy
import matplotlib.image

from buildnet3d_dataset.blender_utils import deselect_all, select_all
from buildnet3d_dataset.compositor_trees.depth_tree import DepthTree
from buildnet3d_dataset.compositor_trees.mask_node_tree import MaskNodeTree
from buildnet3d_dataset.compositor_trees.norm_tree import NormTree
from buildnet3d_dataset.post_process_segmentation import (
    color_quantization_img,
)


class Renderer:
    """
    Class that manages the scene rendering.
    """

    def __init__(
        self,
        output_folder: Path,
        image_size: tuple[int, int],
        nbr_modules: int,
        mandatory_class_color: dict[int, tuple[float, float, float]],
    ) -> None:
        """
        Initializes renderer settings such as engine, view layers, and rendering
        pipelines. `output_folder` is the base path that the outputs will be saved to.
        `image_size` is the desired image size of the rendered images. `nbr_modules` is
        the number of modules applied to the volumes in the building.
        `mandatory_class_color` is a dictionary with object semantic IDs as the keys
        and corresponding semantic class colors as values. The semantic class colors
        are represented by RGB tuples in the linear colour space.
        """
        bpy.types.ImageFormatSettings.color_mode = "RGBA"  # type: ignore
        self._scene_name = bpy.data.scenes[-1].name  # type: ignore
        self._scene = bpy.data.scenes[self._scene_name]  # type: ignore
        self.engine = "CYCLES"
        self._scene.render.engine = self.engine
        self._scene.view_layers["View Layer"].use_pass_object_index = True
        self._scene.view_layers["View Layer"].use_pass_normal = True
        self._scene.render.image_settings.color_mode = "RGBA"
        self._scene.render.resolution_x = image_size[0]
        self._scene.render.resolution_y = image_size[1]
        self._output_folder = output_folder
        self._mandatory_class_color = mandatory_class_color
        self._mask_tree = MaskNodeTree(
            base_path=self._output_folder,
            nbr_modules=nbr_modules,
            class_color_map=self._mandatory_class_color,
        )
        self._depth_tree = DepthTree(base_path=self._output_folder)
        self._norm_tree = NormTree(base_path=self._output_folder)

    def render(self, sub_path: Path) -> None:
        """
        Sets up camera view, updates save paths, renders all images, and quantizes
        colours in segmentation.
        """

        select_all()
        bpy.ops.view3d.camera_to_view_selected()  # type: ignore
        deselect_all()
        self._mask_tree.sub_path = sub_path
        self._depth_tree.sub_path = sub_path
        self._norm_tree.sub_path = sub_path

        self._render(sub_path)

        last_mask_file = self._mask_tree.full_path
        image = matplotlib.image.imread(last_mask_file)
        post_processed_image = color_quantization_img(
            image, list(self._mandatory_class_color.values())
        )
        matplotlib.image.imsave(last_mask_file, post_processed_image)

    def _render(self, sub_path: Path) -> None:
        """
        Sets up renderer formatting, image save path and triggers render and FileOutput
        Node piplines from blender. `sub_path` is the remaining pathing from the
        "images" folder in `self._output_folder` to the image save location as well as
        the name of the image without file extension.
        """
        image_settings = bpy.context.scene.render.image_settings  # type: ignore
        image_settings.file_format = "PNG"
        image_settings.color_depth = "8"
        bpy.data.scenes[self._scene_name].render.engine = self.engine  # type: ignore
        bpy.ops.render.render()  # type: ignore
        image_path = Path(self._output_folder, "images")
        image_path.mkdir(exist_ok=True)
        bpy.data.images["Render Result"].save_render(  # type: ignore
            f"{image_path}/{sub_path}.png"
        )
