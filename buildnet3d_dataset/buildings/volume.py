import bpy
import numpy as np
from bpy.types import Object  # type: ignore

from buildnet3d_dataset.blender_utils import (
    deselect_all,
    extrude,
    intersection_check,
    select,
)
from buildnet3d_dataset.material import Material
from buildnet3d_dataset.modules_manager.module import Module

import bmesh  # isort: skip #type: ignore


class Volume:
    """
    Class that represents one volume of a building.
    """

    mesh: Object  # type: ignore

    def __init__(
        self,
        id: int,
        name: str,
        width: float = 1.0,
        length: float = 1.0,
        height: float = 1.0,
        location: tuple[float, float, float] = (0.0, 0.0, 0.0),
        floor_height: float = 2.7,
        segmentation_color: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        """
        Creates a `Volume` that is used to construct the base structure of a building
        with the input `width`, `length`, `height` and `floor height` at `location` in
        the world space of blender. `id` and `segmentation_color` is used for semantic
        segmentation of the exported mesh.
        """

        ##############################################
        self.width = width
        self.length = length
        self.height = height
        self.dimensions = (width, length, height)
        self.floor_height = floor_height
        self._initial_position = location
        self.segmentation_color = (
            segmentation_color[0],
            segmentation_color[1],
            segmentation_color[2],
            1,
        )
        self.id = id
        self.name = name
        self.create()

    def apply(self, material: Material) -> None:
        self.mesh.active_material = material.value
        if "Mapping" in material.value.node_tree.nodes:
            material.value.node_tree.nodes["Mapping"].inputs[3].default_value[0] = (
                self.height * 10
            )
            material.value.node_tree.nodes["Mapping"].inputs[3].default_value[1] = (
                self.height
            )
            material.value.node_tree.nodes["Mapping"].inputs[3].default_value[2] = (
                self.height * 10
            )  # self.width
            material.value.node_tree.nodes["Mapping"].inputs[3].default_value /= 2
            self.mesh.active_material = material.value

    def create(self) -> None:
        """
        Function that creates a mesh based on the input parameters.
        :return:
        """
        deselect_all()
        bpy.ops.mesh.primitive_plane_add(  # type: ignore
            location=self._initial_position
        )
        bpy.ops.transform.resize(  # type: ignore
            value=(self.width / 2, self.length / 2, 1.0)
        )
        bpy.ops.geometry.color_attribute_add(  # type: ignore
            color=self.segmentation_color
        )
        bpy.context.selected_objects[0].name = self.name  # type: ignore
        self.name = bpy.context.selected_objects[0].name  # type: ignore
        self.mesh = bpy.data.objects[self.name]  # type: ignore
        self._nest()
        self._extrude()
        self.mesh["inst_id"] = self.id
        self.mesh.pass_index = self.id
        deselect_all()

    def _extrude(self) -> None:
        """
        Function that extrudes the plane in order to create a mesh.
        :return:
        """
        deselect_all()
        if self.mesh:
            extrude(self.mesh, self.height)

    def _nest(self) -> bpy.types.Collection:  # type: ignore
        deselect_all()
        names = [x.name for x in bpy.data.collections]  # type: ignore
        if self.name not in names:
            bpy.data.collections.new(self.name)  # type: ignore
            bpy.data.collections["Building"].children.link(  # type: ignore
                bpy.data.collections[self.name]  # type: ignore
            )
            bpy.data.collections[self.name].objects.link(  # type: ignore
                bpy.data.objects[self.mesh.name]  # type: ignore
            )
            return bpy.data.collections[self.name]  # type: ignore

    def triangulate(self) -> None:
        deselect_all()
        if self.mesh is None:
            return

        select(self.mesh)
        bpy.ops.object.modifier_add(type="TRIANGULATE")  # type: ignore
        bpy.ops.object.modifier_apply()  # type: ignore

    def collide(self, mesh: Object) -> bool:
        """
        Checks to see if `mesh` is either touching or contained within `self.mesh`.

        :returns: `True` if `mesh` is touching or within volume, `false` otherwise
        """
        if intersection_check(self.mesh, mesh):
            return True

        min_coords_self = np.array(
            [
                self.position[0] - self.width / 2,
                self.position[1] - self.length / 2,
                self.position[2] - self.height / 2,
            ]
        )

        max_coords_self = np.array(
            [
                self.position[0] + self.width / 2,
                self.position[1] + self.length / 2,
                self.position[2] + self.height / 2,
            ]
        )

        bm = bmesh.new()
        bm.from_mesh(mesh.data)
        bm.transform(mesh.matrix_basis)
        bm.verts.ensure_lookup_table()
        mesh_verts_arr = np.zeros(shape=(len(bm.verts), 3))

        for i, vert in enumerate(bm.verts):
            mesh_verts_arr[i, :] = vert.co

        min_coords_mesh = np.min(mesh_verts_arr, axis=0)
        max_coords_mesh = np.max(mesh_verts_arr, axis=0)

        if np.all(min_coords_mesh >= min_coords_self) & np.all(
            max_coords_mesh <= max_coords_self
        ):
            return True

        return False

    def collide_with_other_modules(self, module: Module) -> bool:
        """
        Checks to see if the provided `module` is touching any other modules
        associated with `self` that do not share the same `self.name`.

        Compares the names of modules to not compare against similar module types.

        :return: `True` if collision found, `False` otherwise.
        """
        for sub in [
            x.name
            for x in bpy.data.collections[self.name].children  # type: ignore
        ]:
            if module.name not in sub and sub not in module.name:
                other_module = bpy.data.collections[sub].objects  # type: ignore
                if intersection_check(module.mesh, other_module):
                    return True
        return False

    def color_volume(self) -> None:
        """
        Function to add color attribute to all nodes of a volume.
        """
        deselect_all()
        select(self.mesh)
        bpy.ops.geometry.color_attribute_add(  # type: ignore
            color=self.segmentation_color
        )
        deselect_all()

    @property
    def position(self) -> tuple[float, float, float]:
        return tuple(self.mesh.location)  # type: ignore
