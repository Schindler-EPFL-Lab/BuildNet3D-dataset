import math

import bpy

from buildnet3d_dataset.modules_manager.module import Module


class Window(Module):
    def __init__(
        self,
        id: int,
        segmentation_color: tuple[float, float, float],
        material: list[str],
        path_to_materials: str,
        name: str = "window",
        width: float = 1.0,
        length: float = 1.0,
        height: float = 1.0,
        mesh: bpy.types.Mesh | None = None,  # type: ignore
    ) -> None:
        """
        Window object handling the mesh.

        A window has a `name`, a `dimensions` (`width`,`length`,`height`) and an `id`
        and `segmentation_color` that will be used for semantic segmentation. A list of
        materials for the module must be provided in `materials` as well as the
        absolute path to the texture folder in `path_to_materials`. It must have a
        `mesh`; if none is given, a mesh is generated using the module's `name`.
        Finally, the window can be associated to a `volume` to which it will be
        attached.
        """
        super().__init__(
            id,
            segmentation_color,
            material,
            path_to_materials,
            width,
            length,
            height,
            name,
            mesh,
        )
        self._triangulate()
        self.y_offset = 1.0

    def _create(self) -> None:
        bpy.ops.mesh.primitive_cube_add(size=1.0)  # type: ignore
        bpy.ops.transform.resize(value=self.dimensions)  # type: ignore
        bpy.ops.geometry.color_attribute_add(  # type: ignore
            color=self.segmentation_color
        )
        bpy.context.selected_objects[0].name = self.name  # type: ignore
        return bpy.context.selected_objects[0]  # type: ignore

    class ModuleConnector(Module.ModuleConnector):
        """This class is a connecter for a windows element"""

        def _connect(self) -> None:
            if self.module.mesh is None:
                raise RuntimeError("the module's mesh was not initialized")
            if self.axis == 0:
                self.module.mesh.rotation_euler[2] = math.radians(90)

            super()._connect()

            self.module.mesh.location[2] = 0
