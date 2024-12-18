import bpy
import numpy as np

from buildnet3d_dataset.blender_utils import deselect_all, select
from buildnet3d_dataset.material_factory import MaterialFactory
from buildnet3d_dataset.modules_manager.window import Window

import bmesh  # isort: skip # type: ignore


class ParametricWindow(Window):
    def __init__(
        self,
        id: int,
        segmentation_color: tuple[float, float, float],
        material: list[str],
        path_to_materials: str,
        width: float = 1.5,
        length: float = 0.04,
        height: float = 1.5,
        name: str = "window",
        mesh: bpy.types.Mesh | None = None,  # type: ignore
    ) -> None:
        """
        Parametric window object handling the mesh.

        A parametric window has a `name`, a `dimensions` (`width`,`length`,`height`)
        and an `id` and `segmentation_color` that will be used for semantic
        segmentation. A list of materials for the module must be provided in
        `materials` as well as the absolute path to the texture folder in
        `path_to_materials`. It must have a `mesh`; if none is given, a mesh is
        generated using the module's `name`. Finally, the window can be associated to a
        `volume` to which it will be attached.
        """
        # NOTE:
        # `np.random.randint` has been commented out as there is currently an issue
        # running the `bpy.ops.mesh.loopcut_slide()` function. The values of the bars
        # have been set to 0 so that this area of the code can be skipped.
        # Needs to be reverted in the future once fix is created from blender's end
        # or a new method is implemented by us. Submitted to blender/blender-addons for
        # fixing: https://projects.blender.org/blender/blender-addons/issues/105242

        self.h_bars = 0  # np.random.randint(0, 5)
        self.v_bars = 0  # np.random.randint(0, 5)
        super().__init__(
            id,
            segmentation_color,
            material,
            path_to_materials,
            name,
            width,
            length,
            height,
            mesh,
        )
        # `self.apply` adds material characteristics
        self.apply()

    def apply(self) -> None:
        _material = MaterialFactory(self.path_to_materials).produce("metall")
        self.mesh.active_material = _material.value

        bpy.ops.object.editmode_toggle()  # type: ignore
        _mesh = bmesh.from_edit_mesh(self.mesh.data)
        self._select_faces(_mesh)
        _glass = MaterialFactory(self.path_to_materials).produce("glass")
        deselect_all()
        select(self.mesh)

        self.mesh.data.materials.append(_glass.value)

        for face in _mesh.faces:
            if face.select:
                face.material_index = 1
        bpy.ops.object.editmode_toggle()  # type: ignore

    def _create(self) -> bpy.types.Mesh:  # type: ignore
        bpy.ops.mesh.primitive_cube_add(size=1.0)  # type: ignore
        bpy.ops.geometry.color_attribute_add(  # type: ignore
            color=self.segmentation_color
        )
        bpy.context.selected_objects[0].name = self.name  # type: ignore
        self.mesh = bpy.context.selected_objects[0]  # type: ignore
        select(self.mesh)
        bpy.ops.object.editmode_toggle()  # type: ignore
        bpy.ops.transform.resize(value=self.dimensions)  # type: ignore

        if self.h_bars > 0:
            self._cut()
        if self.v_bars > 0:
            self._cut(axis=4)

        _mesh = bmesh.from_edit_mesh(self.mesh.data)

        self._inset(_mesh)
        self._extrude(_mesh)

        bpy.ops.object.editmode_toggle()  # type: ignore
        return self.mesh

    def _cut(self, axis=1) -> None:
        # axis = 0 horizontal loops
        # axis = 4 vertical loops
        if axis == 1:
            _bars = self.h_bars
        else:
            axis = 4
            _bars = self.v_bars
        _dict = self._setup()
        with bpy.context.temp_override(  # type: ignore
            scene=_dict["scene"],
            region=_dict["region"],
            area=_dict["area"],
            space=_dict["space"],
        ):
            bpy.ops.mesh.loopcut_slide(  # type: ignore
                MESH_OT_loopcut={
                    "number_cuts": _bars,
                    "smoothness": 0,
                    "falloff": "INVERSE_SQUARE",
                    "object_index": 0,
                    "edge_index": axis,
                    "mesh_select_mode_init": (True, False, False),
                },
                TRANSFORM_OT_edge_slide={
                    "value": 0,
                    "mirror": False,
                    "snap": False,
                    "correct_uv": False,
                    "release_confirm": False,
                    "use_accurate": False,
                },
            )

    def _extrude(self, _mesh: bpy.types.Mesh) -> None:  # type: ignore
        self._select_faces(_mesh)
        bpy.ops.mesh.extrude_faces_move(  # type: ignore
            TRANSFORM_OT_shrink_fatten={
                "value": -0.01,
                "use_even_offset": True,
                "mirror": False,
                "snap": False,
                "release_confirm": False,
            }
        )

    def _inset(self, _mesh: bpy.types.Mesh) -> None:  # type: ignore
        self._select_faces(_mesh)

        bpy.ops.mesh.inset(  # type: ignore
            use_boundary=False,
            use_even_offset=False,
            use_relative_offset=True,
            thickness=np.random.uniform(0.01, 0.1),
            depth=0,
            use_outset=False,
            use_individual=True,
        )

    def _select_faces(self, _mesh: bpy.types.Mesh) -> None:  # type: ignore
        faces = sorted(
            [x for x in _mesh.faces], reverse=True, key=lambda x: x.calc_area()
        )
        for f in faces:
            f.select = False

        for f in faces[: max(1, self.h_bars + 1) * max(1, self.v_bars + 1) * 2]:
            f.select = True

    def _setup(self) -> dict:
        area = [
            x
            for x in bpy.context.window.screen.areas  # type: ignore
            if x.type == "VIEW_3D"
        ][0]
        space = area.spaces[0]
        region = [x for x in area.regions if x.type == "WINDOW"][0]
        return {
            "scene": bpy.context.scene,  # type: ignore
            "region": region,
            "area": area,
            "space": space,
        }

    def generate(self) -> "ParametricWindow":
        """Create a new instance of parametricWindow"""
        return ParametricWindow(
            self.id,
            self.segmentation_color[:3],
            self.material,
            self.path_to_materials,
            name=self.name,
            width=self.width,
            length=self.length,
            height=self.height,
        )

    @property
    def area(self) -> float:
        windows_dimensions = np.array(self.dimensions)
        return float(
            np.prod(
                np.delete(
                    windows_dimensions,
                    np.where(windows_dimensions == np.min(windows_dimensions)),
                )
            )
        )
