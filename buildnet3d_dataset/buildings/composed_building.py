import inspect
from abc import ABC, abstractmethod
from copy import copy
from pathlib import Path

import bpy
import numpy as np
from pyntcloud import PyntCloud

from buildnet3d_dataset.blender_utils import (
    deselect_all,
    dissolve_faces,
    gancio,
    get_min_max,
    mesh_boolean_union,
)
from buildnet3d_dataset.buildings.volume import Volume
from buildnet3d_dataset.material import Material
from buildnet3d_dataset.modules_manager.module import Module
from buildnet3d_dataset.modules_manager.parametric_window import (
    ParametricWindow,
)
from buildnet3d_dataset.modules_manager.roof import Roof


class ComposedBuilding(ABC):
    """
    Class that represents a building composed of one or several volumes.
    """

    def __init__(
        self,
        semantic_id: int,
        segmentation_color: tuple[float, float, float],
        template: bool = False,
    ) -> None:
        """
        `semantic_id` and `segmentation_color` are used for semantic segmentation
        tasks and `template` is used to specify if the generated object should be made
        in blender. If `template` is `True` then the ComposedBuilding is not generated.
        """
        self.semantic_id = semantic_id
        self.segmentation_color = segmentation_color
        self._overlap_distance = 1
        self.volumes: list[Volume] = []
        self.roofs: list[Volume] = []
        self._windows_applied = 0
        self._windows_area = 0
        if template is False:
            self._nest()
            self._generate()

    def demolish(self) -> None:
        for mesh in bpy.data.collections["Building"].objects:  # type: ignore
            try:
                deselect_all()
                mesh.select_set(True)
                bpy.ops.object.delete()  # type: ignore
            except Exception:
                pass

    def get_bb(self) -> list[float]:
        """
        Function that gets the bounding box of the Building in blender coordinate
        space.
        :return: bounding box, [width_from, height_from, width_to, height_to]
        # TODO: 3d bounding box
        """
        x_min, y_min, x_max, y_max = list(get_min_max(self.volumes[0].mesh, 0)) + list(
            get_min_max(self.volumes[0].mesh, 1)
        )
        for v in self.volumes[1:]:
            _bb = list(get_min_max(v.mesh, 0)) + list(get_min_max(v.mesh, 1))
            x_min, y_min = float(min(_bb[0], x_min)), float(min(_bb[1], y_min))
            x_max, y_max = float(max(_bb[2], x_max)), float(max(_bb[3], y_max))
        return [round(x_min, 3), round(y_min, 3), round(x_max, 3), round(y_max, 3)]

    def model_path(self, data_folder: str) -> str:
        model_path = Path(data_folder, "model")
        model_path.mkdir(exist_ok=True)
        return str(model_path)

    def cloud_path(self, data_folder: str) -> str:
        cloud_path = Path(data_folder, "pointCloud")
        cloud_path.mkdir(exist_ok=True)
        return str(cloud_path)

    def blend_path(self, data_folder: str) -> str:
        blend_path = Path(data_folder, "blend")
        blend_path.mkdir(exist_ok=True)
        return str(blend_path)

    def _select_building(self) -> None:
        deselect_all()
        for v in self.volumes:
            v.mesh.select_set(True)

    def save_obj(self, filename: int, data_folder: str) -> None:
        self._select_building()
        bpy.ops.wm.obj_export(  # type: ignore
            filepath="{}/{}.{}".format(self.model_path(data_folder), filename, "obj")
        )

    def save_dae(self, filename: int, data_folder: str) -> None:
        self._select_building()
        bpy.ops.wm.collada_export(  # type: ignore
            filepath="{}/{}.{}".format(self.model_path(data_folder), filename, "dae")
        )

    def save_stl(self, filename: int, data_folder: str) -> None:
        self._select_building()
        bpy.ops.export_mesh.stl(  # type: ignore
            filepath="{}/{}.{}".format(self.cloud_path(data_folder), filename, "stl")
        )

    def save_ply(
        self, filename: int, data_folder: str, point_number: int = 1028
    ) -> None:
        """Save the mesh as a ply file at the path `data_folder` wiht name `filename`.

        If `point_number` is less or equal to 0 zero, no sampling of the mesh is done.
        """
        self._select_building()

        # Export mesh in .ply format
        bpy.ops.wm.ply_export(  # type: ignore
            filepath="{}/{}.{}".format(self.cloud_path(data_folder), filename, "ply"),
            export_normals=True,
            ascii_format=True,
        )

        # Load .ply using pyntcloud and get point_number samples to create a point cloud
        cloud = PyntCloud.from_file(
            "{}/{}.ply".format(self.cloud_path(data_folder), filename)
        )
        if point_number > 0:
            cloud = cloud.get_sample(
                "mesh_random",
                n=point_number,
                rgb=False,
                normals=True,
                as_PyntCloud=True,
            )
        cloud.to_file("{}/{}.ply".format(self.cloud_path(data_folder), filename))

    def save_segmented_mesh(self, filename: int, data_folder: str) -> None:
        """
        Saves a mesh with semantic information with `filename` to `data_folder`/model/.
        The mesh is stored as a `.ply` file.
        """
        bpy.ops.wm.ply_export(  # type: ignore
            filepath=self.model_path(data_folder) + f"/{filename}_segmented.ply",
            export_normals=True,
            export_triangulated_mesh=True,
            ascii_format=True,
            export_colors="LINEAR",
            apply_modifiers=True,
        )

    def save_blend(self, filename: int, data_folder: str) -> None:
        self._select_building()
        bpy.ops.wm.save_as_mainfile(  # type: ignore
            filepath="{}/{}.{}".format(self.blend_path(data_folder), filename, "blend")
        )

    def _nest(self) -> None:
        if "Building" not in [x.name for x in bpy.data.collections]:  # type: ignore
            bpy.data.collections.new("Building")  # type: ignore

    def add_roof(
        self,
        roof_id: int,
        roof_segmentation_color: tuple[float, float, float],
        overhang: float = 0.5,
        thickness: float = 0.5,
    ):
        for v in self.volumes:
            self.roofs.append(
                Roof.from_volume(
                    semantic_id=roof_id,
                    segmentation_color=roof_segmentation_color,
                    volume=v,
                    overhang=overhang,
                    thickness=thickness,
                )
            )

    def remove_overlap_faces(self) -> None:
        """
        Removes the overlaped faces for the building walls and roof.
        """
        ComposedBuilding._remove_overlap_faces(self.volumes)
        ComposedBuilding._remove_overlap_faces(self.roofs)

    def clean_and_triangulate(self) -> None:
        """
        Dissolves unnecessary faces from volumes and then triangulates them
        """
        for v in self.volumes:
            dissolve_faces(v.mesh)
            v.triangulate()
        for r in self.roofs:
            dissolve_faces(r.mesh)
            r.triangulate()

    def clone(self, **kwargs) -> "ComposedBuilding":
        """
        General cloning method that creates a new instance of `self` using the
        properties of `self`. Note that any inputs to the `self.__init__` function that
        are not saved as properties of the same name need to be supplied to the clone
        method as a `kwargs`. Will return a non-template copy of building unless
        specified otherwise.
        """
        init_inputs = inspect.signature(self.__init__).parameters.keys()
        self_parameters = {}
        for key in self.__dict__.keys():
            if key.startswith("_"):
                self_parameters[key.lstrip("_")] = self.__dict__[key]
                continue
            self_parameters[key] = self.__dict__[key]

        clone_inputs = {
            key: self_parameters[key] for key in self_parameters.keys() & init_inputs
        }

        clone_inputs.update(kwargs)
        return self.__class__(**clone_inputs)

    def add_window_elements(
        self,
        module: ParametricWindow,
        x_step: int,
        offset: tuple[int, int, int, int] = (2, 1, 2, 1),
        prob_windows: float = 1,
    ) -> None:
        """
        Funciton to apply Windows to the surface of a building in a grid pattern.
        `module` a ParametricWindow object, `x_step` is the space between the applied
        windows, `offset` is a tuple containing offsets from the left, top, right, and
        bottom, edges of each volume face where windows should not be placed in.
        `prob_windows` is the probability that a window will be placed at each location.
        """

        for vol in self.volumes:
            for axis in range(2):
                for side in range(2):
                    module_copy = module.generate()
                    if axis == 0:
                        module_copy.mesh.rotation_euler[2] = np.radians(90)
                    windows_applied = self.apply_modules_grid(
                        module=module_copy,
                        volume=vol,
                        x_step=x_step,
                        offset=offset,
                        apply_probability=prob_windows,
                        axis=axis,
                        side=side,
                    )
                    self._windows_applied += windows_applied
                    self._windows_area += windows_applied * module.area
        module.remove()

    def apply_modules_grid(
        self,
        module: Module,
        volume: Volume,
        x_step: int,
        offset: tuple[int, int, int, int],
        apply_probability: float,
        axis: int,
        side: int,
    ) -> int:
        """
        This method dispatch `module` objects along a grid.

        `module` is the Module object that will be applied to the `volume`.
        'x_step` and `self.floor_height` create the density of the grid. `x_step` is
        the distance between the centers of subsequent modules. `offset` indicates the
        minimum distance from the border each volume face in the order left, top,
        right, bottom. `axis` and `side` are used describe which face on the volume the
        modules are to be applied. `axis` is 0 for the x axis and 1 for the y axis.
        side is 0 for the minimum side of the given `axis` and 1 is for the maximum
        side.

        :return: The number of modules applied
        """
        modules_applied = 0
        gancio(
            volume,
            module,
            axis=axis,
            border1=side,
            border2=0,
        )

        start1 = int(offset[0] + module.dimensions[abs(1 - axis)] / 2)
        start2 = int(offset[1] + module.y_offset + module.dimensions[2] / 2)
        end1 = int(
            np.diff(get_min_max(volume.mesh, abs(1 - axis)))[0]
            - (int(offset[2] + module.dimensions[abs(1 - axis)] / 2))
        )
        end2 = int(
            volume.height - (int(offset[3] + module.dimensions[abs(1 - axis)] / 2))
        )
        h_step = volume.floor_height

        for x in range(start1, end1, int(x_step)):
            for h in range(start2, end2, int(h_step)):
                object_deleted = False
                if np.random.random() > apply_probability:
                    continue
                m = copy(module)
                position = np.array([0, 0, 0])
                position[abs(1 - axis)] = x
                position[2] = h

                m.position(position)

                for v in self.volumes:
                    if (v.name is not volume.name) & v.collide(m.mesh):
                        m.remove()
                        object_deleted = True
                        break
                if object_deleted:
                    continue

                if volume.collide_with_other_modules(m):
                    m.remove()
                    continue

                modules_applied += 1
        module.remove()
        return modules_applied

    def apply_materials(
        self, building_material: Material, roof_material: Material
    ) -> None:
        """
        Applies provided materials to the roof and building volumes
        """
        for v in self.volumes:
            v.apply(material=building_material)
        for r in self.roofs:
            r.apply(material=roof_material)

    @property
    @abstractmethod
    def wwr(self) -> float:
        raise NotImplementedError("A ComposedBuilding has no WWR property on its own")

    @abstractmethod
    def _generate(self) -> None:
        raise NotImplementedError(
            "A ComposedBuilding has no _generate method on its own"
        )

    @staticmethod
    def _remove_overlap_faces(volumes: list[Volume]) -> None:
        """
        Goes through a list of `volumes` and connects them together where possible.
        Additionally re-colors volumes when necessary.
        """
        if len(volumes) < 2:
            return

        i = 0
        while i < len(volumes):
            j = i + 1
            while j <= len(volumes):
                if mesh_boolean_union(volumes[i].mesh, volumes[j].mesh):
                    volumes.pop(j)
                    volumes[i].color_volume()
                    break
                j += 1
            i += 1
