import bpy
import numpy as np


class SunLight:
    def __init__(self, name: str) -> None:
        self.light = bpy.data.lights.new(name=name, type="SUN")  # type: ignore
        self.light_object = bpy.data.objects.new(  # type: ignore
            name=name, object_data=self.light
        )

        bpy.context.collection.objects.link(self.light_object)  # type: ignore
        bpy.context.view_layer.objects.active = self.light_object  # type: ignore

    def change_orientation(self, new_orientation: np.ndarray) -> None:
        self.light_object.rotation_euler = np.radians(new_orientation)

    def change_energy(self, new_energy: float) -> None:
        self.light.energy = new_energy
