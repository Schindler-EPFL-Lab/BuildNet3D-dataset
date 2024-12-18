import bpy
import numpy as np

from buildnet3d_dataset.light_manager.sun_light import SunLight


class UniformLight:
    def __init__(self) -> None:
        bpy.data.lights["Sun"].energy = 0  # type: ignore

        self.lights = [SunLight("Spot" + str(i)) for i in range(4)]

    def make(self) -> None:
        orientation = np.zeros(3)
        orientation[0:2] = np.random.randint(40, 60, size=2)
        orientation[2] = np.random.randint(0, 90)
        energy = np.random.randint(4, 8)

        for i, light in enumerate(self.lights):
            light.change_orientation(orientation + np.array([0, 0, i * 90]))
            light.change_energy(energy)

        self.lights[0].change_energy(energy + 10)
