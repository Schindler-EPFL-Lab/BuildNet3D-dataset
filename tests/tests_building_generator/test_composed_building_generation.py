import unittest
from unittest.mock import patch

from tests.tests_building_generator.test_helpers.test_composed_building import (
    MockComposedBuilding,
)


class TestComposedBuildingGeneration(unittest.TestCase):
    def test_if_template_no_generation(self) -> None:
        """
        This test is to determine if the `_generate()` method is called from the
        ComposedBuilding Class. If `template` is declared as `True` then the method
        should not be called as the purpose of a template building is that it is not
        generated in blender.
        """
        with patch(
            "tests.tests_building_generator.test_helpers.test_composed_building."
            + "MockComposedBuilding._generate"
        ) as mock_generate:
            _ = MockComposedBuilding(
                template=True,
            )
        mock_generate.assert_not_called()

    def test_if_not_template_generation(self) -> None:
        """
        This test is to determine if the `_generate()` method is called from the
        ComposedBuilding Class. If `template` is declared as `False` then the method
        should be called and a building should be generated in blender.
        """
        with patch(
            "tests.tests_building_generator.test_helpers.test_composed_building."
            + "MockComposedBuilding._generate"
        ) as mock_generate:
            _ = MockComposedBuilding(
                template=False,
            )
        mock_generate.assert_called_once()

    def test_if_template_cloned_generation(self) -> None:
        with patch(
            "tests.tests_building_generator.test_helpers.test_composed_building."
            + "MockComposedBuilding._generate"
        ) as mock_generate:
            building = MockComposedBuilding(
                template=True,
            )
            _ = building.clone()
        mock_generate.assert_called_once()
