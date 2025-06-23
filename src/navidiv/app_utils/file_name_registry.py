
class FileNameRegistry:
    """Registry for mapping internal file/column names to user-friendly display names in the app."""

    def __init__(self):
        # _name_map is a private dictionary for name mapping
        self._name_map = {}

    def register(self, internal_name: str, display_name: str) -> None:
        """Register a mapping from an internal name to a display name."""
        self._name_map[internal_name] = display_name

    def get_display_name(self, internal_name: str) -> str:
        """Get the display name for an internal name. Returns the internal name if not registered."""
        return self._name_map.get(internal_name, internal_name)

    def unregister(self, internal_name: str) -> None:
        """Remove a mapping from the registry."""
        if internal_name in self._name_map:
            del self._name_map[internal_name]

    def as_dict(self) -> dict:
        """Return the current mapping as a dictionary."""
        return dict(self._name_map)


# Example usage:
# registry = FileNameRegistry()
# registry.register('smiles', 'SMILES String')
# display = registry.get_display_name('smiles')
def initiate_file_name_registry():
    """Initialize the file name registry with default mappings."""
    registry = FileNameRegistry()
    registry.register("smiles", "SMILES String")
    registry.register("step", "Generation Step")
    registry.register("score", "Score")
    registry.register("molecule", "Molecule")
    registry.register("index", "ID")
    registry.register("tsne_x", "t-SNE X")
    registry.register("tsne_y", "t-SNE Y")
    registry.register("TSNEx", "t-SNE X")
    registry.register("TSNEy", "t-SNE Y")
    registry.register("scorer_name", "Scorer Name")
    registry.register("scorer_value", "Scorer Value")
    registry.register("scorer_properties", "Scorer Properties")
    registry.register("scorer_selection_criteria", "Selection Criteria")
    registry.register("scorer_output", "Scorer Output")
    registry.register("step min", "First Occurance (Step)")
    registry.register(
        "step count", "Steps with molecules containing fragment"
    )
    registry.register("functional_groups", "Functional Groups")
    registry.register("fragments_none", "Fragments with no Transformation")
    registry.register(
        "fragments_basic_wire_frame", "Fragments with Basic Wireframe"
    )
    registry.register(
        "fragments_elemental_wire_frame", "Fragments with Elemental Wireframe"
    )
    registry.register(
        "fragments_basic_framework", "Fragments with Basic Framework"
    )
    registry.register(
        "scaffold_basic_wire_frame", "Scaffold with Basic Wireframe"
    )
    registry.register(
        "scaffold_elemental_wire_frame", "Scaffold with Elemental Wireframe"
    )
    registry.register(
        "scaffold_basic_framework", "Scaffold with Basic Framework"
    )
    registry.register(
        "scaffold_elemental_framework", "Scaffold with Elemental Framework"
    )
    registry.register(
        "Mean diff score", "Score difference for molcules containing fragment"
    )
    registry.register(
        "Number of Molecules_with_Fragment", "Molecules with Fragment"
    )
    registry.register(
        "Mean diff score", "Score difference for molcules containing fragment"
    )
    registry.register(
        "step_list", "Generation Step"
    )
    # Add more mappings as needed
    return registry
