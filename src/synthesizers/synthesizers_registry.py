from sdv.single_table import CTGANSynthesizer

class SynthesizersRegistry:
    _synthesizers_lookup = {
        "CTGAN": CTGANSynthesizer
    }

    @staticmethod
    def get_synthesizer_type(type_name):
        """Get the Python type for a given type name."""
        return SynthesizersRegistry._synthesizers_lookup.get(type_name)

    @staticmethod
    def register_type(type_name, python_type):
        """Register a new type name and its corresponding Python type."""
        SynthesizersRegistry._synthesizers_lookup[type_name] = python_type