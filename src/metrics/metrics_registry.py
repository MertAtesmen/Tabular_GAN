from src.metrics.js_divergence import ContinuousJSDivergence, JSDivergence
from src.metrics.wasserstein_distance import WassersteinDistance
from src.metrics.propensity_score import PropensityScore
from src.metrics.pcorr_difference import PairwiseCorrelationDifference

class MetricsRegistry:
    _metrics_lookup = {
        'C': PairwiseCorrelationDifference,
        'P': PropensityScore,
        'JSD': JSDivergence,
        'CJSD': ContinuousJSDivergence,
        'WD': WassersteinDistance,
    }

    @staticmethod
    def get_metric_type(type_name):
        """Get the Python type for a given type name."""
        return MetricsRegistry._metrics_lookup.get(type_name)

    @staticmethod
    def register_type(type_name, python_type):
        """Register a new type name and its corresponding Python type."""
        MetricsRegistry._metrics_lookup[type_name] = python_type