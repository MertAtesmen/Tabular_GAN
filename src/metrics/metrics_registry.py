import scipy.spatial
import scipy.stats
from sdmetrics.single_table import (
    KSComplement,
    TVComplement,
    ContinuousKLDivergence, 
    GMLogLikelihood, 
    DiscreteKLDivergence, 
    CSTest, 
    CorrelationSimilarity, 
    LogisticDetection,
    # SVCDetection,
)

from src.metrics.js_divergence import ContinuousJSComplement, JSComplement
from src.metrics.wasserstein_distance import WassersteinDistance

import scipy

class MetricsRegistry:
    _metrics_lookup = {
        # Kolmogorov Smirnov test, for continous features
        "KST": KSComplement,
        # Distribution test for discrete features
        'TVT': TVComplement,
        "CKL": ContinuousKLDivergence,
        "DKL": DiscreteKLDivergence,
        "GMLL": GMLogLikelihood,
        "CS": CSTest,
        "C": CorrelationSimilarity,
        "LRD": LogisticDetection,
        'JSD': JSComplement,
        'CJSD': ContinuousJSComplement,
        'WD': WassersteinDistance,
        # "SVCD": SVCDetection
    }

    @staticmethod
    def get_metric_type(type_name):
        """Get the Python type for a given type name."""
        return MetricsRegistry._metrics_lookup.get(type_name)

    @staticmethod
    def register_type(type_name, python_type):
        """Register a new type name and its corresponding Python type."""
        MetricsRegistry._metrics_lookup[type_name] = python_type