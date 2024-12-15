import numpy as np
import pandas as pd
import sdv
from src.metrics import MetricsRegistry

class MetricsEvaluation:
    def __init__(
        self,
        metrics: list[str] = ['KST', 'CJSD', 'WD', 'JSD', 'TVT'],
    ) -> None:

        self._metric_names = metrics
        self._metrics = [MetricsRegistry.get_metric_type(metric) for metric in metrics]
    
    def fit(
        self,
        dataset: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata: sdv.metadata.SingleTableMetadata
    ) -> 'MetricsEvaluation':
        self._evals = pd.Series(dtype=np.float64)
        for metric_name, metric in zip(self._metric_names, self._metrics):
            if metric is None:
                continue
            self._evals[metric_name] = metric.compute(dataset, synthetic_data, metadata)

        return self
            
    def get_evals(
        self
    ) -> pd.Series:
        return self._evals
