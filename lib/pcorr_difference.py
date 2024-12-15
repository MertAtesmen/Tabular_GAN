import pandas as pd
import numpy as np

# Minimize
class PairwiseCorrelationDifference:
    @classmethod
    def compute(
        cls,     
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        discrete_columns: list[str] = [],
        continuous_columns: list[str] | None  = None
    ) -> float:
        
        if continuous_columns is None:
            continuous_columns = [col for col in real_data.columns if col not in discrete_columns]
        
        real_data_corr = real_data[continuous_columns].corr().to_numpy()
        synthetic_data_corr = synthetic_data[continuous_columns].corr().to_numpy()

        pcorr_difference = np.linalg.norm(real_data_corr - synthetic_data_corr)
        return pcorr_difference