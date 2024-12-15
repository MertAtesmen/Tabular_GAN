import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from scipy.stats import wasserstein_distance

# Numerical, Minimize
class WassersteinDistance:
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

        if len(continuous_columns) == 0:
            return 0

        wd_values = []

        for col in continuous_columns:

            scaler = MinMaxScaler()

            real_data_col = real_data[col].dropna()
            synthetic_data_col = synthetic_data[col].dropna()

            scaler.fit(real_data_col.values.reshape(-1,1))
            l1 = scaler.transform(real_data_col.values.reshape(-1,1)).flatten()
            l2 = scaler.transform(synthetic_data_col.values.reshape(-1,1)).flatten()
            wd_values.append(wasserstein_distance(l1,l2))

        return np.mean(wd_values)
    
    
    
