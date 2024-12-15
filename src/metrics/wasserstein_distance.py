import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from scipy.stats import wasserstein_distance

from sdv.metadata.single_table import SingleTableMetadata

# Numerical, Minimize
class WassersteinDistance:
    @classmethod
    def compute(cls, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, metadata=None):

        if metadata is None:
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(real_data)

        continuous_columns = [
            col for col in metadata.columns if metadata.columns[col]['sdtype'] == 'numerical'
        ]

        if len(continuous_columns) == 0:
            return np.inf

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
    
    
    
