import numpy as np
import pandas as pd

from scipy.spatial.distance import jensenshannon

# Categorical, Minimize
class JSDivergence:
    @classmethod
    def compute(
        cls,     
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        discrete_columns: list[str] = [],
        continuous_columns: list[str] | None  = None
    ) -> float:        

        if len(discrete_columns) == 0:
            return 0
        
        js_divergence_values = []

        for col in discrete_columns:
            real_data_col = real_data[col].dropna()
            synthetic_data_col = synthetic_data[col].dropna()

            concatenated_columns: pd.Series = pd.concat([real_data_col, synthetic_data_col])
            unique_values = concatenated_columns.unique()

            real_distrubition = real_data_col.value_counts(normalize=True).reindex(unique_values, fill_value = 0).values
            synthetic_distrubition = synthetic_data_col.value_counts(normalize=True).reindex(unique_values, fill_value = 0).values

            js_divergence_values.append(jensenshannon(real_distrubition, synthetic_distrubition))

        return np.mean(js_divergence_values)
    

# Numerical, Minimize
class ContinuousJSDivergence:
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
        
        js_divergence_values = []

        for col in continuous_columns:
            real_data_col = real_data[col].dropna()
            synthetic_data_col = synthetic_data[col].dropna()
            
            concatenated_columns: pd.Series = pd.concat([real_data_col, synthetic_data_col])
            bin_edges = np.histogram_bin_edges(concatenated_columns, bins=20)
            
            counts_real, _ = np.histogram(real_data_col, bins=bin_edges, density=True)
            counts_synth, _ = np.histogram(synthetic_data_col, bins=bin_edges, density=True)
            
            real_distrubition = counts_real / counts_real.sum()
            synthetic_distrubition = counts_synth / counts_synth.sum()
            
            js_divergence_values.append(jensenshannon(real_distrubition, synthetic_distrubition))
        
        return np.mean(js_divergence_values)
    
    
    