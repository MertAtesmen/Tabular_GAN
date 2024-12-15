import pandas as pd
import numpy as np

from sdv.metadata.single_table import SingleTableMetadata


# Minimize
class PairwiseCorrelationDifference:
    @classmethod
    def compute(cls, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, metadata=None):
        if metadata is None:
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(real_data)
        
        continuous_columns = [col for col in metadata.columns if metadata.columns[col]['sdtype'] == 'numerical']
        categorical_columns = [col for col in metadata.columns if metadata.columns[col]['sdtype'] == 'categorical']
        
        real_data_corr = real_data[continuous_columns].corr().to_numpy()
        synthetic_data_corr = synthetic_data[continuous_columns].corr().to_numpy()

        pcorr_difference = np.linalg.norm(real_data_corr - synthetic_data_corr)
        return pcorr_difference