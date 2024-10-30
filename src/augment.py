from sdv.tabular.base import BaseTabularModel
import numpy as np
import pandas as pd

def augment_data(
    dataset: np.ndarray,
    GAN: BaseTabularModel,
    n_augment: int | None = None,
    ratio_augmnet: float | None = None
) -> np.ndarray:
    
    if n_augment or ratio_augmnet:
        if n_augment:
            n_sample = n_augment
        else:
            n_sample = len(dataset) * ratio_augmnet
    else:
        raise ValueError('n_augment or ratio_augmnet should be given')
    
    synthetic_data = GAN.sample(n_sample)
    
    # CHECK
    augmented_data = np.concatenate([dataset, synthetic_data], axis=0)
    
    return augmented_data

# TODO Add utility for handling class imbalances