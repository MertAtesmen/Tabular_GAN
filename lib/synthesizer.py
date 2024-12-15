import numpy as np
import pandas as pd

from typing import Literal
from ctgan import CTGAN
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

from abc import ABC, abstractmethod


###!!!
#
#
# The idea behind the wrapper classes is to instantiate this classes easily given that hyper paramteres and metadata.
# It can be flexible since all i need is to get a different hyperparameter or metadata about a class 
#
#
###!!!

# TODO Update this

def get_gan_model(
    model_name: Literal['CTGAN', 'CTGANSynthesizer'],
    metadata: dict,
    hyper_params: dict,
    verbose: bool = False
):
    if model_name == 'CTGAN':
        return CTGANWrapper(metadata, hyper_params, verbose)
    else:
        return CTGANSynthesizerWrapper(metadata, hyper_params, verbose)
    


class SynhesizerBase(ABC):
    """Base wrapper class for GAN models."""
    @abstractmethod
    def fit(self, data: pd.DataFrame):
        raise NotImplementedError("Fit method must be implemented.")

    @abstractmethod
    def sample(self, n: int):
        raise NotImplementedError("Sample method must be implemented.")

    @abstractmethod
    def sample_conditions(self,  conditions: list[dict]):
        raise NotImplementedError("Sample Conditions method must be implemented.")


   
class CTGANSynthesizerWrapper(SynhesizerBase):
    def __init__(
        self,
        metadata: dict,
        hyper_params: dict,
        verbose: bool = False
    ):
        ## TODO More work to do here
        self.metadata = metadata
        self.hyper_params = hyper_params
        self.verbose = verbose
    
    def fit(self, data: pd.DataFrame):
        single_table_metadata = SingleTableMetadata()
        single_table_metadata.detect_from_dataframe(data)
        self.model = CTGANSynthesizer(metadata=single_table_metadata, **self.hyper_params, verbose=self.verbose)

        self.model.fit(data)

    def sample(self, n: int):

        return self.model.sample(n)

    def sample_conditions(self, conditions: list[dict]):
        
        synthetic_data_list = []
        for condition in conditions:
            synthetic_data_list.append(self.model.sample(
                n=condition['n'],
                condition_column=condition['condition_column'],
                condition_value=condition['condition_value']
            ))
            
        synthetic_data = pd.concat(synthetic_data_list, ignore_index=True)
        return synthetic_data


class CTGANWrapper(SynhesizerBase):
    def __init__(
        self,
        metadata: dict,
        hyper_params: dict,
        verbose: bool = False
    ):
        ## TODO More work to do here
        
        self.model = CTGAN(**hyper_params, verbose=verbose)
        self.metadata = metadata
        self.hyper_params = hyper_params
    
    def fit(self, data: pd.DataFrame):
        self.model.fit(data)

    def sample(self, n: int):
        
        return self.model.sample(n)

    def sample_conditions(self, conditions: list[dict]):
        
        synthetic_data_list = []
        for condition in conditions:
            synthetic_data_list.append(self.model.sample(
                n=condition['n'],
                condition_column=condition['condition_column'],
                condition_value=condition['condition_value']
            ))
            
        synthetic_data = pd.concat(synthetic_data_list, ignore_index=True)
        return synthetic_data
