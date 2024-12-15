import pandas as pd
import numpy as np
import sdv
import warnings
from pathlib import Path

from src.metrics.metrics_registry import MetricsRegistry
from src.synthesizers.synthesizers_registry import SynthesizersRegistry
from src.evaluation.metrics_evaluation import MetricsEvaluation
from src.synthesizers.save_model import save_model, SAVING_FOLDER
   
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from src.schemas.synthesizer_hyper_parameters import SynthesizerHyperParameters

warnings.simplefilter("ignore", FutureWarning)

# If save model kwargs will include
#   dataset_name: str|None = None
#   folder_path: str|Path|None = None 
class SynthesizerEvaluation:

    def __init__(
        self,
        synthesizer: str = 'CTGAN',
        param_grid:  dict = {},
        *,
        metrics: list[str] = ['KST', 'CJSD', 'WD', 'JSD', 'TVT'],
        n_samples = 5,
        verbose = False,
        save_synthesizers = False,
        **kwargs
    ) -> None:
        self._synthesizer_name = synthesizer
        self._synthesizer_type = SynthesizersRegistry.get_synthesizer_type(synthesizer)

        if self._synthesizer_type is None:
            raise ValueError('Synthetsizer is not registered')

        self._metrics = metrics
        self._param_grid = param_grid
        self._verbose = verbose
        self._n_samples = n_samples
        self._save = save_synthesizers

        # Save config        
        if self._save:            
            if 'dataset_name' not in kwargs:
                raise ValueError('You must pass the dataset_name')
            
            self._dataset_name = kwargs['dataset_name']
            
            if not isinstance(self._dataset_name, str):
                raise ValueError('The dataset_name must be string')
            
            if 'folder_path' in kwargs:
                _folder_path = kwargs['folder_path']
                if _folder_path is None:
                    self._folder_path = SAVING_FOLDER
                else:
                    self._folder_path = Path(_folder_path)   
            else:
                self._folder_path = SAVING_FOLDER 
                
    
    def fit(
        self,
        dataset: pd.DataFrame,            
        metadata: sdv.metadata.SingleTableMetadata
    ) -> 'SynthesizerEvaluation':
        
        evaluations = pd.DataFrame()
        losses = []
        
        if self._save:
            saved_paths = []

        for params in ParameterGrid(self._param_grid):

            if 'lr' in params.keys():
                params['generator_lr'] = params['lr']
                params['discriminator_lr'] = params['lr']
                params.pop('lr')
                
            if self._verbose:
                print(f'Fitting Hyperparameters: {params}')

            hyper_param_schema = SynthesizerHyperParameters(**params)
            # Fit the synthesizer
            synhtesizer = self._synthesizer_type(metadata=metadata, verbose=self._verbose, **hyper_param_schema.model_dump())
            synhtesizer.fit(dataset)
            
            # Store the loss values
            loss_values = synhtesizer.get_loss_values().copy(deep=True)[['Generator Loss', 'Discriminator Loss']]
            losses.append(loss_values)
            
            metrics_list = []
            metrics_evaluation = MetricsEvaluation(self._metrics)
            
            # Sample n times
            for _ in range(self._n_samples):
                # Generate synthteic data
                synthetic_data = synhtesizer.sample(len(dataset))
            
                # Evaluate the metrics
                metrics_list.append(metrics_evaluation.fit(dataset, synthetic_data, metadata).get_evals())

            # Calculate the avarage of metrics
            metrics = sum(metrics_list) / self._n_samples

            # Concatanate hyper parameters and metrics
            
            
            hyper_params = pd.Series(hyper_param_schema.model_dump())
            evaluations_row = pd.concat([hyper_params, metrics])

            # Add the row to the evaluations
            evaluations= pd.concat([evaluations, evaluations_row.to_frame().T], ignore_index=True)
            
            # Save the model if save parameter is true
            if self._save:
                saved_paths.append(save_model(synhtesizer, self._synthesizer_name, self._dataset_name, None, self._folder_path))

        evaluations['losses'] = losses
        
        if self._save:
            evaluations['paths'] = saved_paths
        
        self._evals = evaluations.convert_dtypes(infer_objects=True)
        
        if self._save:
            self._save_evals()
        
        return self
       
    
    def get_evals(
        self
    ) -> pd.Series:
        return self._evals

    def _save_evals(
        self
    ) -> None:
        synthesizer_folder_path = Path(self._folder_path) / self._dataset_name / self._synthesizer_name
        evals_path = synthesizer_folder_path / 'evals.json'
        
        self._evals.to_json(evals_path, orient="records", indent=4)