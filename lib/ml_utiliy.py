import numpy as np
import pandas as pd
from typing import Literal

from pathlib import Path
import json
import copy
from tqdm import tqdm

from lib.js_divergence import JSDivergence
from lib.wasserstein_distance import WassersteinDistance
from lib.propensity_score import PropensityScore
from lib.pcorr_difference import PairwiseCorrelationDifference

import numpy as np
import pandas as pd
from typing import Literal
# from sdv.single_table import CTGANSynthesizer
from ctgan import CTGAN
from sdv.metadata import SingleTableMetadata
from pathlib import Path
from src.metrics import *
from tqdm import tqdm
import json


DATA_FOLDER = Path('data')


def run_ml_utility_on_datasets(
    synthetic_data_folder: Path,
    real_data_folder: Path = DATA_FOLDER,
    datasets: list[str] = [],
    synhtesizer_name: Literal['CTGAN'] = 'CTGAN',
    is_optuna: bool = False
):
    for dataset in tqdm(datasets):
        tqdm.write(dataset)
        # Read the real data
        real_data = pd.read_csv(real_data_folder / dataset / 'data.csv')
        # Read the metada about the data
        with open(real_data_folder / dataset / 'metadata.json', 'r') as file:
            metadata = json.load(file)
        # Get the discrete and continuous columns
        
        discrete_columns = metadata['categorical_columns']
        continuous_columns = [col for col in real_data.columns if col not in discrete_columns]
        
        # Get the files of the synthetic data
        synthetic_data_files = (synthetic_data_folder / dataset / synhtesizer_name).glob('*.zip')
        # Evaluations for later population
        ml_utility_metrics = {}
        # Iterate over all the files
        for synthetic_file in synthetic_data_files:
            synthetic_data_name = synthetic_file.stem
            synthetic_data = pd.read_csv(synthetic_file)
            
            ml_utility_metrics[synthetic_data_name] = _evaluate_synthetic_data_ml_utility(
                real_data,
                synthetic_data,
                discrete_columns,
                continuous_columns,
            )
        
        # Iteration End Json Dump the statistical similarity values
        _save_ml_utility_evaluations(
            ml_utility_metrics,
            synthetic_data_folder / dataset / synhtesizer_name / 'ml_utility.json',
            is_optuna
        )
        
def _evaluate_synthetic_data_ml_utility(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    discrete_columns: list[str] = [],
    continuous_columns: list[str] | None  = None
) -> dict:
    pass
    # similarity_metrics ={
    #         'JSD':JSDivergence.compute(real_data, synthetic_data, discrete_columns, continuous_columns),
    #         'WD': WassersteinDistance.compute(real_data, synthetic_data, discrete_columns, continuous_columns),
    #         'PCD':PairwiseCorrelationDifference.compute(real_data, synthetic_data, discrete_columns, continuous_columns),
    #         'PS': PropensityScore.compute(real_data, synthetic_data, discrete_columns, continuous_columns),
    # }
    
    # return similarity_metrics




def _save_ml_utility_evaluations(
    statistical_similarity_metrics: dict,
    path: Path,
    is_optuna: bool,
):
    jsd_values = []
    wd_values = []
    pcd_values = []
    ps_values = []

    for experiment_name in statistical_similarity_metrics:
        jsd_values.append(statistical_similarity_metrics[experiment_name]['JSD'])
        wd_values.append(statistical_similarity_metrics[experiment_name]['WD'])
        pcd_values.append(statistical_similarity_metrics[experiment_name]['PCD'])
        ps_values.append(statistical_similarity_metrics[experiment_name]['PS'])
    
    # Deep copy to prevent changes to the original dictionary
    statistical_similarity_metrics_save = copy.deepcopy(statistical_similarity_metrics)
    
    if is_optuna == False:
        # Average all the test results
        statistical_similarity_metrics_save['Average Results'] = {
            'JSD':np.average(jsd_values),
            'WD': np.average(wd_values),
            'PCD':np.average(pcd_values),
            'PS': np.average(ps_values), 
        }
    else:
        # Get the best propensity score
        best_index = np.argmax(ps_values)

        statistical_similarity_metrics_save['Best Result'] = {
            'JSD':jsd_values[best_index],
            'WD': wd_values[best_index],
            'PCD':pcd_values[best_index],
            'PS': ps_values[best_index], 
        }  

    with open(path, 'w+') as file:
        json.dump(statistical_similarity_metrics_save, file, indent=4)
