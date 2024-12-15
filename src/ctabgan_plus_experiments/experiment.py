import numpy as np
import pandas as pd
from typing import Literal
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from pathlib import Path
from src.metrics import *
from tqdm import tqdm
import json


def run_ctabgan_plus_experiment(
    data_folder: Path,
    synthesizer: Literal['CTGAN'] = 'CTGAN',
    n_experiments: int = 3,
    no_train = False,
) -> pd.DataFrame:
    
    # Read the dataset
    dataset = pd.read_csv(data_folder / 'data.csv')
    
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(dataset)
    
    metadata.save_to_json(data_folder / 'metadata.json')
    
    # Get the train test indices
    train_idx = np.load(data_folder / 'train_idx.npy')
    test_idx = np.load(data_folder / 'test_idx.npy')
    
    # Train test Split
    train, test = dataset.iloc[train_idx, :], dataset.iloc[test_idx, :]
    
    # Evaluation for the experiments
    evaluations = pd.DataFrame()
    
    with open(data_folder / 'train_config.json', 'r') as file:
        config = json.load(file)
    
    # n Experiments
    for i in tqdm(range(1, n_experiments + 1)):
        # If no train get the already synthesized data 
        if no_train:
            synthetic_data = pd.read_csv(data_folder / synthesizer / f'experiment_{i}.csv')
        else: 
            synthesizer_model = train_synthesizer_model(train, synthesizer, hyper_params={'epochs': config['epochs']})
            synthetic_data = synthesizer_model.sample(len(dataset))
            (data_folder / synthesizer).mkdir(parents=True, exist_ok=True)
            synthetic_data.to_csv(data_folder / synthesizer / f'experiment_{i}.csv')
        
        # Evaluate the data
        eval = evaluate_synthetic_data_similarity(dataset, synthetic_data)
        evaluations= pd.concat([evaluations, eval], ignore_index=True)
    
    # Save evaluations
    evaluations.to_csv(data_folder / synthesizer / 'evaluations.csv', index=False)
    
    return evaluations


def train_synthesizer_model(
    dataset: pd.DataFrame,
    synthesizer: Literal['CTGAN'] = 'CTGAN',
    hyper_params: dict = {},
    metadata = None
):
    if metadata == None:
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(dataset)
    
    if synthesizer == 'CTGAN':
        synthesizer_model = CTGANSynthesizer(metadata, **hyper_params, verbose=True)
    
    synthesizer_model.fit(dataset)
    
    return synthesizer_model


        
def evaluate_synthetic_data_similarity(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    metadata = None,
) -> pd.DataFrame:
    similarity_metrics = pd.DataFrame(
        {
            'JSD': [JSDivergence.compute(real_data, synthetic_data, metadata)],
            'WD': [WassersteinDistance.compute(real_data, synthetic_data, metadata)],
            'PCD': [PairwiseCorrelationDifference.compute(real_data, synthetic_data, metadata)],
            'PS': [PropensityScore.compute(real_data, synthetic_data, metadata)],
        }
    )
    
    return similarity_metrics
