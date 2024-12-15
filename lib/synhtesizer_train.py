import numpy as np
import pandas as pd
from typing import Literal

from pathlib import Path
import json
from tqdm import tqdm

from lib.synthesizer import get_gan_model

DATA_FOLDER = Path('data')

def run_train_experiment(
    synthetic_data_folder: Path,
    real_data_folder: Path = DATA_FOLDER,
    datasets: list[str] = [],
    synhtesizer_name: Literal['CTGAN', 'CTGANSynthesizer'] = 'CTGAN',
    n_experiments: int = 3
):
    for dataset in tqdm(datasets):
        tqdm.write(dataset)
        # Read the real data
        real_data = pd.read_csv(real_data_folder / dataset / 'data.csv')
        
        # Load the train indices and partition
        train_idx = np.load(real_data_folder / dataset / 'train_idx.npy')
        train= real_data.iloc[train_idx, :]
    
        # Read the metadata about the data
        with open(real_data_folder / dataset / 'metadata.json', 'r') as file:
            metadata = json.load(file)
            print(metadata)
        # Read the train config about the data
        with open(real_data_folder / dataset / 'train_config.json', 'r') as file:
            train_config = json.load(file)

        print(train_config)
        for nth_exp in range(1, n_experiments + 1):
            # Instantiate GAN Model
            synthesizer = get_gan_model(synhtesizer_name, metadata, hyper_params={'epochs': train_config['epochs']}, verbose=True)
            # Fit the model to the train set
            synthesizer.fit(train)
            # Sample data from the model
            synthetic_data = synthesizer.sample(len(real_data))
            # Generate folders if not exist
            (synthetic_data_folder / dataset /synhtesizer_name).mkdir(parents=True, exist_ok=True)
            # Save the synhtetic dataset
            synthetic_data.to_csv(
                synthetic_data_folder / dataset / synhtesizer_name / f'experiment_{nth_exp}.zip',
                index=False, 
                compression='zip'
            )