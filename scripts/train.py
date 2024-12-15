from pathlib import Path

import argparse
import copy

# CTGAN Stnhtesizr Train untl Loan


DATASETS = [
    'Adult',
    'Covertype',
    'Credit',
    'Intrusion',
    'Loan',
    'King',
    'Insurance'
]

DATA_FOLDER = Path('data/')
SYNTHETIC_DATA_FOLDER = Path('data_synthetic/')

def main():
    parser = argparse.ArgumentParser()

    # The dataset argument is optional. If it is not defined or it is omitted train on all the datasets
    parser.add_argument('-d', '--dataset',  type=str)
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-n', '--n_experiments', type=int, default=3)
    
    args = parser.parse_args()
        
    if args.dataset is None or str.lower(args.dataset) == 'all':
        # Since i've already trained all the CTGAN models i dont want any other trainings on that models 
        if str.lower(args.model) == 'ctgan':
            print('\n\nDo not train all the models with CTGAN again!!!!\n\n')
            exit(-1)
            
        datasets = copy.deepcopy(DATASETS)
    else:
        datasets = [args.dataset]
    
    # Save time
    from lib.synhtesizer_train import run_train_experiment
     
    run_train_experiment(
        synthetic_data_folder=SYNTHETIC_DATA_FOLDER,
        real_data_folder=DATA_FOLDER,
        datasets=datasets,
        synhtesizer_name=args.model,
        # n_experiments=3
    )


if __name__ == '__main__':
    main()
    
    
    