from pathlib import Path

import argparse
import copy


# S
# I should just store all the optuna search datasets and nothing more

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
OPTUNA_DATA_FOLDER = Path('data_optuna/')
SYNTHETIC_DATA_FOLDER = Path('data_synthetic/')

# Mostly done

def main():
    parser = argparse.ArgumentParser()

    # The dataset argument is optional. If it is not defined or it is omitted train on all the datasets
    parser.add_argument('-d', '--dataset',  type=str, help='Dataset Name, All datasets are chosen if NULL')
    parser.add_argument('-m', '--model', type=str, required=True, help='Name of the Model')
    parser.add_argument('--optuna', action='store_true', default=False, help='Choose this to evaluate optuna results')
    args = parser.parse_args()
        
    if args.dataset is None or str.lower(args.dataset) == 'all':
        datasets = copy.deepcopy(DATASETS)
    else:
        datasets = [args.dataset]

    # Import here to avoid unnecassary wait
    from lib.statistical_similarity import run_statistical_similarity_on_datasets

    run_statistical_similarity_on_datasets(
        SYNTHETIC_DATA_FOLDER,
        DATA_FOLDER,
        datasets=datasets,
        synhtesizer_name = 'CTGAN',
        is_optuna=args.optuna
    )

    

if __name__ == '__main__':
    main()