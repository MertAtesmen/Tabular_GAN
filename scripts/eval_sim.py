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
    parser.add_argument('--no_compute', action='store_true', default=False, help='Chose this to only produce report')
    parser.add_argument('-o', type=str, default='.', help='Redirect the output')
    
    args = parser.parse_args()
        
    if args.dataset is None or str.lower(args.dataset) == 'all':
        datasets = copy.deepcopy(DATASETS)
    else:
        datasets = [args.dataset]

    # Import here to avoid unnecassary wait
    from lib.statistical_similarity import run_statistical_similarity_on_datasets
    
    print(args)

    run_statistical_similarity_on_datasets(
        SYNTHETIC_DATA_FOLDER,
        DATA_FOLDER,
        datasets=datasets,
        synhtesizer_name = args.model,
        is_optuna=args.optuna,
        no_compute=args.no_compute,
        outputh_path=args.o,
    )
    

    

if __name__ == '__main__':
    main()