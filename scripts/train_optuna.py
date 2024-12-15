from pathlib import Path

import argparse
import copy


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


# # Find a CTGAN hyper parameter optimizing github code and yoink the code from that

# dataset = fetch_openml('iris')
# X = dataset.data

# metadata = SingleTableMetadata()
# metadata.detect_from_dataframe(X)

# synthesizer = CTGANSynthesizer(metadata=metadata)
# synthesizer.fit(X)


# synthesizer.get_parameters()

# # This way you can get all the hyper parameters


def main():
    parser = argparse.ArgumentParser()

    # The dataset argument is optional. If it is not defined or it is omitted train on all the datasets
    parser.add_argument('-d', '--dataset',  type=str)
    parser.add_argument( '-m', '--model', type=str, required=True)
    
    args = parser.parse_args()
        
    if args.dataset is None or str.lower(args.dataset) == 'all':
        datasets = copy.deepcopy(DATASETS)
    else:
        datasets = [args.dataset]
        
    
    
    
    print(args)
    

    

if __name__ == '__main__':
    main()