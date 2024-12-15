import numpy as np
import pandas as pd

import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler


def evaluate_binary_classification_stratifedkfold(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    classifier: sklearn.base.ClassifierMixin,
    transformers: sklearn.base.TransformerMixin | None = None,
    n_fold = 5,
    shuffle = True,
    random_state=42,
) -> tuple[tuple[float], list[sklearn.base.ClassifierMixin]]:

    f1_scores = []
    recall_scores = []
    precision_scores = []
    classifiers = []

    scaler = StandardScaler()
    scaler.set_output(transform='pandas')
    
    kf = StratifiedKFold(n_splits=n_fold, shuffle=shuffle, random_state=random_state if shuffle else None)
    
    for i, (train_indices, test_indices) in enumerate(kf.split(X, y)):
        X_train, X_test, y_train, y_test = X.iloc[train_indices], X.iloc[test_indices], \
                                       y[train_indices], y[test_indices] 
                                       
                                       
        if transformers is not None:
            X_train = transformers.fit_transform(X_train)
            X_test = transformers.transform(X_test)
        
        classifiers.append(sklearn.base.clone(classifier))
        classifiers[i].fit(X_train, y_train)

        predictions = classifiers[i].predict(X_test)
        
        f1_scores.append(f1_score(y_test, predictions, average='binary'))
        precision_scores.append(precision_score(y_test, predictions, average='binary'))
        recall_scores.append(recall_score(y_test, predictions, average='binary'))
        
    return (
        (np.mean(f1_scores), np.mean(precision_scores), np.mean(recall_scores)),
        classifiers
    )