from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names = ['year', 'month', 'day', 'day_of_week']
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: pd.DataFrame):
        X_copy = X.copy(deep=True)
        
        for col in X_copy.columns:
            
            X_copy[col] = pd.to_datetime(X_copy[col])
            # Extract features  
            X_copy[f'{col}_year'] = X_copy[col].dt.year
            X_copy[f'{col}_month'] = X_copy[col].dt.month
            X_copy[f'{col}_day'] = X_copy[col].dt.day
            X_copy[f'{col}_day_of_week'] = X_copy[col].dt.dayofweek
            
            #Drop the original columns
            X_copy.drop(columns=[col], inplace=True)
            
        return X_copy
    
    def set_output(self, *args, **kwargs) -> BaseEstimator:
        pass