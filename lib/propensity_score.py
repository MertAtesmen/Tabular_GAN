import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Minimize
class PropensityScore:
    @classmethod
    def compute(
        cls,     
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        discrete_columns: list[str] = [],
        continuous_columns: list[str] | None  = None
    ) -> float:

        # Transform the data
        real_data_transformed, synthetic_data_transformed = _transform_data(
            real_data, 
            synthetic_data, 
            discrete_columns, 
            continuous_columns
        )
        
        n_real = len(real_data)
        n_synhtetic = len(synthetic_data)
        
        # Generate the training set and its labels
        X = pd.concat([real_data_transformed, synthetic_data_transformed], ignore_index=True)
        y = np.concatenate([np.ones(shape=(n_real,)), np.zeros(shape=(n_synhtetic,))])
        
        # Create the classifier model
        # TODO: Change the classiffier
        classifier = LogisticRegression()
        
        # Fit the model and extract probabilites
        classifier.fit(X, y)
        prob_predictions = classifier.predict_proba(X)[:, 1]
        
        # Propensity_Score(X) = 1/n sum_n (pi - 0.5)^2
        propensity_score = mean_squared_error(prob_predictions, np.full(shape=(len(prob_predictions),), fill_value=0.5))

        return propensity_score



def _transform_data(
    real_data: pd.DataFrame, 
    synthetic_data: pd.DataFrame,
    discrete_columns: list[str] = [],
    continuous_columns: list[str] | None  = None
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if continuous_columns is None:
        continuous_columns = [col for col in real_data.columns if col not in discrete_columns]

    numerical_pipeline = Pipeline(
        steps=[
            ('impute', SimpleImputer(strategy='mean')),
            ('preprocessing', RobustScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('preprocessing', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]
    )
    
    transformer = ColumnTransformer(
        transformers=[
            ('numerical', numerical_pipeline, continuous_columns),
            ('categorical', categorical_pipeline, discrete_columns),
        ],
        remainder='drop',
        sparse_threshold=0,
    )
    
    transformer.set_output(transform='pandas')

    real_data_transformed = transformer.fit_transform(real_data)
    synthetic_data_transformed = transformer.transform(synthetic_data)
    
    return real_data_transformed, synthetic_data_transformed