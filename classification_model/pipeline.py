# from feature_engine.encoding import OrdinalEncoder, RareLabelEncoder
# from feature_engine.imputation import (
#     AddMissingIndicator,
#     CategoricalImputer,
#     MeanMedianImputer,
# )
# from classification_model.processing import features as pp
# from feature_engine.selection import DropFeatures
# from feature_engine.transformation import LogTransformer
# from feature_engine.wrappers import SklearnTransformerWrapper
# from sklearn.preprocessing import Binarizer

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classification_model.config.core import config


trade_pipe = Pipeline([

    # TODO Add pre-processing steps


    ('scaler', StandardScaler()),

    ('Logistic Regression', 
    LogisticRegression(
        C = config.model_config.C,
        class_weight = config.model_config.class_weight, 
        random_state = config.model_config.random_state
    )),
])
