from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from classification_model.config.core import config


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    validated_data = input_data.copy()
    new_vars_with_na = [
        var
        for var in config.model_config.features
        if validated_data[var].isnull().sum() > 0
        # and var not in config.model_config.categorical_vars_with_na_frequent
        
    ]
    validated_data.dropna(subset=new_vars_with_na, inplace=True)

    return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    # Add relevant replacement or fiel renaming
    relevant_data = input_data[config.model_config.features].copy()
    validated_data = relevant_data # drop_na_inputs(input_data=relevant_data)
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleTradingDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class TradingDataSchema(BaseModel):
    Open: Optional[float]
    Close: Optional[float]
    High: Optional[float]
    Low: Optional[float]
    Volume: Optional[int]


class MultipleTradingDataInputs(BaseModel):
    inputs: List[TradingDataSchema]
