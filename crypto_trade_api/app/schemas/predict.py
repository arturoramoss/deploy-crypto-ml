from typing import Any, List, Optional

# from classification_model.processing.validation import TradingDataSchema
from pydantic import BaseModel


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[bool]]


# class MultipleTradingDataInputs(BaseModel):
#     inputs: List[TradingDataSchema]

#     class Config:
#         schema_extra = {
#             "example": {
#                 "inputs": [
#           {"Open": 10, "Close": 11.5, "High": 13.5, "Low": 9.5, "Volume": 100}
#                 ]
#             }
#         }
