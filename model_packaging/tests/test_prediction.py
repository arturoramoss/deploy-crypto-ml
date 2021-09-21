# import math

# import numpy as np

from classification_model.predict import make_prediction


def test_make_prediction(sample_input_data):
    # Given
    expected_predictions_value = set([True, False])
    expected_no_predictions = 10

    # When
    result = make_prediction(input_data=sample_input_data)

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], bool)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    assert set(predictions) <= expected_predictions_value
