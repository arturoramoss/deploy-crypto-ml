import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

def test_make_prediction(client: TestClient, test_data: pd.DataFrame) -> None:
    # Given
    payload = {
        # ensure pydantic plays well with np.nan
        "inputs": test_data.replace({np.nan: None}).to_dict(orient="records")
    }

    # When
    response = client.post(
        "http://localhost:8001/api/v1/predict",
        json=payload,
    )

    # Then
    assert response.status_code == 200
    prediction_data = response.json()
    assert prediction_data["predictions"]
    assert prediction_data["errors"] is None
    assert set(prediction_data) <= set([True, False])


def test_health(client: TestClient) -> None:
    # Given

    # When
    response = client.get("http://localhost:8001/api/v1/health")

    # Then
    assert response.status_code == 200
    res_data = response.json()
    assert "name" in res_data.keys()
    assert "api_version" in res_data.keys()
    assert "model_version" in res_data.keys()
