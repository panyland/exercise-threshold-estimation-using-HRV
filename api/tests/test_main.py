import io
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def subject_csv_bytes():
    """Real beat-level data for one subject, trimmed to the (time, RR, power) columns the API expects."""
    data = pd.read_csv(REPO_ROOT / "data" / "test_measure.csv")
    subject = data[data["ID"] == 1][["time", "RR", "power"]]
    buf = io.BytesIO()
    subject.to_csv(buf, index=False)
    return buf.getvalue()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize("model", ["rf", "xgboost"])
def test_predict_valid_subject(subject_csv_bytes, model):
    response = client.post(
        "/predict",
        files={"file": ("subject1.csv", subject_csv_bytes, "text/csv")},
        data={"model": model},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["vt1_power_W"] is not None
    assert body["vt2_power_W"] is not None
    assert 0 < body["vt1_power_W"] < body["vt2_power_W"]
    assert 60 < body["vt1_hr_bpm"] < body["vt2_hr_bpm"] < 220


def test_predict_excel(subject_csv_bytes):
    df = pd.read_csv(io.BytesIO(subject_csv_bytes))
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    response = client.post(
        "/predict",
        files={
            "file": (
                "subject1.xlsx",
                buf.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
        data={"model": "rf"},
    )
    assert response.status_code == 200


def test_predict_missing_columns():
    bad = b"foo,bar\n1,2\n"
    response = client.post(
        "/predict",
        files={"file": ("bad.csv", bad, "text/csv")},
        data={"model": "rf"},
    )
    assert response.status_code == 400
    assert "missing columns" in response.json()["detail"].lower()


def test_predict_wrong_extension(subject_csv_bytes):
    response = client.post(
        "/predict",
        files={"file": ("subject1.txt", subject_csv_bytes, "text/plain")},
        data={"model": "rf"},
    )
    assert response.status_code == 400


def test_predict_invalid_model(subject_csv_bytes):
    response = client.post(
        "/predict",
        files={"file": ("subject1.csv", subject_csv_bytes, "text/csv")},
        data={"model": "not_a_model"},
    )
    assert response.status_code == 422
