from fastapi.testclient import TestClient


def test_training_dry_run_does_not_download_model(app_client: TestClient) -> None:
    response = app_client.post("/api/v1/training/run", json={"dry_run": True})
    assert response.status_code == 202
    run_id = response.json()["id"]
    status = app_client.get(f"/api/v1/training/runs/{run_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "completed"
    assert status.json()["dataset_version"].startswith("dataset-")
    assert app_client.get("/api/v1/models").json() == []
