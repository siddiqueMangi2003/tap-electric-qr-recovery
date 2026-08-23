from datetime import UTC, datetime

from fastapi.testclient import TestClient


def _scan_form(**overrides: object) -> dict[str, str]:
    values: dict[str, str] = {
        "latitude": "52.3676",
        "longitude": "4.9041",
        "timestamp": datetime.now(UTC).isoformat(),
        "native_qr_success": "true",
        "native_qr_result": "https://tap-electric.com/c/NL-TAP-E12345",
        "client_scan_id": "mobile-scan-001",
    }
    values.update({key: str(value) for key, value in overrides.items()})
    return values


def test_scan_is_stored_processed_and_confirmed(app_client: TestClient, png_bytes: bytes) -> None:
    response = app_client.post(
        "/api/v1/scans",
        data=_scan_form(),
        files={"image": ("sticker.png", png_bytes, "image/png")},
    )
    assert response.status_code == 202
    accepted = response.json()
    assert accepted["image_uri"].startswith("local://")

    scan = app_client.get(f"/api/v1/scans/{accepted['scan_id']}")
    assert scan.status_code == 200
    body = scan.json()
    assert body["status"] == "completed"
    assert body["final_prediction"] == "https://tap-electric.com/c/NL-TAP-E12345"
    assert body["resolved_charger_id"] == "NL-TAP-E12345"
    assert body["prediction_source"] == "native_qr"

    confirmation = app_client.post(
        f"/api/v1/scans/{accepted['scan_id']}/confirm",
        json={
            "correct_qr_payload": "https://tap-electric.com/c/NL-TAP-E12345",
            "charger_id": "NL-TAP-E12345",
            "confirmation_source": "operator",
            "verified": True,
        },
    )
    assert confirmation.status_code == 201
    assert confirmation.json()["training_eligible"] is True
    assert confirmation.json()["review_status"] == "verified"


def test_client_scan_id_is_idempotent(app_client: TestClient, png_bytes: bytes) -> None:
    first = app_client.post(
        "/api/v1/scans",
        data=_scan_form(),
        files={"image": ("sticker.png", png_bytes, "image/png")},
    )
    second = app_client.post(
        "/api/v1/scans",
        data=_scan_form(),
        files={"image": ("retry.png", png_bytes, "image/png")},
    )
    assert first.status_code == second.status_code == 202
    assert first.json()["scan_id"] == second.json()["scan_id"]


def test_scan_api_rejects_invalid_coordinates(app_client: TestClient, png_bytes: bytes) -> None:
    response = app_client.post(
        "/api/v1/scans",
        data=_scan_form(latitude=95),
        files={"image": ("sticker.png", png_bytes, "image/png")},
    )
    assert response.status_code == 422


def test_scan_api_rejects_fake_image(app_client: TestClient) -> None:
    response = app_client.post(
        "/api/v1/scans",
        data=_scan_form(),
        files={"image": ("fake.png", b"not an image", "image/png")},
    )
    assert response.status_code == 422


def test_duplicate_confirmation_is_rejected(app_client: TestClient, png_bytes: bytes) -> None:
    accepted = app_client.post(
        "/api/v1/scans",
        data=_scan_form(),
        files={"image": ("sticker.png", png_bytes, "image/png")},
    ).json()
    payload = {
        "correct_qr_payload": "https://tap-electric.com/c/NL-TAP-E12345",
        "charger_id": "NL-TAP-E12345",
        "verified": True,
    }
    assert (
        app_client.post(f"/api/v1/scans/{accepted['scan_id']}/confirm", json=payload).status_code
        == 201
    )
    assert (
        app_client.post(f"/api/v1/scans/{accepted['scan_id']}/confirm", json=payload).status_code
        == 409
    )
