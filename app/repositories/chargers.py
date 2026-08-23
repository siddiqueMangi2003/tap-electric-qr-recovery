from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Charger


class ChargerRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, charger_id: str) -> Charger | None:
        return self.session.get(Charger, charger_id)

    def get_by_payload(self, payload: str) -> Charger | None:
        return self.session.scalar(
            select(Charger).where(Charger.qr_payload == payload, Charger.active.is_(True))
        )

    def list_in_bounding_box(
        self,
        *,
        min_latitude: float,
        max_latitude: float,
        min_longitude: float,
        max_longitude: float,
    ) -> list[Charger]:
        statement = select(Charger).where(
            Charger.active.is_(True),
            Charger.latitude.between(min_latitude, max_latitude),
            Charger.longitude.between(min_longitude, max_longitude),
        )
        return list(self.session.scalars(statement))
