from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.db.models import ReviewStatus, Scan, ScanLabel


class ScanRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, scan: Scan) -> Scan:
        self.session.add(scan)
        self.session.flush()
        return scan

    def get(self, scan_id: str, *, with_label: bool = False) -> Scan | None:
        statement = select(Scan).where(Scan.id == scan_id)
        if with_label:
            statement = statement.options(joinedload(Scan.label))
        return self.session.scalar(statement)

    def get_by_client_id(self, client_scan_id: str) -> Scan | None:
        return self.session.scalar(select(Scan).where(Scan.client_scan_id == client_scan_id))

    def add_label(
        self,
        scan: Scan,
        *,
        correct_qr_payload: str,
        charger_id: str,
        confirmation_source: str,
        confirmed_by: str | None,
        verified: bool,
    ) -> ScanLabel:
        label = ScanLabel(
            scan_id=scan.id,
            correct_qr_payload=correct_qr_payload,
            charger_id=charger_id,
            confirmation_source=confirmation_source,
            confirmed_by=confirmed_by,
            review_status=(ReviewStatus.VERIFIED.value if verified else ReviewStatus.PENDING.value),
            training_eligible=verified,
        )
        self.session.add(label)
        self.session.flush()
        return label

    def list_training_eligible(self) -> list[Scan]:
        statement = (
            select(Scan)
            .join(ScanLabel)
            .where(ScanLabel.training_eligible.is_(True))
            .options(joinedload(Scan.label))
            .order_by(Scan.created_at, Scan.id)
        )
        return list(self.session.scalars(statement).unique())
