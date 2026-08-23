from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.db.models import ModelVersion


class ModelVersionRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, version: ModelVersion) -> ModelVersion:
        self.session.add(version)
        self.session.flush()
        return version

    def list_all(self) -> list[ModelVersion]:
        statement = select(ModelVersion).order_by(ModelVersion.created_at.desc())
        return list(self.session.scalars(statement))

    def promoted(self) -> ModelVersion | None:
        statement = (
            select(ModelVersion)
            .where(ModelVersion.promoted.is_(True))
            .order_by(ModelVersion.created_at.desc())
        )
        return self.session.scalar(statement)

    def promote(self, model: ModelVersion) -> None:
        self.session.execute(update(ModelVersion).values(promoted=False))
        model.promoted = True
        self.session.flush()
