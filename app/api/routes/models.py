from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.repositories.model_versions import ModelVersionRepository
from app.schemas.training import ModelVersionResponse

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelVersionResponse])
def list_models(db: Annotated[Session, Depends(get_db)]) -> list[object]:
    return list(ModelVersionRepository(db).list_all())
