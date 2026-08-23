from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.db.models import TrainingRun
from app.schemas.training import TrainingRunRequest, TrainingRunResponse

router = APIRouter(prefix="/training", tags=["training"])


@router.post("/run", response_model=TrainingRunResponse, status_code=202)
def start_training(
    payload: TrainingRunRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Annotated[Session, Depends(get_db)],
) -> TrainingRun:
    training_run = TrainingRun(dry_run=payload.dry_run)
    db.add(training_run)
    db.commit()
    db.refresh(training_run)
    background_tasks.add_task(
        request.app.state.training_service.run, training_run.id, epochs=payload.epochs
    )
    return training_run


@router.get("/runs/{training_run_id}", response_model=TrainingRunResponse)
def get_training_run(training_run_id: str, db: Annotated[Session, Depends(get_db)]) -> TrainingRun:
    training_run = db.get(TrainingRun, training_run_id)
    if training_run is None:
        raise HTTPException(status_code=404, detail="Training run not found")
    return training_run
