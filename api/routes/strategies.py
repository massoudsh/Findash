from fastapi import APIRouter, Depends, status
from pydantic import BaseModel
from typing import List, Dict, Any
from celery.result import AsyncResult

from database.models import User
from ..auth.dependencies import get_current_active_user
from ..tasks import run_backtest_task
from ..worker import celery_app

router = APIRouter(
    prefix="/strategies",
    tags=["Strategies"],
)

class BacktestRequest(BaseModel):
    """Defines the expected request body for a backtest."""
    tickers: List[str]
    start_date: str
    end_date: str
    capital_base: float = 100000.0
    strategy_params: Dict[str, Any]

class TaskResponse(BaseModel):
    """Defines the response when a task is submitted."""
    task_id: str
    status: str

@router.post("/backtest", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
def submit_backtest_strategy(
    request: BacktestRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Submit a strategy backtest task to the background worker.

    This endpoint is non-blocking. It immediately returns a task ID.
    Use the /results/{task_id} endpoint to check the status and retrieve results.
    """
    # Convert user.id to string to ensure it's JSON serializable for Celery
    task = run_backtest_task.delay(user_id=str(current_user.id), request_params=request.dict())
    return {"task_id": task.id, "status": "PENDING"}

@router.get("/results/{task_id}")
def get_task_status(task_id: str):
    """
    Retrieve the status and result of a background task.
    """
    task_result = AsyncResult(task_id, app=celery_app)
    
    response = {
        "task_id": task_id,
        "status": task_result.status,
        "result": None
    }
    
    if task_result.successful():
        response["result"] = task_result.get()
    elif task_result.failed():
        # Provide a structured error from the task's failure info
        response["result"] = {
            "error": "Task failed",
            "details": str(task_result.info)
        }
        
    return response 