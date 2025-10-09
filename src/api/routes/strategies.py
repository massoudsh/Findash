from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from pydantic import BaseModel
from celery.result import AsyncResult
import uuid

from src.database.models import User
from ..auth.dependencies import get_current_active_user
from ...core.celery_app import run_backtest_task

router = APIRouter()
# ... existing code ... 