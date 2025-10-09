from sqlalchemy.orm import Session
from uuid import UUID
from src.database.models import BacktestResult, User

def create_backtest_result(db: Session, user_id: int, task_id: str, parameters: dict, results: dict):
    db_backtest = BacktestResult(
        task_id=task_id,
        parameters=parameters,
        results=results,
        user_id=user_id
    )
    db.add(db_backtest)
    db.commit()

def get_backtest_result(db: Session, task_id: UUID, user_id: int):
    return db.query(BacktestResult).filter(BacktestResult.task_id == str(task_id), BacktestResult.user_id == user_id).first() 