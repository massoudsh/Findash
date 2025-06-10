from .worker import celery_app
from .M10_Backtesting.api_interface import run_quantum_backtest_for_api
from .database.session import SessionLocal
from .crud import backtests as crud_backtests
from .models import User  # Assuming models are accessible this way
from typing import Dict, Any

@celery_app.task(bind=True)
def run_backtest_task(self, user_id: str, request_params: Dict[str, Any]):
    """
    Celery task to run a backtest asynchronously.
    """
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            self.update_state(state='FAILURE', meta={'exc_type': 'UserNotFound', 'exc_message': 'User not found'})
            return

        # Run the computationally intensive backtest
        result = run_quantum_backtest_for_api(
            tickers=request_params["tickers"],
            start_date=request_params["start_date"],
            end_date=request_params["end_date"],
            capital_base=request_params["capital_base"],
            strategy_params=request_params["strategy_params"]
        )

        if "error" in result:
            self.update_state(state='FAILURE', meta={'exc_type': 'BacktestError', 'exc_message': result["error"]})
            return

        # Save the result to the database
        crud_backtests.create_backtest_result(
            db=db,
            user=user,
            backtest_data=result,
            request_params=request_params
        )

        return result
    finally:
        db.close() 