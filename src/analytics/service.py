from src.core.celery_app import celery_app
from src.database.postgres_connection import get_db_session
from src.analytics.warehouse import FinancialDataWarehouse
import logging

logger = logging.getLogger(__name__)

@celery_app.task(name='analytics.sync_postgres_to_duckdb')
def sync_postgres_to_duckdb():
    """
    A Celery task to synchronize data from the primary PostgreSQL database
    to the DuckDB analytical data warehouse.
    """
    logger.info("Starting synchronization from PostgreSQL to DuckDB warehouse.")

    pg_session = get_db_session()
    if not pg_session:
        logger.error("Could not get a PostgreSQL session. Aborting sync.")
        return {"status": "error", "message": "DB session failed."}

    try:
        # 1. Fetch data from PostgreSQL
        # Example: Fetching the last 10,000 market data points.
        # This can be made more sophisticated (e.g., incremental updates).
        market_data_query = "SELECT symbol, price, volume, time as timestamp FROM financial_time_series ORDER BY time DESC LIMIT 10000"
        market_data = pg_session.execute(market_data_query).mappings().all()
        logger.info(f"Fetched {len(market_data)} market data records from PostgreSQL.")

        # (Add queries for other tables like risk_metrics, model_predictions here)

        # 2. Open a connection to the DuckDB warehouse
        with FinancialDataWarehouse() as dw:
            # 3. Insert data into DuckDB
            if market_data:
                # Clear existing data for a full refresh (or use incremental logic)
                dw.conn.execute("DELETE FROM market_data")
                dw.insert_market_data(market_data)
                logger.info("Successfully synchronized market data to DuckDB.")
            else:
                logger.info("No new market data to synchronize.")

        return {"status": "success", "synchronized_records": len(market_data)}

    except Exception as e:
        logger.error(f"An error occurred during data synchronization: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
    finally:
        pg_session.close()
        logger.info("PostgreSQL session closed.") 