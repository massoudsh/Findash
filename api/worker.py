from celery import Celery
from ..core.config import settings

# Initialize the Celery application
# The first argument is the name of the current module, which is '__main__' when run directly
# but will be the actual module name when imported.
# The `broker` and `backend` arguments are taken from our centralized settings.
celery_app = Celery(
    "tasks",
    broker=settings.celery.BROKER_URL,
    backend=settings.celery.RESULT_BACKEND,
    include=["api.tasks"]  # List of modules to import when the worker starts.
)

# Optional configuration
celery_app.conf.update(
    task_track_started=True,
) 