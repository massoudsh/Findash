from src.core.celery_app import celery_app
from .. import unsloth_training

@celery_app.task(name="llm.trigger_unsloth_finetuning")
def trigger_unsloth_finetuning(model_name: str, output_dir: str):
    """
    Placeholder Celery task to start an Unsloth fine-tuning job.
    """
    print(f"Starting Unsloth fine-tuning for {model_name}.")
    try:
        unsloth_training.run_unsloth_training()
        print("Unsloth fine-tuning job finished successfully.")
        return {"status": "success", "output_dir": output_dir}
    except Exception as e:
        print(f"Error during Unsloth fine-tuning job: {e}")
        return {"status": "error", "error_message": str(e)} 